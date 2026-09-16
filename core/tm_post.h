/* 端上的「稳定版 v2」后处理：跟 label_service/postprocess.py 的
 * stabilize(..., algo="viterbi") 同一套规则，改写成**流式**的。
 *
 * 为什么板子上要有这个：现在的 tinyml_task 只看每个窗口的 argmax，
 * 连续两三个窗口判成抓挠就报一次。而模型逐窗口的输出是抖的——
 * 服务端那份后处理之所以效果好，靠的就是把整条时间轴放在一起看：
 * 切换类别要付代价（viterbi）、间隔几秒内的合成一段、太短太弱的丢掉。
 * 板上不做这一步，上报的次数和时长跟平台上看到的对不上，而两边用的
 * 是同一个模型——差异全在后处理。
 *
 * ── 跟服务端那份的唯一差别：有界前瞻 ─────────────────────────────────
 *
 * Python 那份是**离线**的：viterbi 在整条序列上做 DP，再从最后一个窗口
 * 回溯。板上一天几万个窗口，既没有那么多 RAM，也不能等到第二天才报。
 *
 * 这里用有界回溯：留 TM_POST_LOOKAHEAD 个窗口的回溯指针，每来一个窗口
 * 就看所有状态的回溯链有没有汇合到同一个祖先——汇合了，那之前的部分
 * 就**再也不会变**，可以定稿。viterbi 的路径汇合得很快（切换代价越大
 * 越快），所以绝大多数情况下汇合发生在几个窗口之内。
 *
 * 缓冲填满还没汇合时按当前最优状态强制定稿。这种情况下**可能**跟离线
 * 结果不同——不是"大概不会"，是 tests/test_post_c.py 在真实数据上量了：
 * 见那里的数字。量不出来就不该说"等价"。
 *
 * 没有 malloc，所有状态在 tm_post_t 里，编译期定大小。
 */

#ifndef TM_POST_H
#define TM_POST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 类别数上限。现在的模型是 5 类（活动/睡觉/抓挠/未佩戴/甩身体）。
 * 主要吃 RAM 的是延迟线里的概率（DELAY × MAX_CLASSES × 4 字节），
 * 所以这里按实际类别数配，不留富余。
 * 配小了**不会算错**：tm_post_on_window 直接返回 -1。 */
#ifndef TM_POST_MAX_CLASSES
#define TM_POST_MAX_CLASSES 5
#endif

/* viterbi 回溯缓冲：路径没汇合时最多往回看这么多个窗口。
 * 调大只花 RAM（LOOKAHEAD × MAX_CLASSES 字节），不花时间——
 * 汇合检测是从头往回走，汇合了就停。 */
#ifndef TM_POST_LOOKAHEAD
#define TM_POST_LOOKAHEAD 48
#endif

/* 事件组装的延迟线：一个"安全块"最多能有多长。
 * 必须 ≥ ceil((event_gap_s + shake_absorb_s) / stride_s) + 4，
 * tm_post_on_window 会检查，不够就返回 -1——**不是截断**。
 * 截断的表现是间隔大一点的两段不合并了，而那不会报错。
 *
 * 为什么是 192：事件密集时块装不下会被强制切开，而切开的地方可能跟
 * 服务端分家。在 18 万个窗口上扫过一遍（tests/test_post_c.py）：
 *     DELAY= 64  片段一致 99.92%   强制切块 59 次
 *     DELAY= 96  片段一致 100%     强制切块  4 次
 *     DELAY=192  片段一致 100%     强制切块  0 次
 * 192 × 0.5s = 96 秒，也就是"连续事件 96 秒以内保证跟服务端逐段一致"。
 * 代价是 RAM：延迟线 ≈ 192 × (5×4 + 4 + 5) = 5.6 KB。 */
#ifndef TM_POST_DELAY
#define TM_POST_DELAY 192
#endif

/* 一个块里最多几段 bout。块最长 TM_POST_DELAY 个窗口，
 * 一段至少占一个窗口、段间至少隔一个窗口，所以一半足够。 */
#ifndef TM_POST_MAX_BOUTS
#define TM_POST_MAX_BOUTS (TM_POST_DELAY / 2 + 1)
#endif

typedef struct {
    int   n_classes;
    /* 事件类别（抓挠/甩身体）。其余是状态类别（活动/睡觉/未佩戴）。
     * 事件和状态的处理方式完全不同：状态是时间轴的底色，事件是盖在上面的。 */
    uint8_t is_event[TM_POST_MAX_CLASSES];
    int   scratch;              /* 抓挠的类别号，-1 = 没有 */
    int   shake;                /* 甩身体的类别号，-1 = 没有 */

    float viterbi_switch;       /* 切换类别的代价（对数单位），默认 3.0 */
    float event_gap_s;          /* 同类事件间隔多久以内合成一段，默认 4 */
    float shake_absorb_s;       /* 抓挠前后多少秒内的甩身体并进来，默认 3 */
    int   event_min_windows;    /* 一段至少几个窗口，默认 2 */
    float event_min_mean;       /* 段内平均概率下限，默认 0.45 */
    float event_single_conf;    /* 单窗口也留的最高概率下限，默认 0.85 */

    float window_s;             /* 窗口长度（秒） */
    float stride_s;             /* 步长（秒） */
} tm_post_cfg_t;

/* 配好 5 类模型的默认参数，值跟 label_service/config.py 的 STABLE_* 一致。
 * classes 里找不到抓挠/甩身体时对应的 scratch/shake 置 -1，那两步自动跳过。 */
void tm_post_cfg_default(tm_post_cfg_t *cfg, int n_classes,
                         int scratch_class, int shake_class,
                         float window_s, float stride_s);

/* 一个片段。时间是调用方传进来的毫秒数，板上是开机以来的 tick。 */
typedef struct {
    int      cls;
    uint32_t start_ms;
    uint32_t end_ms;
    uint16_t n_windows;
    float    conf_max;
    float    conf_mean;
} tm_seg_t;

typedef struct {
    tm_post_cfg_t cfg;

    /* ── viterbi ── */
    float   score[TM_POST_MAX_CLASSES];
    uint8_t bp[TM_POST_LOOKAHEAD][TM_POST_MAX_CLASSES];
    /* 回溯缓冲里当前有几步（bp 的有效长度），以及环形起点 */
    int     bp_n;
    int     bp_head;
    int     started;

    /* ── 延迟线：解码结果 + 概率，等事件组装定稿 ── */
    uint32_t dl_ms[TM_POST_DELAY];
    float    dl_p[TM_POST_DELAY][TM_POST_MAX_CLASSES];
    uint8_t  dl_dec[TM_POST_DELAY];     /* viterbi 解出来的类别 */
    uint8_t  dl_arg[TM_POST_DELAY];     /* 逐窗口 argmax（**没经过 viterbi**） */
    uint8_t  dl_final[TM_POST_DELAY];   /* 事件盖上去之后的最终类别 */
    uint8_t  dl_fixed[TM_POST_DELAY];   /* 解码定了没 */
    uint8_t  dl_taken[TM_POST_DELAY];   /* 被抓挠 bout 占掉了（Python 的 taken） */
    uint8_t  dl_done[TM_POST_DELAY];    /* 事件组装过了，final 可以吐了 */
    int      dl_n;
    int      dl_head;

    /* ── 当前这个"安全块" ── */
    long     chunk_lo;                  /* -1 = 还没开始 */
    long     quiet;                     /* 连续多少个"安静"窗口 */
    long     last_quiet;                /* 最近一个安静窗口的绝对序号，-1 = 没有 */

    /* ── 正在攒的输出片段 ── */
    int      run_cls;
    uint32_t run_start_ms, run_end_ms;
    uint16_t run_n;
    float    run_cmax, run_csum;

    long     seq;                       /* 进来过多少个窗口 */
    long     emitted;                   /* 延迟线已经吐掉到哪个绝对序号 */
    int      last_state;                /* 最近一次见到的状态类别 */

    /* 诊断：回溯缓冲满了还没汇合的次数。**要能读出来**——
     * 它就是"这份端上结果可能跟服务端不同"的次数 */
    uint32_t forced;
    /* 诊断：事件密集到块装不下、被迫从中间切开的次数。
     * 同样是"可能跟服务端不同"的计数 */
    uint32_t forced_split;
} tm_post_t;

void tm_post_init(tm_post_t *st, const tm_post_cfg_t *cfg);

/* 一个窗口的概率进来（probs[n_classes]，和为 1）。
 * 返回这次定稿了几个片段，写进 out[0..max_out)。
 * 返回 -1 = 配置不合法（延迟线不够长），这时候什么都没做。 */
int tm_post_on_window(tm_post_t *st, const float *probs, uint32_t t_ms,
                      tm_seg_t *out, int max_out);

/* 数据到头了（文件结束 / 设备要休眠）：把还压着的全部吐出来。 */
int tm_post_flush(tm_post_t *st, tm_seg_t *out, int max_out);

#ifdef __cplusplus
}
#endif

#endif /* TM_POST_H */
