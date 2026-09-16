#include "tm_post.h"

#include <stdint.h>
#include <string.h>

/* 跟 Python 那份的 eps 一样：log(max(1e-6, p))。
 * 这个钳位不是"数值保险"，它会改结果——概率低于 1e-6 的类别一律按 1e-6 算，
 * 也就是"再不可能也不会比这更不可能"。两边必须用同一个值，否则
 * 一个极端窗口就能让路径分叉。 */
#define TM_POST_EPS 1e-6f


/* ── 自己算 log，不用 libm ─────────────────────────────────────────────
 *
 * 为什么不用 logf()：**它在两个平台上不是同一份实现**。板上是 newlib，
 * PC 上（服务的 @board 模式、所有对照测试）是 glibc，末位可能不一样。
 * 而这个值是 viterbi 的发射项，末位不同就可能在某个接近的地方把路径翻过去。
 *
 * 那点差别多半永远碰不上——但"多半"没法验证，而这里有更省事的办法：
 * 只用 IEEE-754 的浮点加减乘除。那些在两个平台上是**逐位确定**的
 * （配合 -ffp-contract=off 禁掉 FMA 合并），于是两边算出来一模一样，
 * 不是"应该一样"，是同一个函数同一套运算。
 *
 * 做法是标准的：x = m·2^e，m∈[1,2)；log(x) = e·ln2 + log(m)。
 * 把 m 规到 [√½, √2) 之后用 atanh 级数，s=(m-1)/(m+1)，|s|≤0.1716，
 * 截到 s⁹ 项的截断误差约 s¹¹/11 ≈ 2e-10，远在 float 精度之下。
 */
static float tm_log(float x)
{
    union { float f; uint32_t u; } v;
    v.f = x;
    int e = (int)((v.u >> 23) & 0xFFu) - 127;
    /* 尾数拼回 [1,2) */
    v.u = (v.u & 0x007FFFFFu) | 0x3F800000u;
    float m = v.f;
    if (m > 1.41421356f) { m *= 0.5f; e += 1; }

    const float s = (m - 1.0f) / (m + 1.0f);
    const float s2 = s * s;
    /* 2s·(1 + s²/3 + s⁴/5 + s⁶/7 + s⁸/9) */
    const float poly = 1.0f + s2 * (0.333333343f + s2 * (0.200000003f
                     + s2 * (0.142857149f + s2 * 0.111111112f)));
    /* ln2 拆成高低两半：e 最大到 ±127，单精度一次乘会把低位丢光 */
    const float LN2_HI = 0.693145752f;      /* 前 12 位尾数，末尾是 0 */
    const float LN2_LO = 1.42860677e-06f;   /* 剩下的部分 */
    const float ef = (float)e;
    return ef * LN2_HI + (2.0f * s * poly + ef * LN2_LO);
}

void tm_post_cfg_default(tm_post_cfg_t *cfg, int n_classes,
                         int scratch_class, int shake_class,
                         float window_s, float stride_s)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_classes = n_classes;
    cfg->scratch = scratch_class;
    cfg->shake = shake_class;
    if (scratch_class >= 0 && scratch_class < TM_POST_MAX_CLASSES)
        cfg->is_event[scratch_class] = 1;
    if (shake_class >= 0 && shake_class < TM_POST_MAX_CLASSES)
        cfg->is_event[shake_class] = 1;
    /* 这些默认值跟 label_service/config.py 的 STABLE_* 逐个对应。
     * 改这里之前先改那边——服务端是基准，端上是跟着走的那个。 */
    cfg->viterbi_switch    = 3.0f;
    cfg->event_gap_s       = 4.0f;
    cfg->shake_absorb_s    = 3.0f;
    cfg->event_min_windows = 2;
    cfg->event_min_mean    = 0.45f;
    cfg->event_single_conf = 0.85f;
    cfg->window_s = window_s;
    cfg->stride_s = stride_s;
}

void tm_post_init(tm_post_t *st, const tm_post_cfg_t *cfg)
{
    memset(st, 0, sizeof(*st));
    st->cfg = *cfg;
    st->chunk_lo = -1;
    st->last_quiet = -1;
    st->run_cls = -1;
    st->last_state = -1;
}

/* 延迟线是环形的；abs 是绝对序号，转成槽位 */
static int slot_of(const tm_post_t *st, long abs_i)
{
    long rel = abs_i - (st->seq - st->dl_n);
    if (rel < 0 || rel >= st->dl_n) return -1;
    return (int)((st->dl_head + rel) % TM_POST_DELAY);
}

static float prob_at(const tm_post_t *st, long abs_i, int cls)
{
    const int s = slot_of(st, abs_i);
    if (s < 0) return 0.0f;
    return st->dl_p[s][cls];
}

static int dec_at(const tm_post_t *st, long abs_i)
{
    const int s = slot_of(st, abs_i);
    if (s < 0) return -1;
    return st->dl_dec[s];
}

/* 这个窗口算不算"甩身体"——**argmax 或 解码结果，两个都算**。
 *
 * Python 那边是 `raw_label[i] == sh or decoded[i] == sh`，raw_label 是
 * 逐窗口 argmax，没经过 viterbi。只看解码结果的话，一个 argmax 是甩身体、
 * 但被 viterbi 归成活动的窗口就吞不进来，抓挠段短一个窗口。
 * 实测 3000 个窗口里就这么差了几段出来——段的类别全对，只是短一格。 */
static int is_shake_like(const tm_post_t *st, long abs_i)
{
    const int s = slot_of(st, abs_i);
    if (s < 0 || st->cfg.shake < 0) return 0;
    return st->dl_dec[s] == (uint8_t)st->cfg.shake
           || st->dl_arg[s] == (uint8_t)st->cfg.shake;
}

static uint32_t ms_at(const tm_post_t *st, long abs_i)
{
    const int s = slot_of(st, abs_i);
    if (s < 0) return 0;
    return st->dl_ms[s];
}

/* 窗口 k 在时间轴上负责的区间 [start, end)，跟 postprocess._zones 的
 * label_mode != "center" 那一支一致：end = 下一个窗口的 ts，
 * 最后一个窗口 end = ts + window_s。
 *
 * 端上是流式的，"下一个窗口的 ts"要么已经在延迟线里，要么还没来。
 * 还没来的时候用 ts + stride_s 顶上——**这跟离线那份的唯一差别**，
 * 而且只影响最后一个窗口的结束时间。 */
static uint32_t zone_start(const tm_post_t *st, long abs_i)
{
    return ms_at(st, abs_i);
}

static uint32_t zone_end(const tm_post_t *st, long abs_i)
{
    const int s = slot_of(st, abs_i + 1);
    if (s >= 0) return st->dl_ms[s];
    return ms_at(st, abs_i) + (uint32_t)(st->cfg.window_s * 1000.0f + 0.5f);
}

/* 两个窗口之间的间隔秒数：zones[j][0] - zones[i][1] */
static float gap_s(const tm_post_t *st, long i, long j)
{
    const uint32_t a = zone_end(st, i), b = zone_start(st, j);
    /* 无符号相减：b < a（窗口重叠，stride < window 时正常）要算成 0 或负，
     * 不能绕回成 40 亿毫秒——那会让"隔了很久"变成"紧挨着"，两段莫名合并 */
    if (b <= a) return -(float)(a - b) / 1000.0f;
    return (float)(b - a) / 1000.0f;
}

/* ── viterbi 一步 ──────────────────────────────────────────────────────
 * 跟 postprocess._viterbi 同一个递推：
 *     stay   = score[c]
 *     switch = score[best_prev] - switch_cost
 *     new[c] = max(stay, switch) + emit[c]     （相等时取 stay，跟 Python 的 >= 一致）
 */
static void viterbi_step(tm_post_t *st, const float *emit, uint8_t *bp_out)
{
    const int m = st->cfg.n_classes;
    float ns[TM_POST_MAX_CLASSES];
    int best_prev = 0;
    for (int j = 1; j < m; ++j)
        if (st->score[j] > st->score[best_prev]) best_prev = j;

    const float sw = st->score[best_prev] - st->cfg.viterbi_switch;
    for (int c = 0; c < m; ++c) {
        if (st->score[c] >= sw) {
            ns[c] = st->score[c] + emit[c];
            bp_out[c] = (uint8_t)c;
        } else {
            ns[c] = sw + emit[c];
            bp_out[c] = (uint8_t)best_prev;
        }
    }
    memcpy(st->score, ns, sizeof(float) * (size_t)m);
}

/* 回溯缓冲里第 k 步（0 = 最早的一步）的指针数组 */
static const uint8_t *bp_at(const tm_post_t *st, int k)
{
    return st->bp[(st->bp_head + k) % TM_POST_LOOKAHEAD];
}

/* 从最新一步往回走，看所有状态的链在第几步汇合。
 * 返回汇合处之前**可以定稿**的步数（0 = 还不能定）。 */
static int viterbi_settled(const tm_post_t *st, int *first_label)
{
    const int m = st->cfg.n_classes;
    int cur[TM_POST_MAX_CLASSES];
    for (int c = 0; c < m; ++c) cur[c] = c;

    /* 从最新一步往回走到最早一步 */
    for (int k = st->bp_n - 1; k >= 0; --k) {
        const uint8_t *bp = bp_at(st, k);
        int same = 1;
        for (int c = 0; c < m; ++c) {
            cur[c] = bp[cur[c]];
            if (cur[c] != cur[0]) same = 0;
        }
        if (same) {
            /* 第 k 步的所有链都指向 cur[0]，也就是第 k 步**之前**那个窗口的
             * 标签已经确定是 cur[0]，且第 0..k-1 步也都确定了 */
            *first_label = cur[0];
            return k;   /* 可以定稿 k 个（第 0..k-1 步对应的窗口） + 这一个 */
        }
    }
    return -1;
}

/* ── 事件组装 ──────────────────────────────────────────────────────────
 *
 * 这一段是 postprocess.stabilize 事件部分的搬运，**按"安全块"整块处理**，
 * 不是一个窗口一个窗口地增量攒。
 *
 * 为什么不能增量攒：Python 那边每个事件类别**各自独立**地建 bout 再按
 * gap_s 合并，两个类别的 bout 是可以互相跨越的——一段甩身体 bout 完全
 * 可以横跨中间那几个抓挠窗口。按时间顺序边走边收的写法做不到这一点：
 * 抓挠窗口一来就把甩身体那段收了，于是本该合并的两截甩身体变成两段，
 * 抓挠往后吞并时又还看不到后面的甩身体（前瞻不够）。
 * 这不是少了个 if，是结构不对。实测就是这里差出来的：应该 8 个窗口的
 * 抓挠段变成 7 个。
 *
 * 安全块：连续出现 ≥ (gap_s + absorb_s)/stride 个非事件窗口的地方，
 * 一定不会有 bout 跨过去（bout 最多跨 gap_s，吞并最多再伸 absorb_s）。
 * 所以在那里切一刀，前面那块可以完全按离线算法处理，结果跟离线**一样**。
 *
 * 块一直不结束时（事件密集到延迟线装满）强制切，计数在 st->forced_split。
 */

/* 一段 bout。trim=1 表示要跳过已经被抓挠占掉的窗口（Python 里那个 taken）。 */
typedef struct { long i0, i1; } bout_t;

static int is_taken(const tm_post_t *st, long i)
{
    const int s = slot_of(st, i);
    return s >= 0 && st->dl_taken[s];
}

static void mark_taken(tm_post_t *st, long i0, long i1)
{
    for (long i = i0; i <= i1; ++i) {
        const int s = slot_of(st, i);
        if (s >= 0) st->dl_taken[s] = 1;
    }
}

/* 把 [lo,hi] 里 decoded==cls 的窗口按 gap_s 合并成 bout，写进 b[]，返回条数。
 * 跟 _bouts + _merge_gaps 合起来是一回事：单个窗口各自成 bout，再合并。 */
static int build_bouts(const tm_post_t *st, long lo, long hi, int cls,
                       bout_t *b, int max_b)
{
    int n = 0;
    for (long i = lo; i <= hi; ++i) {
        if (dec_at(st, i) != cls) continue;
        if (n > 0 && gap_s(st, b[n - 1].i1, i) <= st->cfg.event_gap_s) {
            b[n - 1].i1 = i;
        } else if (n < max_b) {
            b[n].i0 = b[n].i1 = i;
            ++n;
        }
    }
    return n;
}

/* 门槛过滤。skip_taken 时按 Python 那样只看没被抓挠占掉的窗口——
 * 数量和均值都要用**裁剪之后**的那组算，不然一段几乎全被抓挠吞掉的
 * 甩身体还会因为原来的长度活下来。 */
static int bout_keep(const tm_post_t *st, long i0, long i1, int cls, int skip_taken)
{
    float sum = 0.0f, mx = 0.0f;
    int n = 0;
    for (long i = i0; i <= i1; ++i) {
        if (skip_taken && is_taken(st, i)) continue;
        const float v = prob_at(st, i, cls);
        sum += v;
        if (v > mx) mx = v;
        ++n;
    }
    if (n == 0) return 0;
    const float mean = sum / (float)n;
    return (n >= st->cfg.event_min_windows && mean >= st->cfg.event_min_mean)
           || mx >= st->cfg.event_single_conf;
}

static void bout_paint(tm_post_t *st, long i0, long i1, int cls, int skip_taken)
{
    const int is_scratch = (cls == st->cfg.scratch);
    for (long i = i0; i <= i1; ++i) {
        if (skip_taken && is_taken(st, i)) continue;
        const int s = slot_of(st, i);
        if (s < 0) continue;
        /* 跟 Python 的 `if final[i] not in events or ev == sl` 一致：
         * 抓挠盖得过甩身体，反过来不行 */
        if (!st->cfg.is_event[st->dl_final[s]] || is_scratch)
            st->dl_final[s] = (uint8_t)cls;
    }
}

/* 抓挠 bout 往前后吞掉 absorb_s 以内的甩身体窗口（限制在块内） */
static void absorb_shake(const tm_post_t *st, long *i0, long *i1, long lo, long hi)
{
    if (st->cfg.shake < 0) return;
    while (*i0 - 1 >= lo && is_shake_like(st, *i0 - 1)
           && gap_s(st, *i0 - 1, *i0) <= st->cfg.shake_absorb_s)
        --*i0;
    while (*i1 + 1 <= hi && is_shake_like(st, *i1 + 1)
           && gap_s(st, *i1, *i1 + 1) <= st->cfg.shake_absorb_s)
        ++*i1;
}

static void process_chunk(tm_post_t *st, long lo, long hi)
{
    if (hi < lo) return;
    bout_t b[TM_POST_MAX_BOUTS];

    /* ① 抓挠：建 bout → 吞并甩身体 → 再合并一次 → 过滤 → 盖上去 */
    const int sl = st->cfg.scratch;
    if (sl >= 0) {
        const int n = build_bouts(st, lo, hi, sl, b, TM_POST_MAX_BOUTS);
        /* 吞并之后要重新合并：吞出来的两段可能就挨上了（Python 里
         * grown 之后又调了一次 _merge_gaps） */
        int m = 0;
        for (int k = 0; k < n; ++k) {
            long a = b[k].i0, c = b[k].i1;
            absorb_shake(st, &a, &c, lo, hi);
            if (m > 0 && gap_s(st, b[m - 1].i1, a) <= st->cfg.event_gap_s)
                b[m - 1].i1 = c;
            else { b[m].i0 = a; b[m].i1 = c; ++m; }
        }
        /* taken 是**过滤之前**的全部抓挠 bout 窗口——Python 就是在
         * 过滤之前取的。被过滤掉的抓挠段也会把甩身体挡住，
         * 这看着别扭，但两边必须一致 */
        for (int k = 0; k < m; ++k) mark_taken(st, b[k].i0, b[k].i1);
        for (int k = 0; k < m; ++k)
            if (bout_keep(st, b[k].i0, b[k].i1, sl, 0))
                bout_paint(st, b[k].i0, b[k].i1, sl, 0);
    }

    /* ② 甩身体：建 bout → 去掉被抓挠占掉的窗口 → 过滤 → 盖上去 */
    const int sh = st->cfg.shake;
    if (sh >= 0 && sh != sl) {
        const int n = build_bouts(st, lo, hi, sh, b, TM_POST_MAX_BOUTS);
        for (int k = 0; k < n; ++k)
            if (bout_keep(st, b[k].i0, b[k].i1, sh, 1))
                bout_paint(st, b[k].i0, b[k].i1, sh, 1);
    }

    /* 块里的窗口 final 定了，可以吐 */
    for (long i = lo; i <= hi; ++i) {
        const int s = slot_of(st, i);
        if (s >= 0) st->dl_done[s] = 1;
    }
}

/* 把绝对序号 i 的窗口吐成片段 */
static int emit_one(tm_post_t *st, long i, tm_seg_t *out, int max_out, int n_out)
{
    const int s = slot_of(st, i);
    if (s < 0) return n_out;
    const int lab = st->dl_final[s];

    if (lab == st->run_cls) {
        st->run_end_ms = zone_end(st, i);
        st->run_n++;
        const float v = prob_at(st, i, lab);
        st->run_csum += v;
        if (v > st->run_cmax) st->run_cmax = v;
        return n_out;
    }
    if (st->run_cls >= 0 && n_out < max_out) {
        out[n_out].cls = st->run_cls;
        out[n_out].start_ms = st->run_start_ms;
        out[n_out].end_ms = st->run_end_ms;
        out[n_out].n_windows = st->run_n;
        out[n_out].conf_max = st->run_cmax;
        out[n_out].conf_mean = st->run_n ? st->run_csum / (float)st->run_n : 0.0f;
        ++n_out;
    }
    st->run_cls = lab;
    st->run_start_ms = zone_start(st, i);
    st->run_end_ms = zone_end(st, i);
    st->run_n = 1;
    st->run_cmax = st->run_csum = prob_at(st, i, lab);
    return n_out;
}

static int drain(tm_post_t *st, tm_seg_t *out, int max_out)
{
    int n_out = 0;
    while (st->emitted < st->seq) {
        const int s = slot_of(st, st->emitted);
        if (s < 0) { ++st->emitted; continue; }
        if (!st->dl_done[s]) break;
        n_out = emit_one(st, st->emitted, out, max_out, n_out);
        ++st->emitted;
    }
    return n_out;
}

/* 状态底色：最近一次见到的状态类别。
 *
 * Python 那边开头那几个窗口用的是**整条序列里第一个**状态类别
 * （`next((d for d in decoded if d in states), ...)`）——那是往后看的。
 * 流式看不到后面，所以先记个待定，等第一个状态出现再回填。
 * 不回填的话开头几个窗口的状态是瞎猜的，而那正是每次开机后的头几秒。 */
#define TM_POST_PENDING 0xFEu

static void set_state_base(tm_post_t *st, long i, int dec)
{
    const int s = slot_of(st, i);
    if (s < 0) return;
    if (!st->cfg.is_event[dec]) {
        if (st->last_state < 0) {
            st->last_state = dec;
            /* 回填前面那些待定的 */
            for (long j = st->seq - st->dl_n; j < i; ++j) {
                const int t = slot_of(st, j);
                if (t >= 0 && st->dl_final[t] == TM_POST_PENDING)
                    st->dl_final[t] = (uint8_t)dec;
            }
        } else {
            st->last_state = dec;
        }
    }
    st->dl_final[s] = (uint8_t)(st->last_state >= 0 ? st->last_state
                                                    : (int)TM_POST_PENDING);
}

static int cfg_ok(const tm_post_t *st)
{
    const float need = (st->cfg.event_gap_s + st->cfg.shake_absorb_s)
                       / (st->cfg.stride_s > 0 ? st->cfg.stride_s : 1.0f);
    return st->cfg.n_classes > 0
           && st->cfg.n_classes <= TM_POST_MAX_CLASSES
           && st->cfg.stride_s > 0
           && (int)(need + 0.9999f) + 4 <= TM_POST_DELAY;
}

/* 安全块边界：连续多少个非事件窗口之后可以切一刀 */
static long safe_run_needed(const tm_post_t *st)
{
    const float s = (st->cfg.event_gap_s + st->cfg.shake_absorb_s) / st->cfg.stride_s;
    return (long)(s + 0.9999f) + 1;
}

/* 一个窗口的解码结果定下来了 */
static void on_decoded(tm_post_t *st, long i, int dec)
{
    const int s = slot_of(st, i);
    if (s < 0) return;
    st->dl_dec[s] = (uint8_t)dec;
    st->dl_taken[s] = 0;
    st->dl_done[s] = 0;
    set_state_base(st, i, dec);
    st->dl_fixed[s] = 1;

    /* "安静"要求这个窗口**既不是事件、argmax 也不是甩身体**。
     *
     * 只看"不是事件"是不够的：往前吞并是沿着连续的甩身体窗口走的
     * （相邻窗口间隔 0 秒，absorb_s 这个时间预算根本限制不住它），
     * 而"argmax 是甩身体、但被 viterbi 归成别的"的窗口可以出现在
     * 一段安静里。块边界切在那种地方，下一块开头的抓挠段就吞不到
     * 前一块尾巴上的那个窗口——段起点晚一格，别的全对。 */
    if (st->cfg.is_event[dec] || is_shake_like(st, i)) {
        st->quiet = 0;
    } else {
        st->quiet++;
        st->last_quiet = i;
    }

    if (st->chunk_lo < 0) st->chunk_lo = i;

    /* 安静够久 → 前面整块可以按离线算法处理 */
    if (st->quiet >= safe_run_needed(st)) {
        process_chunk(st, st->chunk_lo, i);
        st->chunk_lo = i + 1;
        st->quiet = 0;
    } else if (i - st->chunk_lo + 1 >= TM_POST_DELAY - 4) {
        /* 事件密集到延迟线要装不下了：只能强制切。
         * **这里可能跟离线不同**（一个 bout 被从中间切开），所以计数。
         *
         * 切在哪也有讲究：能切在最近一个安静窗口上就切在那儿，
         * 至少"往前吞并甩身体"那一步不会被切断（吞并是沿着连续的
         * 甩身体窗口走的，安静窗口本来就会让它停）。
         * 随便切在当前窗口的话，切点很可能正落在一段事件中间。 */
        long cut = i;
        if (st->last_quiet > st->chunk_lo && st->last_quiet < i)
            cut = st->last_quiet;
        st->forced_split++;
        process_chunk(st, st->chunk_lo, cut);
        st->chunk_lo = cut + 1;
        st->quiet = 0;
    }
}

int tm_post_on_window(tm_post_t *st, const float *probs, uint32_t t_ms,
                      tm_seg_t *out, int max_out)
{
    if (!cfg_ok(st)) return -1;
    const int m = st->cfg.n_classes;

    /* 1. 进延迟线 */
    if (st->dl_n == TM_POST_DELAY) {
        st->dl_head = (st->dl_head + 1) % TM_POST_DELAY;
        st->dl_n--;
    }
    const int slot = (st->dl_head + st->dl_n) % TM_POST_DELAY;
    st->dl_ms[slot] = t_ms;
    int arg = 0;
    for (int c = 0; c < m; ++c) {
        st->dl_p[slot][c] = probs[c];
        /* 严格大于：并列时取下标小的，跟 numpy argmax 一致 */
        if (probs[c] > probs[arg]) arg = c;
    }
    st->dl_arg[slot] = (uint8_t)arg;
    st->dl_dec[slot] = 0;
    st->dl_final[slot] = TM_POST_PENDING;
    st->dl_fixed[slot] = 0;
    st->dl_taken[slot] = 0;
    st->dl_done[slot] = 0;
    st->dl_n++;
    const long my_i = st->seq;
    st->seq++;

    /* 2. viterbi 递推。emit = log(max(eps, p))，跟 Python 一模一样 */
    float emit[TM_POST_MAX_CLASSES];
    for (int c = 0; c < m; ++c) {
        const float p = probs[c] > TM_POST_EPS ? probs[c] : TM_POST_EPS;
        emit[c] = tm_log(p);
    }
    if (!st->started) {
        memcpy(st->score, emit, sizeof(float) * (size_t)m);
        st->started = 1;
        return 0;   /* 第一个窗口没有回溯指针，标签由后面决定 */
    }

    uint8_t bp_tmp[TM_POST_MAX_CLASSES];
    viterbi_step(st, emit, bp_tmp);

    if (st->bp_n == TM_POST_LOOKAHEAD) {
        st->forced++;
        int best = 0;
        for (int j = 1; j < m; ++j) if (st->score[j] > st->score[best]) best = j;
        /* **下标约定跟"汇合定稿"那段不同**：bp_tmp 还没入队，队里的
         * bp_n 个指针对应窗口 my_i-bp_n .. my_i-1，而 best 是窗口 my_i 的
         * 标签。先用 bp_tmp 退一格到 my_i-1，再沿队列往回走。
         * 少这一格的话整条链错位一个窗口 */
        int cc = bp_tmp[best];
        for (int k = st->bp_n - 1; k >= 1; --k) cc = bp_at(st, k)[cc];
        on_decoded(st, my_i - st->bp_n, cc);
        st->bp_head = (st->bp_head + 1) % TM_POST_LOOKAHEAD;
        st->bp_n--;
    }
    memcpy(st->bp[(st->bp_head + st->bp_n) % TM_POST_LOOKAHEAD], bp_tmp,
           sizeof(uint8_t) * (size_t)m);
    st->bp_n++;

    /* 3. 路径汇合了就定稿 */
    int lab0 = -1;
    const int k = viterbi_settled(st, &lab0);
    if (k > 0) {
        const long settle_i = my_i - st->bp_n + k;
        int labels[TM_POST_LOOKAHEAD + 1];
        long idxs[TM_POST_LOOKAHEAD + 1];
        int cnt = 0;
        int cc = lab0;
        long idx = settle_i;
        while (cnt <= TM_POST_LOOKAHEAD) {
            const int s = slot_of(st, idx);
            if (s < 0 || st->dl_fixed[s]) break;
            labels[cnt] = cc;
            idxs[cnt] = idx;
            ++cnt;
            /* 入队之后，队里第 j 个指针对应窗口 my_i-bp_n+1+j，
             * 所以窗口 idx 的指针在第 idx-my_i+bp_n-1 个。
             * 写成 idx-(my_i-bp_n) 会大一格、拿错指针，
             * 后果是某个窗口的标签跟服务端差一个、别的全对 */
            const int step = (int)(idx - my_i + st->bp_n - 1);
            if (step < 0) break;
            cc = bp_at(st, step)[cc];
            --idx;
        }
        for (int q = cnt - 1; q >= 0; --q) on_decoded(st, idxs[q], labels[q]);
        st->bp_head = (st->bp_head + k) % TM_POST_LOOKAHEAD;
        st->bp_n -= k;
    }

    return drain(st, out, max_out);
}

int tm_post_flush(tm_post_t *st, tm_seg_t *out, int max_out)
{
    if (!cfg_ok(st)) return -1;
    const int m = st->cfg.n_classes;

    /* 还没定稿的沿当前最优路径收尾——离线那份也是这么收的：
     * 从最后一个窗口的最优状态回溯 */
    if (st->bp_n > 0) {
        int best = 0;
        for (int j = 1; j < m; ++j) if (st->score[j] > st->score[best]) best = j;
        int labels[TM_POST_LOOKAHEAD + 1];
        int cc = best;
        for (int k = st->bp_n - 1; k >= 0; --k) { labels[k] = cc; cc = bp_at(st, k)[cc]; }
        /* bp_n 个指针对应最后 bp_n 个窗口；链首那个窗口是再往前一个 */
        on_decoded(st, st->seq - st->bp_n - 1, cc);
        for (int k = 0; k < st->bp_n; ++k) {
            const long i = st->seq - st->bp_n + k;
            const int s = slot_of(st, i);
            if (s >= 0 && !st->dl_fixed[s]) on_decoded(st, i, labels[k]);
        }
        st->bp_n = 0;
    } else if (st->started && st->seq >= 1) {
        const int s = slot_of(st, st->seq - 1);
        if (s >= 0 && !st->dl_fixed[s]) {
            int best = 0;
            for (int j = 1; j < m; ++j) if (st->score[j] > st->score[best]) best = j;
            on_decoded(st, st->seq - 1, best);
        }
    }

    /* 最后一块收掉 */
    if (st->chunk_lo >= 0 && st->chunk_lo < st->seq)
        process_chunk(st, st->chunk_lo, st->seq - 1);
    st->chunk_lo = -1;
    st->last_quiet = -1;

    int n_out = drain(st, out, max_out);
    if (st->run_cls >= 0 && n_out < max_out) {
        out[n_out].cls = st->run_cls;
        out[n_out].start_ms = st->run_start_ms;
        out[n_out].end_ms = st->run_end_ms;
        out[n_out].n_windows = st->run_n;
        out[n_out].conf_max = st->run_cmax;
        out[n_out].conf_mean = st->run_n ? st->run_csum / (float)st->run_n : 0.0f;
        ++n_out;
        st->run_cls = -1;
    }
    return n_out;
}
