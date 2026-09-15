/* 把 tm_post 编成共享库，给 Python 直接调，跟 label_service/postprocess.py 对答案。
 *
 * 逐个窗口喂进去（就是板上的调用方式），把吐出来的片段收集起来。
 * **不是"批量算一遍"**——那样测的就不是流式那条路，而流式正是端上跟
 * 服务端唯一的结构性差别。
 */

#include <stdlib.h>
#include <string.h>

#include "tm_post.h"

/* 一次跑完整条序列，片段写进调用方给的数组。返回片段数，-1 = 配置不合法。
 *
 * probs: [n][n_classes] 行主序；ts_ms: [n]
 * 片段用扁平数组回去（cls, start, end, n_windows, conf_max, conf_mean），
 * 免得在 ctypes 那边摆结构体对齐——**对齐猜错不会报错，只会读到垃圾数**。
 */
int tm_post_run(const float *probs, const unsigned int *ts_ms, int n,
                int n_classes, int scratch, int shake,
                float window_s, float stride_s,
                float viterbi_switch, float event_gap_s, float shake_absorb_s,
                int event_min_windows, float event_min_mean, float event_single_conf,
                int *out_cls, unsigned int *out_start, unsigned int *out_end,
                int *out_nwin, float *out_cmax, float *out_cmean, int max_out,
                unsigned int *forced_out)
{
    tm_post_cfg_t cfg;
    tm_post_cfg_default(&cfg, n_classes, scratch, shake, window_s, stride_s);
    cfg.viterbi_switch = viterbi_switch;
    cfg.event_gap_s = event_gap_s;
    cfg.shake_absorb_s = shake_absorb_s;
    cfg.event_min_windows = event_min_windows;
    cfg.event_min_mean = event_min_mean;
    cfg.event_single_conf = event_single_conf;

    tm_post_t st;
    tm_post_init(&st, &cfg);

    tm_seg_t buf[64];
    int n_out = 0;
    for (int i = 0; i < n; ++i) {
        const int k = tm_post_on_window(&st, probs + (size_t)i * n_classes,
                                        ts_ms[i], buf, 64);
        if (k < 0) return -1;
        for (int j = 0; j < k && n_out < max_out; ++j, ++n_out) {
            out_cls[n_out] = buf[j].cls;
            out_start[n_out] = buf[j].start_ms;
            out_end[n_out] = buf[j].end_ms;
            out_nwin[n_out] = buf[j].n_windows;
            out_cmax[n_out] = buf[j].conf_max;
            out_cmean[n_out] = buf[j].conf_mean;
        }
    }
    const int k = tm_post_flush(&st, buf, 64);
    if (k < 0) return -1;
    for (int j = 0; j < k && n_out < max_out; ++j, ++n_out) {
        out_cls[n_out] = buf[j].cls;
        out_start[n_out] = buf[j].start_ms;
        out_end[n_out] = buf[j].end_ms;
        out_nwin[n_out] = buf[j].n_windows;
        out_cmax[n_out] = buf[j].conf_max;
        out_cmean[n_out] = buf[j].conf_mean;
    }
    /* 两个计数合起来报：都是"可能跟服务端不同"的地方，
     * 只报一个的话另一个出问题时看不见 */
    if (forced_out) { forced_out[0] = st.forced; forced_out[1] = st.forced_split; }
    return n_out;
}

/* 只要解码出来的逐窗口标签，用来单独定位"是 viterbi 不一样"还是
 * "事件组装不一样"——混在一起看片段差异，查不出是哪一步的锅。 */
int tm_post_decode_only(const float *probs, const unsigned int *ts_ms, int n,
                        int n_classes, float viterbi_switch,
                        float window_s, float stride_s,
                        int *out_labels, unsigned int *forced_out)
{
    tm_post_cfg_t cfg;
    /* scratch/shake = -1：没有事件类别，事件那一路整个跳过，
     * final 就等于 decoded 的状态填充——但我们要的是 dl_dec，见下面 */
    tm_post_cfg_default(&cfg, n_classes, -1, -1, window_s, stride_s);
    cfg.viterbi_switch = viterbi_switch;

    tm_post_t st;
    tm_post_init(&st, &cfg);
    tm_seg_t buf[64];

    /* 逐窗口喂完之后 dl_dec 里只剩最后 DELAY 个，所以边喂边抄出来。
     * 抄的时机：一个窗口一旦 fixed 就不会再变 */
    char *done = (char *)calloc((size_t)n, 1);
    if (!done) return -1;
    for (int i = 0; i < n; ++i) {
        if (tm_post_on_window(&st, probs + (size_t)i * n_classes, ts_ms[i], buf, 64) < 0) {
            free(done);
            return -1;
        }
        for (int j = 0; j < n; ++j) {
            if (done[j]) continue;
            const long rel = (long)j - (st.seq - st.dl_n);
            if (rel < 0 || rel >= st.dl_n) continue;
            const int s = (int)((st.dl_head + rel) % TM_POST_DELAY);
            if (st.dl_fixed[s]) { out_labels[j] = st.dl_dec[s]; done[j] = 1; }
        }
    }
    tm_post_flush(&st, buf, 64);
    for (int j = 0; j < n; ++j) {
        if (done[j]) continue;
        const long rel = (long)j - (st.seq - st.dl_n);
        if (rel < 0 || rel >= st.dl_n) continue;
        const int s = (int)((st.dl_head + rel) % TM_POST_DELAY);
        if (st.dl_fixed[s]) { out_labels[j] = st.dl_dec[s]; done[j] = 1; }
    }
    int missing = 0;
    for (int j = 0; j < n; ++j) if (!done[j]) ++missing;
    free(done);
    if (forced_out) *forced_out = st.forced;
    /* 有窗口没定稿 = 流式逻辑漏了，返回负数让测试当场失败，
     * **不要悄悄留个 0 标签** */
    return missing ? -1 - missing : n;
}
