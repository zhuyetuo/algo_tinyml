/* 上板自检的结果。做成结构体而不是"返回 0/-1 + 打日志"：
 * 固件上不一定有串口，得能把结论通过 BLE 报出去。 */

#ifndef TINYML_SELFTEST_H
#define TINYML_SELFTEST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TM_SELFTEST_OK = 0,
    TM_SELFTEST_NO_VECTORS = 1,     /* 一条都没验到。这**不算**通过 */
    TM_SELFTEST_INVOKE_FAILED = 2,
    TM_SELFTEST_MISMATCH = 3
} tm_selftest_kind_t;

typedef struct {
    int n_checked;
    int fail_index;     /* 第几条挂的；没挂是 -1 */
    tm_selftest_kind_t fail_kind;
    int8_t got, want;           /* CNN 路线：第一个不一致的字节 */
    uint32_t got_bits, want_bits;  /* RF 路线：第一个不一致的 float 位模式 */
} tm_selftest_report_t;

/* 返回 0 表示全过。0 之外一律当成"这块板子的推理结果不可信"，
 * 别接着看准确率——那个数在自检没过的前提下没有意义。 */
int tm_selftest_run(tm_selftest_report_t *r);

/* 给人看的解释，含排查顺序。 */
const char *tm_selftest_explain(const tm_selftest_report_t *r);

#ifdef __cplusplus
}
#endif

#endif
