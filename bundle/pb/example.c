/* pet_behavior 调用示例 + 集成验收。
 *
 * 板上：把 test_input.bin / test_expect.bin 放进固件（如 xxd -i 转成数组），调用
 *       pb_acceptance(in, in_len, expect, expect_len)，返回 0 即通过。
 * PC 上：gcc -DPB_EXAMPLE_MAIN example.c libpetbehavior_host.a -lm
 *       ./a.out test_input.bin test_expect.bin
 *
 * test_input.bin（小端）：
 *   "PBIN"  pb_config_t  uint16 每批样本数  uint16 0  uint32 样本数 N  N × pb_sample_t
 *   第 k 批的 t0_ms = k × 每批样本数 × 1000 / 采样率（四舍五入）
 * test_expect.bin：按输出顺序排列的 pb_segment_t（每个 12 字节）
 */
#include <stdint.h>
#include <string.h>

#include "pet_behavior.h"

static uint8_t pb_mem[PB_WORK_SIZE] __attribute__((aligned(4)));

static uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

/* 输出的片段逐个交给 sink；sink 返回非 0 即停止。 */
typedef int (*pb_sink_t)(const pb_segment_t *seg, void *ctx);

static int emit(int n, const pb_segment_t *segs, pb_sink_t sink, void *ctx)
{
    for (int i = 0; i < n; i++)
        if (sink(&segs[i], ctx)) return -100;
    return 0;
}

/* 跑一遍验收输入，片段依次交给 sink。返回 0 成功，负数为库的错误码或输入格式错。 */
int pb_run_input(const uint8_t *in, uint32_t in_len, pb_sink_t sink, void *ctx)
{
    if (in_len < 16 || memcmp(in, "PBIN", 4) != 0) return -101;
    pb_config_t cfg;
    cfg.sample_rate_hz = rd16(in + 4);
    cfg.acc_range_g = in[6];
    cfg.reserved = 0;
    const uint16_t batch = rd16(in + 8);
    const uint32_t n = rd32(in + 12);
    if (batch == 0 || batch > 256 || in_len < 16 + n * 6u) return -101;

    int r = pb_init(&cfg, pb_mem, sizeof pb_mem);   /* 内含自检 */
    if (r != 0) return r;

    static pb_sample_t buf[256];
    pb_segment_t segs[8];
    for (uint32_t off = 0, k = 0; off < n; off += batch, k++) {
        const uint16_t m = (uint16_t)((n - off) < batch ? (n - off) : batch);
        for (uint16_t i = 0; i < m && i < 256; i++) {
            const uint8_t *p = in + 16 + (off + i) * 6u;
            buf[i].x = (int16_t)rd16(p);
            buf[i].y = (int16_t)rd16(p + 2);
            buf[i].z = (int16_t)rd16(p + 4);
        }
        const uint32_t t0 = (uint32_t)(((uint64_t)off * 1000u + cfg.sample_rate_hz / 2) / cfg.sample_rate_hz);
        r = pb_push(buf, m, t0, segs, 8);
        while (r > 0) {                       /* 一次取不完：n=0 再取 */
            if (emit(r, segs, sink, ctx)) return -100;
            r = r == 8 ? pb_push(NULL, 0, 0, segs, 8) : 0;
        }
        if (r < 0) return r;
    }
    r = pb_flush(segs, 8);
    while (r > 0) {
        if (emit(r, segs, sink, ctx)) return -100;
        r = r == 8 ? pb_flush(segs, 8) : 0;
    }
    return r < 0 ? r : 0;
}

typedef struct {
    const uint8_t *exp;
    uint32_t len, pos;
    int bad;
} cmp_ctx_t;

static int cmp_sink(const pb_segment_t *s, void *c)
{
    cmp_ctx_t *x = (cmp_ctx_t *)c;
    uint8_t b[12];
    for (int i = 0; i < 4; i++) { b[i] = (uint8_t)(s->start_ms >> (8 * i)); b[4 + i] = (uint8_t)(s->end_ms >> (8 * i)); }
    b[8] = s->type; b[9] = s->confidence; b[10] = b[11] = 0;
    if (x->pos + 12 > x->len || memcmp(x->exp + x->pos, b, 12) != 0) x->bad++;
    x->pos += 12;
    return 0;
}

/* 集成验收：输出与 test_expect.bin 逐字节一致返回 0；不一致返回 1；库报错返回负数。 */
int pb_acceptance(const uint8_t *in, uint32_t in_len, const uint8_t *expect, uint32_t expect_len)
{
    cmp_ctx_t c = {expect, expect_len, 0, 0};
    const int r = pb_run_input(in, in_len, cmp_sink, &c);
    if (r < 0) return r;
    return (c.bad == 0 && c.pos == expect_len) ? 0 : 1;
}

#ifdef PB_EXAMPLE_MAIN
#include <stdio.h>
#include <stdlib.h>

static uint8_t *load(const char *path, uint32_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc(n > 0 ? (size_t)n : 1);
    *len = (uint32_t)fread(p, 1, (size_t)n, f);
    fclose(f);
    return p;
}

static const char *TYPE_NAMES[] = {"休息", "进食", "喝水", "舔舐", "抓挠", "吠叫", "其他", "未佩戴"};

static int print_sink(const pb_segment_t *s, void *c)
{
    FILE *f = (FILE *)c;
    if (f) {
        fwrite(s, 1, 12, f);   /* x86 / ARM 都是小端，结构体就是协议里的 12 字节 */
    } else {
        printf("%9.3f %9.3f  %u %-6s %3u\n", s->start_ms / 1000.0, s->end_ms / 1000.0, s->type,
               s->type < 8 ? TYPE_NAMES[s->type] : "?", s->confidence);
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "用法：%s test_input.bin [test_expect.bin]   比对；只给输入则打印片段\n"
                        "      %s test_input.bin --write out.bin       生成期望输出\n", argv[0], argv[0]);
        return 2;
    }
    uint32_t in_len = 0, exp_len = 0;
    uint8_t *in = load(argv[1], &in_len);
    if (!in) { fprintf(stderr, "读不了 %s\n", argv[1]); return 2; }
    printf("pb_version = 0x%08x\n", (unsigned)pb_version());
    if (argc >= 4 && strcmp(argv[2], "--write") == 0) {
        FILE *f = fopen(argv[3], "wb");
        const int r = pb_run_input(in, in_len, print_sink, f);
        fclose(f);
        return r == 0 ? 0 : 1;
    }
    if (argc < 3) {
        const int r = pb_run_input(in, in_len, print_sink, NULL);
        printf("返回 %d\n", r);
        return r == 0 ? 0 : 1;
    }
    uint8_t *exp = load(argv[2], &exp_len);
    if (!exp) { fprintf(stderr, "读不了 %s\n", argv[2]); return 2; }
    const int r = pb_acceptance(in, in_len, exp, exp_len);
    printf("selftest: %s\n", pb_selftest() == 0 ? "通过" : "不通过");
    printf("集成验收：%s（%u 个片段）\n", r == 0 ? "通过" : r > 0 ? "输出不一致" : "库报错", exp_len / 12);
    if (r < 0) printf("错误码 %d\n", r);
    return r == 0 ? 0 : 1;
}
#endif
