/* qemu mps2-an386（Cortex-M4）上跑验收：向量表 + 打开 FPU。测试用，不进包。 */
extern void _start(void);
extern unsigned long __stack_top;
__attribute__((section(".vectors"), used)) const void *vectors[] = {&__stack_top, (void *)_start};
__attribute__((constructor(101))) static void fpu_on(void){ *(volatile unsigned*)0xE000ED88 |= (0xFu<<20); __asm volatile("dsb; isb"); }
