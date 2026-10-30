/* 板上验收的 qemu 版：semihosting 读两个 bin，调 example.c 的 pb_acceptance。 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
int pb_acceptance(const uint8_t *in, uint32_t in_len, const uint8_t *expect, uint32_t expect_len);
static uint8_t *load(const char *p, uint32_t *n){FILE*f=fopen(p,"rb");if(!f)return 0;static uint8_t b[2][60000];static int k;uint8_t*q=b[k++];*n=fread(q,1,60000,f);fclose(f);return q;}
int main(void){uint32_t a,b;uint8_t*i=load("test_input.bin",&a),*e=load("test_expect.bin",&b);if(!i||!e){printf("load fail\n");return 2;}int r=pb_acceptance(i,a,e,b);printf("ARM acceptance=%d (%u segs)\n",r,b/12);return r;}
