/* 自动生成，别手改 —— 改了下次导出就没了。 */
#ifndef TM_MODEL_H
#define TM_MODEL_H

#include "tm_runtime.h"
#include "tm_prep.h"

#define TM_ARENA_BYTES 2048
#define TM_N_CH 8
#define TM_N_T 16
#define TM_N_CLASSES 5

/* 类别顺序就是模型输出的下标顺序，跟训练时的 label 编码一致 */
static const char *const TM_CLASS_NAMES[] = {"活动", "睡觉", "抓挠", "未佩戴", "甩身体"};

extern const tm_model_t tm_model;
extern const tm_prep_t tm_model_prep;

#endif
