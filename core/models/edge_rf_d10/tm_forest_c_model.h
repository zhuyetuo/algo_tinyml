/* 自动生成，别手改 —— 改了下次导出就没了。 */
#ifndef TM_FOREST_C_MODEL_H
#define TM_FOREST_C_MODEL_H

#include "tm_forest_c.h"

#define TM_FC_N_TREES 20
#define TM_FC_N_NODES 11810
#define TM_FC_N_LEAVES 5915
#define TM_FC_N_FEATURES 193
#define TM_FC_N_CLASSES 5
/* flash：节点 82,670 B + 叶子 29,575 B + 树表 84 B = 112,329 B（109.7 KB） */

static const char *const TM_FC_CLASS_NAMES[] = {"活动", "睡觉", "抓挠", "未佩戴", "甩身体"};

extern const tm_forest_c_t tm_forest_c;

#endif
