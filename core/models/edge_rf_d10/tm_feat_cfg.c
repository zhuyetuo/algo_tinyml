#include "tm_features.h"

static const float tm_feat_cfg_win[] = {0x0.0p+0f, 0x1.37ca180000000p-5f, 0x1.2bec340000000p-3f, 0x1.3c10ea0000000p-2f, 0x1.0000000000000p-1f, 0x1.61f78a0000000p-1f, 0x1.b504f40000000p-1f, 0x1.ec835e0000000p-1f, 0x1.0000000000000p+0f, 0x1.ec835e0000000p-1f, 0x1.b504f40000000p-1f, 0x1.61f78a0000000p-1f, 0x1.0000000000000p-1f, 0x1.3c10ea0000000p-2f, 0x1.2bec340000000p-3f, 0x1.37ca180000000p-5f};
static const float tm_feat_cfg_cos[] = {0x1.0000000000000p+0f, 0x1.d906bc0000000p-1f, 0x1.6a09e60000000p-1f, 0x1.87de2a0000000p-2f, 0x1.1a62640000000p-54f, -0x1.87de2a0000000p-2f, -0x1.6a09e60000000p-1f, -0x1.d906bc0000000p-1f};
static const float tm_feat_cfg_sin[] = {-0x0.0p+0f, -0x1.87de2a0000000p-2f, -0x1.6a09e60000000p-1f, -0x1.d906bc0000000p-1f, -0x1.0000000000000p+0f, -0x1.d906bc0000000p-1f, -0x1.6a09e60000000p-1f, -0x1.87de2a0000000p-2f};
static const int16_t tm_feat_cfg_br[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

const tm_feat_cfg_t tm_feat_cfg = {
    16, 8, 16, 0x1.0000000000000p+4f,
    tm_feat_cfg_win, tm_feat_cfg_cos, tm_feat_cfg_sin, tm_feat_cfg_br
};
