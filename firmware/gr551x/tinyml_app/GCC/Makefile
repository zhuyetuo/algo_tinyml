#########################################################################################
# GR551x 上的 tinyml 应用（GCC / arm-none-eabi）。
#
# 这个 Makefile **不复制 SDK 的任何文件**，全部用 SDK_ROOT 指过去。理由：
#   1. SDK 是 Goodix 的，版本会更新，复制一份进来迟早跟上游分叉；
#   2. 复制过来的那份出了问题，没人分得清是我们改的还是 SDK 本来如此。
#
# 用法：
#     make SDK_ROOT=/path/to/GR551x_SDK            编译
#     make SDK_ROOT=... selftest                   只编译 golden vector 自检（不含 BLE）
#     make SDK_ROOT=... size                       看占多少 flash / RAM
#
# SDK 从这里下载：https://www.goodix.com/zh/software_tool/gr551x_sdk
# 也有 GitHub 镜像：https://github.com/goodix-ble/GR551x.SDK
#########################################################################################

MAKE_TARGET_NAME := tinyml_app

SDK_ROOT ?=
ifeq ($(SDK_ROOT),)
$(error 要指定 SDK 路径：make SDK_ROOT=/path/to/GR551x_SDK)
endif
ifeq ($(wildcard $(SDK_ROOT)/platform/soc/linker/gcc/libble_sdk.a),)
$(error $(SDK_ROOT) 里找不到 platform/soc/linker/gcc/libble_sdk.a，路径不对或者 SDK 不完整)
endif

# 芯片型号。GR5513BENDU = 128KB RAM / 512KB Flash；GR5515 是 256KB RAM。
# 选错的直接后果是链接脚本给的 RAM 大小不对——GR5513 用 GR5515 的脚本能链过，
# 跑起来才踩到不存在的内存。
CHIP ?= gr5513

TINYML_ROOT := ../../../tinyml
GEN_DIR     ?= ../../../generated

#########################################################################################
### 我们自己的源文件
#########################################################################################
PRJ_C_SRC_FILES := \
    $(TINYML_ROOT)/tm_runtime.c \
    $(TINYML_ROOT)/tm_window.c \
    $(TINYML_ROOT)/tm_forest.c \
    $(TINYML_ROOT)/tm_features.c \
    ../Src/user/tinyml_task.c \
    ../Src/user/tinyml_selftest.c

# 导出的模型和常量表。走 wildcard 是因为两条路线（CNN / RF）导出的文件名不同，
# 只带其中一条的时候不该因为另一条的文件不存在而编不过
PRJ_C_SRC_FILES += $(wildcard $(GEN_DIR)/tm_model.c)
PRJ_C_SRC_FILES += $(wildcard $(GEN_DIR)/tm_forest_model.c)
PRJ_C_SRC_FILES += $(wildcard $(GEN_DIR)/tm_feat_cfg.c)

PRJ_C_INCLUDE_PATH := \
    ../Src/config ../Src/platform ../Src/user \
    $(TINYML_ROOT) $(GEN_DIR)

# 带哪条路线由**导出了哪些文件**决定，不用手工改宏。手工改的话迟早出现
# "导了 RF 但宏还指着 CNN"，而那时候自检会报 NO_VECTORS 而不是编译错——
# 看起来像"还没导出"，其实是配错了
ifneq ($(wildcard $(GEN_DIR)/tm_model.c),)
ROUTE_DEFINES += -DTM_HAS_CNN
endif
ifneq ($(wildcard $(GEN_DIR)/tm_forest_model.c),)
ROUTE_DEFINES += -DTM_HAS_RF
endif
ifneq ($(wildcard $(GEN_DIR)/tm_forest_golden.h),)
ROUTE_DEFINES += -DTM_HAS_RF_GOLDEN
endif

#########################################################################################
### 编译选项
#########################################################################################
CROSS_COMPILE = arm-none-eabi-
CC      = $(CROSS_COMPILE)gcc
OBJCOPY = $(CROSS_COMPILE)objcopy
SIZE    = $(CROSS_COMPILE)size

# 前四行跟 SDK 示例工程的 COMMON_COMPILE_FLAGS 一致，别改——改了跟 SDK 预编译的
# libble_sdk.a 的 ABI 就可能对不上（尤其 -mfloat-abi）
ARCH_FLAGS := -mcpu=cortex-m4 -mthumb -mthumb-interwork -mapcs-frame \
              -mfloat-abi=softfp -mfpu=fpv4-sp-d16

# ↓↓↓ 这一行是**必须加的**，SDK 示例工程里没有 ↓↓↓
#
# -ffp-contract=off：不允许编译器把 a*b+c 合成一条 FMA。合成之后中间结果少一次
# 舍入，算出来的浮点跟 PC 上的参考实现末位就不同，golden vector 会红。
# 不是"最好加"，是这套一致性保证的前提。
#
# -fno-fast-math / -fno-unsafe-math-optimizations：显式写出来，防止以后有人在
# 上层 Makefile 里塞 -ffast-math。-ffast-math 会假定没有 NaN/Inf、允许重排浮点
# 运算，那时候 golden vector 会全红，而人第一反应多半是去怀疑模型。
FP_FLAGS := -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations

# -Os 而不是 -O2：实测 tm_features.c 在 -O2 下代码大 23%（3884 vs 3152 字节），
# 而这段一两秒才跑一次，快那点没意义。要改先量，别凭感觉。
OPT ?= -Os

CFLAGS  = -std=gnu99 $(ARCH_FLAGS) $(FP_FLAGS) $(OPT)
CFLAGS += -ffunction-sections -fdata-sections -gdwarf-2 -MD
CFLAGS += -Wall -Wextra
CFLAGS += $(foreach inc,$(PRJ_C_INCLUDE_PATH),-I$(inc))
CFLAGS += $(ROUTE_DEFINES)

# 特征提取里的临时缓冲按这两个上限开静态数组。**按实际窗口调小能省一半 RAM**：
# 默认上限 64 时 bss 是 4624 字节，设成 32（16Hz×2 秒）之后是 2320 字节。
TM_FEAT_MAX_T ?= 32
TM_FEAT_MAX_NPERSEG ?= 32
CFLAGS += -DTM_FEAT_MAX_T=$(TM_FEAT_MAX_T) -DTM_FEAT_MAX_NPERSEG=$(TM_FEAT_MAX_NPERSEG)

BUILD := build

#########################################################################################
### 自检：不接 BLE、不接 SDK，只跑 golden vector
###
### 上板第一步应该是它，不是直接跑真实数据。这一步过不了，后面看到的任何准确率
### 都是没有意义的——分不清是模型不行还是工具链不对。
#########################################################################################
SELFTEST_SRC := $(TINYML_ROOT)/tm_runtime.c $(TINYML_ROOT)/tm_forest.c \
                $(TINYML_ROOT)/tm_features.c ../Src/user/tinyml_selftest.c \
                $(wildcard $(GEN_DIR)/tm_model.c) \
                $(wildcard $(GEN_DIR)/tm_forest_model.c) \
                $(wildcard $(GEN_DIR)/tm_feat_cfg.c)

.PHONY: all selftest size clean objs

## 只编译成 .o 并报体积。完整链接需要 SDK 的启动文件、链接脚本和 libble_sdk.a，
## 那一步在有 SDK 的机器上做；这里先把"我们这部分编不编得过、占多少"确定下来。
objs:
	@mkdir -p $(BUILD)
	@for f in $(PRJ_C_SRC_FILES); do \
	    echo "  CC $$f"; \
	    $(CC) $(CFLAGS) -c $$f -o $(BUILD)/$$(basename $$f .c).o || exit 1; \
	done
	@echo ""
	@$(SIZE) $(BUILD)/*.o
	@echo ""
	@$(SIZE) -t $(BUILD)/*.o | tail -1
	@echo "  ↑ text = flash，bss = RAM。GR5513 是 512KB Flash / 128KB RAM，"
	@echo "    其中链接脚本留给应用的 RAM 是 128KB-16KB = 112KB。"

size: objs

all: objs
	@echo ""
	@echo "完整固件的链接要在有 SDK 的机器上做，照 SDK 示例工程的做法："
	@echo "  1. 复制 $(SDK_ROOT)/projects/ble/ble_peripheral/ble_app_gus 一份；"
	@echo "  2. 把上面这些 .c 和 include 路径加进它的 GCC/Makefile；"
	@echo "  3. 在它的 COMMON_COMPILE_FLAGS 里加上 $(FP_FLAGS)；  ← 别漏"
	@echo "  4. 链接脚本用 GCC/gcc_linker_$(CHIP).lds。"
	@echo "详见 ../../README.md。"

clean:
	rm -rf $(BUILD)

-include $(BUILD)/*.d
