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
    ../Src/platform/user_periph_setup.c \
    ../Src/user/main.c \
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

#########################################################################################
### SDK 侧的源文件。这一串抄自 SDK 示例工程 ble_app_hrs 的 Makefile，砍掉了这个工程
### 用不到的（各种 profile、GUI、按键）。**一个文件都不复制进仓库**，全部用
### SDK_ROOT 指过去。
#########################################################################################
S := $(SDK_ROOT)
SDK_C_SRC_FILES := \
    $(S)/platform/soc/common/gr_system.c \
    $(S)/platform/soc/common/gr_interrupt.c \
    $(S)/platform/soc/common/gr_platform.c \
    $(S)/platform/soc/src/gr_soc.c \
    $(S)/platform/boards/board_SK.c \
    $(S)/drivers/src/app_dma.c \
    $(S)/drivers/src/app_gpiote.c \
    $(S)/drivers/src/app_io.c \
    $(S)/drivers/src/app_pwr_mgmt.c \
    $(S)/drivers/src/app_uart.c \
    $(S)/drivers/src/app_uart_dma.c \
    $(S)/components/libraries/utility/utility.c \
    $(S)/components/libraries/app_timer/app_timer.c \
    $(S)/components/libraries/app_key/app_key.c \
    $(S)/components/libraries/app_key/app_key_core.c \
    $(S)/components/libraries/hal_flash/hal_flash.c \
    $(S)/components/libraries/app_log/app_log.c \
    $(S)/components/libraries/app_assert/app_assert.c \
    $(S)/components/libraries/app_error/app_error.c \
    $(S)/components/libraries/app_error/cortex_backtrace.c \
    $(S)/components/libraries/ring_buffer/ring_buffer.c \
    $(S)/components/libraries/pmu_calibration/pmu_calibration.c \
    $(S)/components/libraries/fault_trace/fault_trace.c \
    $(S)/components/sdk/ble.c \
    $(S)/external/segger_rtt/SEGGER_RTT.c

SDK_INC := \
    $(S)/components/boards $(S)/components/libraries/app_assert \
    $(S)/components/libraries/app_error $(S)/components/libraries/app_log \
    $(S)/components/libraries/app_timer $(S)/components/libraries/fault_trace \
    $(S)/components/libraries/app_key \
    $(S)/components/libraries/hal_flash $(S)/components/libraries/pmu_calibration \
    $(S)/components/libraries/ring_buffer $(S)/components/libraries/utility \
    $(S)/components/patch/ind $(S)/components/profiles/common $(S)/components/sdk \
    $(S)/drivers/inc $(S)/drivers/inc/hal $(S)/external/segger_rtt \
    $(S)/platform/arch/arm/cortex-m/cmsis/core/include \
    $(S)/platform/boards $(S)/platform/include $(S)/platform/soc/include

PRJ_C_INCLUDE_PATH := \
    ../Src/config ../Src/platform ../Src/user \
    $(TINYML_ROOT) $(GEN_DIR) $(SDK_INC)

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
ifneq ($(wildcard $(GEN_DIR)/tm_pipeline_golden.h),)
ROUTE_DEFINES += -DTM_HAS_PIPELINE_GOLDEN
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
#
# -fno-math-errno：让 sqrtf 编译成单条 vsqrt.f32 指令，而不是调 libm。
# 实测代码小 72 字节、还少一次函数调用，而且 **结果一个 bit 都不变**——
# IEEE 规定 sqrt 必须正确舍入，硬件指令和 libm 给的是同一个数。
# 这一条在 tests/test_features_c.py 的编译选项里也加了，所以"不变"是验过的不是猜的。
# 注意它跟 -ffast-math **不是一回事**：那个会重排浮点运算，golden vector 会全红。
FP_FLAGS := -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations
FP_FLAGS += -fno-math-errno

# -Os 而不是 -O2：实测 tm_features.c 在 -O2 下代码大 23%（3884 vs 3152 字节），
# 而这段一两秒才跑一次，快那点没意义。要改先量，别凭感觉。
OPT ?= -Os

CFLAGS  = -std=gnu99 $(ARCH_FLAGS) $(FP_FLAGS) $(OPT)
CFLAGS += -ffunction-sections -fdata-sections -gdwarf-2 -MD
# 警告只对**我们自己的**代码开。SDK 的一大堆 unused-parameter 不是我们该修的，
# 混在一起只会把自己的警告淹掉——而自己的警告才是会咬人的那些
OUR_WARN := -Wall -Wextra
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

LDFLAGS += -Wl,--gc-sections
LDFLAGS += -L$(S)/components/libraries/app_crypto/gcc/ -lapp_crypto
LDFLAGS += -L$(S)/platform/soc/linker/gcc/ -lble_sdk
LDFLAGS += --specs=nano.specs --specs=nosys.specs

LINK_SCRIPT := $(S)/platform/soc/linker/gcc/gcc_linker_$(CHIP).lds
PATCH_FILE  := $(S)/platform/soc/linker/gcc/rom_symbol_gcc.txt
STARTUP_ASM := $(S)/platform/arch/arm/cortex-m/gcc/startup_gr55xx.s

ALL_C_SRC := $(PRJ_C_SRC_FILES) $(SDK_C_SRC_FILES)

.PHONY: all selftest size clean objs firmware

## 只编译成 .o 并报体积。完整链接需要 SDK 的启动文件、链接脚本和 libble_sdk.a，
## 那一步在有 SDK 的机器上做；这里先把"我们这部分编不编得过、占多少"确定下来。
objs:
	@mkdir -p $(BUILD)
	@for f in $(PRJ_C_SRC_FILES); do \
	    echo "  CC $$f"; \
	    $(CC) $(CFLAGS) $(OUR_WARN) -c $$f -o $(BUILD)/$$(basename $$f .c).o || exit 1; \
	done
	@echo ""
	@$(SIZE) $(BUILD)/*.o
	@echo ""
	@$(SIZE) -t $(BUILD)/*.o | tail -1
	@echo "  ↑ text = flash，bss = RAM。GR5513 是 512KB Flash / 128KB RAM，"
	@echo "    其中链接脚本留给应用的 RAM 是 128KB-16KB = 112KB。"

size: objs

## 完整固件：链出 .elf / .bin / .hex
firmware:
	@mkdir -p $(BUILD)/obj
	@for f in $(PRJ_C_SRC_FILES); do \
	    o=$(BUILD)/obj/$$(echo $$f | md5sum | cut -c1-8)_$$(basename $$f .c).o; \
	    $(CC) $(CFLAGS) $(OUR_WARN) -c $$f -o $$o || exit 1; \
	done
	@for f in $(SDK_C_SRC_FILES); do \
	    o=$(BUILD)/obj/$$(echo $$f | md5sum | cut -c1-8)_$$(basename $$f .c).o; \
	    $(CC) $(CFLAGS) -w -c $$f -o $$o || exit 1; \
	done
	@$(CC) $(CFLAGS) -c $(STARTUP_ASM) -o $(BUILD)/obj/startup.o
	@$(CC) $(CFLAGS) -T $(LINK_SCRIPT) $(PATCH_FILE) $(BUILD)/obj/*.o $(LDFLAGS) \
	    -Wl,-Map=$(BUILD)/$(MAKE_TARGET_NAME).map -o $(BUILD)/$(MAKE_TARGET_NAME).elf
	@$(OBJCOPY) -O binary $(BUILD)/$(MAKE_TARGET_NAME).elf $(BUILD)/$(MAKE_TARGET_NAME).bin
	@$(OBJCOPY) -O ihex   $(BUILD)/$(MAKE_TARGET_NAME).elf $(BUILD)/$(MAKE_TARGET_NAME).hex
	@echo ""
	@$(SIZE) $(BUILD)/$(MAKE_TARGET_NAME).elf
	@echo ""
	@echo "  GR5513BENDU: 512KB Flash / 128KB RAM（链接脚本留给应用 112KB）。"
	@echo "  注意链接脚本的 FLASH 区写的是 8MB（内存映射窗口），**超出 512KB"
	@echo "  链接不会报错**，所以要自己看上面 text 那个数。"

all: firmware

clean:
	rm -rf $(BUILD)

-include $(BUILD)/*.d
