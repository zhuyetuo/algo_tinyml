# Linux 上编 PC 参考实现 / 命令行测试。只要 gcc + make，不要 Python、不要板子。
#
#   make            编出 ./edge_cli
#   make test       golden 自检 + 跑一遍 sample.csv
#   make lib        只编运行时 + 模型成 build/libtinyml_host.a（给别的 PC 程序链）
#
# 编的就是 core/ 和 model/ 下板上那份 C；编译选项跟固件一致（尤其 -ffp-contract=off，
# 不带的话浮点末位跟导出时对不上，selftest 会红）。

KIND     ?= @KIND@
N_T      ?= @N_T@
N_CLASS  ?= @N_CLASSES@

CC       ?= gcc
CFLAGS   ?= -O2
CFLAGS   += -std=c99 -Wall -ffp-contract=off -fno-math-errno \
            -DTM_FEAT_MAX_T=$(N_T) -DTM_FEAT_MAX_NPERSEG=$(N_T) -DTM_POST_MAX_CLASSES=$(N_CLASS) \
            -Icore -Imodel -Ipc
LDLIBS   += -lm

ifeq ($(KIND),rf)
  CFLAGS += -DTM_KIND_RF
  SRCS    = core/tm_imu.c core/tm_features.c core/tm_forest_c.c model/tm_feat_cfg.c model/tm_forest_c_model.c
else
  CFLAGS += -DTM_KIND_CNN
  SRCS    = core/tm_imu.c core/tm_prep.c core/tm_runtime.c model/tm_model.c
endif
OBJS = $(patsubst %.c,build/%.o,$(SRCS))

all: edge_cli

build/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

build/libtinyml_host.a: $(OBJS)
	$(AR) rcs $@ $^

lib: build/libtinyml_host.a

edge_cli: pc/edge_cli.c build/libtinyml_host.a
	$(CC) $(CFLAGS) $< build/libtinyml_host.a -o $@ $(LDLIBS)

test: edge_cli
	./edge_cli --selftest
	./edge_cli pc/sample.csv --out build/sample_result.csv
	@echo "逐窗口结果：build/sample_result.csv（前几行如下）"
	@head -5 build/sample_result.csv

clean:
	rm -rf build edge_cli

.PHONY: all lib test clean
