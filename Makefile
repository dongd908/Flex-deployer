ifeq (${ASCEND_INSTALL_PATH},)
    ASCEND_PATH := /home/HwHiAiUser/Ascend/ascend-toolkit/latest
else
    ASCEND_PATH := ${ASCEND_INSTALL_PATH}
endif


LOCAL_DIR  := ./
ATC_INCLUDE_DIR := $(ASCEND_PATH)/compiler/include
FWK_INCLUDE_DIR := $(ASCEND_PATH)/compiler/include
OPP_INCLUDE_DIR := $(ASCEND_PATH)/opp/op_proto/built-in/inc

LOCAL_MODULE_NAME := ir_build
LOCAL_FWK_MODULE_NAME := fwk_ir_build
CC := g++
CFLAGS := -std=c++11 -g -Wall -D_GLIBCXX_USE_CXX11_ABI=0
PYTHONFLAGS := $(shell python3.7-config --cflags --ldflags)
SRCS := $(wildcard $(LOCAL_DIR)/src/main.cpp) $(wildcard $(LOCAL_DIR)/src/scheme_lhq.cpp) $(wildcard $(LOCAL_DIR)/src/subgraph_lhq.cpp) $(wildcard $(LOCAL_DIR)/src/infershape_lhq.cpp)

INCLUDES := -I $(ASCEND_PATH)/opp/op_proto/built-in/inc \
            -I $(ATC_INCLUDE_DIR)/graph \
            -I $(ATC_INCLUDE_DIR)/ge \
            -I $(ATC_INCLUDE_DIR)/parser \
            -I $(ASCEND_PATH)/compiler/include \
            -I $(LOCAL_DIR)/src/ \

FWK_INCLUDES := -I $(ASCEND_PATH)/opp/op_proto/built-in/inc \
            -I $(FWK_INCLUDE_DIR)/graph \
            -I $(FWK_INCLUDE_DIR)/ge \
            -I $(FWK_INCLUDE_DIR)/parser \
            -I $(ASCEND_PATH)/compiler/include \
            -I $(LOCAL_DIR)/src/ \

LIBS := -L ${ASCEND_PATH}/compiler/lib64/stub \
    -lgraph \
    -lge_compiler \
    -lfmk_parser \

FWK_LIBS := -L ${ASCEND_PATH}/compiler/lib64/stub \
    -lgraph \
    -lge_runner \
    -lfmk_parser \

ir_build:
	mkdir -p out
	$(CC) $(SRCS) $(INCLUDES) $(LIBS) $(CFLAGS) -o ./out/$(LOCAL_MODULE_NAME)  $(PYTHONFLAGS)

fwk_ir_build:
	mkdir -p out
	$(CC) $(SRCS) $(FWK_INCLUDES) $(FWK_LIBS) $(CFLAGS) -o ./out/$(LOCAL_FWK_MODULE_NAME)

clean:
