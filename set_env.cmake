# This file is used to configure DDK-related environment variables. 
# It is generated or updated in the following two scenarios:
# 1. This file does not exist or does not contain "ENV{DDK_PATH}".
# 2. The CANN version has been changed.

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux") 
    set(ENV{DDK_PATH} /home/HwHiAiUser/Ascend/ascend-toolkit/latest)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine OUTPUT_VARIABLE arch)
    if(${arch} MATCHES "x86_64-linux-gnu")
        set(ENV{NPU_HOST_LIB} /home/HwHiAiUser/Ascend/ascend-toolkit/latest/x86_64-linux/runtime/lib64/stub)
    else()
        set(ENV{NPU_HOST_LIB} /home/HwHiAiUser/Ascend/ascend-toolkit/latest/aarch64-linux/runtime/lib64/stub)
    endif()
endif()
