cmake_minimum_required(VERSION 3.5.1)
project(GE_IR[CXX])
include(set_env.cmake)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /home/HwHiAiUser/Ascend/ascend-toolkit/latest)
endif()

set(ATC_INCLUDE_DIR ${ASCEND_PATH}/compiler/include)
set(FWK_INCLUDE_DIR ${ASCEND_PATH}/compiler/include)
set(OPP_INCLUDE_DIR $(ASCEND_PATH)/opp/op_proto/built-in/inc)

set(SRC_LIST
    src/main.cpp
    src/subgraph_lhq.cpp
)

set(common_compile_options
    -std=c++11
    -g
    -Wall    
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0	
)

############ ir_build ############
add_executable(ir_build
    ${SRC_LIST}
)

target_compile_options(ir_build PRIVATE
    ${common_compile_options}
)

target_compile_definitions(ir_build PRIVATE
    ${common_compile_definitions}	
)

target_include_directories(ir_build PRIVATE
    ${ASCEND_PATH}/opp/op_proto/built-in/inc
    ${ATC_INCLUDE_DIR}/graph
    ${ATC_INCLUDE_DIR}/ge
    ${ATC_INCLUDE_DIR}/parser    
    ${ATC_INCLUDE_DIR}
    src/
)

target_link_directories(ir_build PRIVATE
    ${ASCEND_PATH}/compiler/lib64/stub	
)

target_link_libraries(ir_build PRIVATE
    -Wl,--no-as-needed
    graph
    ge_compiler
    fmk_parser
    -Wl,--as-needed    
)

############ fwk_ir_build ############
add_executable(fwk_ir_build
    ${SRC_LIST}
)

target_compile_options(fwk_ir_build PRIVATE
    ${common_compile_options}
)

target_compile_definitions(fwk_ir_build PRIVATE
    ${common_compile_definitions}
)

target_include_directories(fwk_ir_build PRIVATE
    ${ASCEND_PATH}/opp/op_proto/built-in/inc
    ${FWK_INCLUDE_DIR}/graph
    ${FWK_INCLUDE_DIR}/ge
    ${FWK_INCLUDE_DIR}/parser
    ${FWK_INCLUDE_DIR}
    src/
)

target_link_directories(fwk_ir_build PRIVATE
    ${ASCEND_PATH}/compiler/lib64/stub
)

target_link_libraries(fwk_ir_build PRIVATE
    -Wl,--no-as-needed
    graph
    ge_runner
    fmk_parser
    -Wl,--as-needed
)

############ install ############
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})

install(TARGETS ir_build fwk_ir_build OPTIONAL
    RUNTIME DESTINATION output	
)
