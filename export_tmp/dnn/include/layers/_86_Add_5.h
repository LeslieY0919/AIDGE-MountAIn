#ifndef _86_ADD_5_LAYER_H
#define _86_ADD_5_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _79_SPLIT_3_OUTPUT_1_IN_BATCH 1
#define _79_SPLIT_3_OUTPUT_1_NB_CHANNELS 128
#define _79_SPLIT_3_OUTPUT_1_IN_HEIGHT 10
#define _79_SPLIT_3_OUTPUT_1_IN_WIDTH 10

// OUTPUT CONF
#define _86_ADD_5_OUTPUT_0_NB_OUTPUTS 128
#define _86_ADD_5_OUTPUT_0_OUT_HEIGHT 10
#define _86_ADD_5_OUTPUT_0_OUT_WIDTH 10
#define _86_ADD_5_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _86_ADD_5_OUTPUT_0_MEM_SIZE 512
#define _86_ADD_5_OUTPUT_0_MEM_OFFSET 102400
#define _86_ADD_5_OUTPUT_0_MEM_STRIDE 512
#define _86_ADD_5_OUTPUT_0_MEM_LENGTH 10
#define _86_ADD_5_OUTPUT_0_MEM_CONT_OFFSET 0
#define _86_ADD_5_OUTPUT_0_MEM_CONT_SIZE 0
#define _86_ADD_5_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _86_ADD_5_OUTPUT_0_MEM_WRAP_SIZE 0


#define _86_ADD_5_NB_MAT 1
#define _86_ADD_5_INPUT1_CONT_SIZE 12800
#define _86_ADD_5_INPUT2_CONT_SIZE 12800
#define _86_ADD_5_OUTPUT_CONT_SIZE 12800

constexpr size_t _86_ADD_5_OFFSET_IN1[] =  { 0 };
constexpr size_t _86_ADD_5_OFFSET_IN2[] = { 0 };

#define _86_ADD_5_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _86_ADD_5_ELEM_OP export_cpp::ElemWise_T::Add

#define _86_ADD_5_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _86_ADD_5_RESCALING = {};

#endif /* _86_ADD_5_LAYER_H */
