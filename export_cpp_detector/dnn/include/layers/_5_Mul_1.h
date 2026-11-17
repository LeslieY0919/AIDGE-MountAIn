#ifndef _5_MUL_1_LAYER_H
#define _5_MUL_1_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _3_PADCONV_1_OUTPUT_0_IN_BATCH 1
#define _3_PADCONV_1_OUTPUT_0_NB_CHANNELS 32
#define _3_PADCONV_1_OUTPUT_0_IN_HEIGHT 160
#define _3_PADCONV_1_OUTPUT_0_IN_WIDTH 160

// OUTPUT CONF
#define _5_MUL_1_OUTPUT_0_NB_OUTPUTS 32
#define _5_MUL_1_OUTPUT_0_OUT_HEIGHT 160
#define _5_MUL_1_OUTPUT_0_OUT_WIDTH 160
#define _5_MUL_1_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _5_MUL_1_OUTPUT_0_MEM_SIZE 128
#define _5_MUL_1_OUTPUT_0_MEM_OFFSET 6553600
#define _5_MUL_1_OUTPUT_0_MEM_STRIDE 128
#define _5_MUL_1_OUTPUT_0_MEM_LENGTH 160
#define _5_MUL_1_OUTPUT_0_MEM_CONT_OFFSET 0
#define _5_MUL_1_OUTPUT_0_MEM_CONT_SIZE 0
#define _5_MUL_1_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _5_MUL_1_OUTPUT_0_MEM_WRAP_SIZE 0


#define _5_MUL_1_NB_MAT 1
#define _5_MUL_1_INPUT1_CONT_SIZE 819200
#define _5_MUL_1_INPUT2_CONT_SIZE 819200
#define _5_MUL_1_OUTPUT_CONT_SIZE 819200

constexpr size_t _5_MUL_1_OFFSET_IN1[] =  { 0 };
constexpr size_t _5_MUL_1_OFFSET_IN2[] = { 0 };

#define _5_MUL_1_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _5_MUL_1_ELEM_OP export_cpp::ElemWise_T::Mul

#define _5_MUL_1_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _5_MUL_1_RESCALING = {};

#endif /* _5_MUL_1_LAYER_H */
