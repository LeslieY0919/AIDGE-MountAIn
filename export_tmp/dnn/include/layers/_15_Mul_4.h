#ifndef _15_MUL_4_LAYER_H
#define _15_MUL_4_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _13_PADCONV_3_OUTPUT_0_IN_BATCH 1
#define _13_PADCONV_3_OUTPUT_0_NB_CHANNELS 16
#define _13_PADCONV_3_OUTPUT_0_IN_HEIGHT 80
#define _13_PADCONV_3_OUTPUT_0_IN_WIDTH 80

// OUTPUT CONF
#define _15_MUL_4_OUTPUT_0_NB_OUTPUTS 16
#define _15_MUL_4_OUTPUT_0_OUT_HEIGHT 80
#define _15_MUL_4_OUTPUT_0_OUT_WIDTH 80
#define _15_MUL_4_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _15_MUL_4_OUTPUT_0_MEM_SIZE 64
#define _15_MUL_4_OUTPUT_0_MEM_OFFSET 1638400
#define _15_MUL_4_OUTPUT_0_MEM_STRIDE 64
#define _15_MUL_4_OUTPUT_0_MEM_LENGTH 80
#define _15_MUL_4_OUTPUT_0_MEM_CONT_OFFSET 0
#define _15_MUL_4_OUTPUT_0_MEM_CONT_SIZE 0
#define _15_MUL_4_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _15_MUL_4_OUTPUT_0_MEM_WRAP_SIZE 0


#define _15_MUL_4_NB_MAT 1
#define _15_MUL_4_INPUT1_CONT_SIZE 102400
#define _15_MUL_4_INPUT2_CONT_SIZE 102400
#define _15_MUL_4_OUTPUT_CONT_SIZE 102400

constexpr size_t _15_MUL_4_OFFSET_IN1[] =  { 0 };
constexpr size_t _15_MUL_4_OFFSET_IN2[] = { 0 };

#define _15_MUL_4_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _15_MUL_4_ELEM_OP export_cpp::ElemWise_T::Mul

#define _15_MUL_4_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _15_MUL_4_RESCALING = {};

#endif /* _15_MUL_4_LAYER_H */
