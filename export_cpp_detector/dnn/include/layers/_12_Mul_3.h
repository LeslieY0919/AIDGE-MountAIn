#ifndef _12_MUL_3_LAYER_H
#define _12_MUL_3_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _10_PADCONV_2_OUTPUT_0_IN_BATCH 1
#define _10_PADCONV_2_OUTPUT_0_NB_CHANNELS 16
#define _10_PADCONV_2_OUTPUT_0_IN_HEIGHT 160
#define _10_PADCONV_2_OUTPUT_0_IN_WIDTH 160

// OUTPUT CONF
#define _12_MUL_3_OUTPUT_0_NB_OUTPUTS 16
#define _12_MUL_3_OUTPUT_0_OUT_HEIGHT 160
#define _12_MUL_3_OUTPUT_0_OUT_WIDTH 160
#define _12_MUL_3_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _12_MUL_3_OUTPUT_0_MEM_SIZE 64
#define _12_MUL_3_OUTPUT_0_MEM_OFFSET 6553600
#define _12_MUL_3_OUTPUT_0_MEM_STRIDE 64
#define _12_MUL_3_OUTPUT_0_MEM_LENGTH 160
#define _12_MUL_3_OUTPUT_0_MEM_CONT_OFFSET 0
#define _12_MUL_3_OUTPUT_0_MEM_CONT_SIZE 0
#define _12_MUL_3_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _12_MUL_3_OUTPUT_0_MEM_WRAP_SIZE 0


#define _12_MUL_3_NB_MAT 1
#define _12_MUL_3_INPUT1_CONT_SIZE 409600
#define _12_MUL_3_INPUT2_CONT_SIZE 409600
#define _12_MUL_3_OUTPUT_CONT_SIZE 409600

constexpr size_t _12_MUL_3_OFFSET_IN1[] =  { 0 };
constexpr size_t _12_MUL_3_OFFSET_IN2[] = { 0 };

#define _12_MUL_3_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _12_MUL_3_ELEM_OP export_cpp::ElemWise_T::Mul

#define _12_MUL_3_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _12_MUL_3_RESCALING = {};

#endif /* _12_MUL_3_LAYER_H */
