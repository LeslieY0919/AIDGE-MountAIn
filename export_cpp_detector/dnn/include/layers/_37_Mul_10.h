#ifndef _37_MUL_10_LAYER_H
#define _37_MUL_10_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _35_PADCONV_7_OUTPUT_0_IN_BATCH 1
#define _35_PADCONV_7_OUTPUT_0_NB_CHANNELS 32
#define _35_PADCONV_7_OUTPUT_0_IN_HEIGHT 80
#define _35_PADCONV_7_OUTPUT_0_IN_WIDTH 80

// OUTPUT CONF
#define _37_MUL_10_OUTPUT_0_NB_OUTPUTS 32
#define _37_MUL_10_OUTPUT_0_OUT_HEIGHT 80
#define _37_MUL_10_OUTPUT_0_OUT_WIDTH 80
#define _37_MUL_10_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _37_MUL_10_OUTPUT_0_MEM_SIZE 128
#define _37_MUL_10_OUTPUT_0_MEM_OFFSET 4096000
#define _37_MUL_10_OUTPUT_0_MEM_STRIDE 128
#define _37_MUL_10_OUTPUT_0_MEM_LENGTH 80
#define _37_MUL_10_OUTPUT_0_MEM_CONT_OFFSET 0
#define _37_MUL_10_OUTPUT_0_MEM_CONT_SIZE 0
#define _37_MUL_10_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _37_MUL_10_OUTPUT_0_MEM_WRAP_SIZE 0


#define _37_MUL_10_NB_MAT 1
#define _37_MUL_10_INPUT1_CONT_SIZE 204800
#define _37_MUL_10_INPUT2_CONT_SIZE 204800
#define _37_MUL_10_OUTPUT_CONT_SIZE 204800

constexpr size_t _37_MUL_10_OFFSET_IN1[] =  { 0 };
constexpr size_t _37_MUL_10_OFFSET_IN2[] = { 0 };

#define _37_MUL_10_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _37_MUL_10_ELEM_OP export_cpp::ElemWise_T::Mul

#define _37_MUL_10_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _37_MUL_10_RESCALING = {};

#endif /* _37_MUL_10_LAYER_H */
