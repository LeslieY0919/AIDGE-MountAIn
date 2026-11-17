#ifndef _40_MUL_11_LAYER_H
#define _40_MUL_11_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _38_PADCONV_8_OUTPUT_0_IN_BATCH 1
#define _38_PADCONV_8_OUTPUT_0_NB_CHANNELS 32
#define _38_PADCONV_8_OUTPUT_0_IN_HEIGHT 40
#define _38_PADCONV_8_OUTPUT_0_IN_WIDTH 40

// OUTPUT CONF
#define _40_MUL_11_OUTPUT_0_NB_OUTPUTS 32
#define _40_MUL_11_OUTPUT_0_OUT_HEIGHT 40
#define _40_MUL_11_OUTPUT_0_OUT_WIDTH 40
#define _40_MUL_11_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _40_MUL_11_OUTPUT_0_MEM_SIZE 128
#define _40_MUL_11_OUTPUT_0_MEM_OFFSET 1024000
#define _40_MUL_11_OUTPUT_0_MEM_STRIDE 128
#define _40_MUL_11_OUTPUT_0_MEM_LENGTH 40
#define _40_MUL_11_OUTPUT_0_MEM_CONT_OFFSET 0
#define _40_MUL_11_OUTPUT_0_MEM_CONT_SIZE 0
#define _40_MUL_11_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _40_MUL_11_OUTPUT_0_MEM_WRAP_SIZE 0


#define _40_MUL_11_NB_MAT 1
#define _40_MUL_11_INPUT1_CONT_SIZE 51200
#define _40_MUL_11_INPUT2_CONT_SIZE 51200
#define _40_MUL_11_OUTPUT_CONT_SIZE 51200

constexpr size_t _40_MUL_11_OFFSET_IN1[] =  { 0 };
constexpr size_t _40_MUL_11_OFFSET_IN2[] = { 0 };

#define _40_MUL_11_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _40_MUL_11_ELEM_OP export_cpp::ElemWise_T::Mul

#define _40_MUL_11_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _40_MUL_11_RESCALING = {};

#endif /* _40_MUL_11_LAYER_H */
