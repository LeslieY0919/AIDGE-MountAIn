#ifndef _82_MUL_22_LAYER_H
#define _82_MUL_22_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _80_PADCONV_13_OUTPUT_0_IN_BATCH 1
#define _80_PADCONV_13_OUTPUT_0_NB_CHANNELS 128
#define _80_PADCONV_13_OUTPUT_0_IN_HEIGHT 10
#define _80_PADCONV_13_OUTPUT_0_IN_WIDTH 10

// OUTPUT CONF
#define _82_MUL_22_OUTPUT_0_NB_OUTPUTS 128
#define _82_MUL_22_OUTPUT_0_OUT_HEIGHT 10
#define _82_MUL_22_OUTPUT_0_OUT_WIDTH 10
#define _82_MUL_22_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _82_MUL_22_OUTPUT_0_MEM_SIZE 512
#define _82_MUL_22_OUTPUT_0_MEM_OFFSET 204800
#define _82_MUL_22_OUTPUT_0_MEM_STRIDE 512
#define _82_MUL_22_OUTPUT_0_MEM_LENGTH 10
#define _82_MUL_22_OUTPUT_0_MEM_CONT_OFFSET 0
#define _82_MUL_22_OUTPUT_0_MEM_CONT_SIZE 0
#define _82_MUL_22_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _82_MUL_22_OUTPUT_0_MEM_WRAP_SIZE 0


#define _82_MUL_22_NB_MAT 1
#define _82_MUL_22_INPUT1_CONT_SIZE 12800
#define _82_MUL_22_INPUT2_CONT_SIZE 12800
#define _82_MUL_22_OUTPUT_CONT_SIZE 12800

constexpr size_t _82_MUL_22_OFFSET_IN1[] =  { 0 };
constexpr size_t _82_MUL_22_OFFSET_IN2[] = { 0 };

#define _82_MUL_22_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _82_MUL_22_ELEM_OP export_cpp::ElemWise_T::Mul

#define _82_MUL_22_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _82_MUL_22_RESCALING = {};

#endif /* _82_MUL_22_LAYER_H */
