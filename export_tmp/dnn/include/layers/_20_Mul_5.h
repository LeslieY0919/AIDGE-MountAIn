#ifndef _20_MUL_5_LAYER_H
#define _20_MUL_5_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _18_CONV2D_1_OUTPUT_0_IN_BATCH 1
#define _18_CONV2D_1_OUTPUT_0_NB_CHANNELS 32
#define _18_CONV2D_1_OUTPUT_0_IN_HEIGHT 80
#define _18_CONV2D_1_OUTPUT_0_IN_WIDTH 80

// OUTPUT CONF
#define _20_MUL_5_OUTPUT_0_NB_OUTPUTS 32
#define _20_MUL_5_OUTPUT_0_OUT_HEIGHT 80
#define _20_MUL_5_OUTPUT_0_OUT_WIDTH 80
#define _20_MUL_5_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _20_MUL_5_OUTPUT_0_MEM_SIZE 128
#define _20_MUL_5_OUTPUT_0_MEM_OFFSET 1638400
#define _20_MUL_5_OUTPUT_0_MEM_STRIDE 128
#define _20_MUL_5_OUTPUT_0_MEM_LENGTH 80
#define _20_MUL_5_OUTPUT_0_MEM_CONT_OFFSET 0
#define _20_MUL_5_OUTPUT_0_MEM_CONT_SIZE 0
#define _20_MUL_5_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _20_MUL_5_OUTPUT_0_MEM_WRAP_SIZE 0


#define _20_MUL_5_NB_MAT 1
#define _20_MUL_5_INPUT1_CONT_SIZE 204800
#define _20_MUL_5_INPUT2_CONT_SIZE 204800
#define _20_MUL_5_OUTPUT_CONT_SIZE 204800

constexpr size_t _20_MUL_5_OFFSET_IN1[] =  { 0 };
constexpr size_t _20_MUL_5_OFFSET_IN2[] = { 0 };

#define _20_MUL_5_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _20_MUL_5_ELEM_OP export_cpp::ElemWise_T::Mul

#define _20_MUL_5_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _20_MUL_5_RESCALING = {};

#endif /* _20_MUL_5_LAYER_H */
