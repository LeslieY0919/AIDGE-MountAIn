#ifndef _45_MUL_12_LAYER_H
#define _45_MUL_12_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _43_CONV2D_3_OUTPUT_0_IN_BATCH 1
#define _43_CONV2D_3_OUTPUT_0_NB_CHANNELS 64
#define _43_CONV2D_3_OUTPUT_0_IN_HEIGHT 80
#define _43_CONV2D_3_OUTPUT_0_IN_WIDTH 80

// OUTPUT CONF
#define _45_MUL_12_OUTPUT_0_NB_OUTPUTS 64
#define _45_MUL_12_OUTPUT_0_OUT_HEIGHT 80
#define _45_MUL_12_OUTPUT_0_OUT_WIDTH 80
#define _45_MUL_12_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _45_MUL_12_OUTPUT_0_MEM_SIZE 256
#define _45_MUL_12_OUTPUT_0_MEM_OFFSET 3276800
#define _45_MUL_12_OUTPUT_0_MEM_STRIDE 256
#define _45_MUL_12_OUTPUT_0_MEM_LENGTH 80
#define _45_MUL_12_OUTPUT_0_MEM_CONT_OFFSET 0
#define _45_MUL_12_OUTPUT_0_MEM_CONT_SIZE 0
#define _45_MUL_12_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _45_MUL_12_OUTPUT_0_MEM_WRAP_SIZE 0


#define _45_MUL_12_NB_MAT 1
#define _45_MUL_12_INPUT1_CONT_SIZE 409600
#define _45_MUL_12_INPUT2_CONT_SIZE 409600
#define _45_MUL_12_OUTPUT_CONT_SIZE 409600

constexpr size_t _45_MUL_12_OFFSET_IN1[] =  { 0 };
constexpr size_t _45_MUL_12_OFFSET_IN2[] = { 0 };

#define _45_MUL_12_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _45_MUL_12_ELEM_OP export_cpp::ElemWise_T::Mul

#define _45_MUL_12_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _45_MUL_12_RESCALING = {};

#endif /* _45_MUL_12_LAYER_H */
