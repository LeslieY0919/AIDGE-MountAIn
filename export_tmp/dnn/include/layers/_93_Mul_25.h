#ifndef _93_MUL_25_LAYER_H
#define _93_MUL_25_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _91_CONV2D_10_OUTPUT_0_IN_BATCH 1
#define _91_CONV2D_10_OUTPUT_0_NB_CHANNELS 128
#define _91_CONV2D_10_OUTPUT_0_IN_HEIGHT 10
#define _91_CONV2D_10_OUTPUT_0_IN_WIDTH 10

// OUTPUT CONF
#define _93_MUL_25_OUTPUT_0_NB_OUTPUTS 128
#define _93_MUL_25_OUTPUT_0_OUT_HEIGHT 10
#define _93_MUL_25_OUTPUT_0_OUT_WIDTH 10
#define _93_MUL_25_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _93_MUL_25_OUTPUT_0_MEM_SIZE 512
#define _93_MUL_25_OUTPUT_0_MEM_OFFSET 102400
#define _93_MUL_25_OUTPUT_0_MEM_STRIDE 512
#define _93_MUL_25_OUTPUT_0_MEM_LENGTH 10
#define _93_MUL_25_OUTPUT_0_MEM_CONT_OFFSET 0
#define _93_MUL_25_OUTPUT_0_MEM_CONT_SIZE 0
#define _93_MUL_25_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _93_MUL_25_OUTPUT_0_MEM_WRAP_SIZE 0


#define _93_MUL_25_NB_MAT 1
#define _93_MUL_25_INPUT1_CONT_SIZE 12800
#define _93_MUL_25_INPUT2_CONT_SIZE 12800
#define _93_MUL_25_OUTPUT_CONT_SIZE 12800

constexpr size_t _93_MUL_25_OFFSET_IN1[] =  { 0 };
constexpr size_t _93_MUL_25_OFFSET_IN2[] = { 0 };

#define _93_MUL_25_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _93_MUL_25_ELEM_OP export_cpp::ElemWise_T::Mul

#define _93_MUL_25_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _93_MUL_25_RESCALING = {};

#endif /* _93_MUL_25_LAYER_H */
