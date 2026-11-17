#ifndef _59_MUL_16_LAYER_H
#define _59_MUL_16_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _57_PADCONV_10_OUTPUT_0_IN_BATCH 1
#define _57_PADCONV_10_OUTPUT_0_NB_CHANNELS 64
#define _57_PADCONV_10_OUTPUT_0_IN_HEIGHT 40
#define _57_PADCONV_10_OUTPUT_0_IN_WIDTH 40

// OUTPUT CONF
#define _59_MUL_16_OUTPUT_0_NB_OUTPUTS 64
#define _59_MUL_16_OUTPUT_0_OUT_HEIGHT 40
#define _59_MUL_16_OUTPUT_0_OUT_WIDTH 40
#define _59_MUL_16_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _59_MUL_16_OUTPUT_0_MEM_SIZE 256
#define _59_MUL_16_OUTPUT_0_MEM_OFFSET 1638400
#define _59_MUL_16_OUTPUT_0_MEM_STRIDE 256
#define _59_MUL_16_OUTPUT_0_MEM_LENGTH 40
#define _59_MUL_16_OUTPUT_0_MEM_CONT_OFFSET 0
#define _59_MUL_16_OUTPUT_0_MEM_CONT_SIZE 0
#define _59_MUL_16_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _59_MUL_16_OUTPUT_0_MEM_WRAP_SIZE 0


#define _59_MUL_16_NB_MAT 1
#define _59_MUL_16_INPUT1_CONT_SIZE 102400
#define _59_MUL_16_INPUT2_CONT_SIZE 102400
#define _59_MUL_16_OUTPUT_CONT_SIZE 102400

constexpr size_t _59_MUL_16_OFFSET_IN1[] =  { 0 };
constexpr size_t _59_MUL_16_OFFSET_IN2[] = { 0 };

#define _59_MUL_16_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _59_MUL_16_ELEM_OP export_cpp::ElemWise_T::Mul

#define _59_MUL_16_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _59_MUL_16_RESCALING = {};

#endif /* _59_MUL_16_LAYER_H */
