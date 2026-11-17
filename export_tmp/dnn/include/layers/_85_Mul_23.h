#ifndef _85_MUL_23_LAYER_H
#define _85_MUL_23_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _83_PADCONV_14_OUTPUT_0_IN_BATCH 1
#define _83_PADCONV_14_OUTPUT_0_NB_CHANNELS 128
#define _83_PADCONV_14_OUTPUT_0_IN_HEIGHT 10
#define _83_PADCONV_14_OUTPUT_0_IN_WIDTH 10

// OUTPUT CONF
#define _85_MUL_23_OUTPUT_0_NB_OUTPUTS 128
#define _85_MUL_23_OUTPUT_0_OUT_HEIGHT 10
#define _85_MUL_23_OUTPUT_0_OUT_WIDTH 10
#define _85_MUL_23_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _85_MUL_23_OUTPUT_0_MEM_SIZE 512
#define _85_MUL_23_OUTPUT_0_MEM_OFFSET 204800
#define _85_MUL_23_OUTPUT_0_MEM_STRIDE 512
#define _85_MUL_23_OUTPUT_0_MEM_LENGTH 10
#define _85_MUL_23_OUTPUT_0_MEM_CONT_OFFSET 0
#define _85_MUL_23_OUTPUT_0_MEM_CONT_SIZE 0
#define _85_MUL_23_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _85_MUL_23_OUTPUT_0_MEM_WRAP_SIZE 0


#define _85_MUL_23_NB_MAT 1
#define _85_MUL_23_INPUT1_CONT_SIZE 12800
#define _85_MUL_23_INPUT2_CONT_SIZE 12800
#define _85_MUL_23_OUTPUT_CONT_SIZE 12800

constexpr size_t _85_MUL_23_OFFSET_IN1[] =  { 0 };
constexpr size_t _85_MUL_23_OFFSET_IN2[] = { 0 };

#define _85_MUL_23_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _85_MUL_23_ELEM_OP export_cpp::ElemWise_T::Mul

#define _85_MUL_23_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _85_MUL_23_RESCALING = {};

#endif /* _85_MUL_23_LAYER_H */
