#ifndef _63_MUL_17_LAYER_H
#define _63_MUL_17_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _61_PADCONV_11_OUTPUT_0_IN_BATCH 1
#define _61_PADCONV_11_OUTPUT_0_NB_CHANNELS 64
#define _61_PADCONV_11_OUTPUT_0_IN_HEIGHT 20
#define _61_PADCONV_11_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _63_MUL_17_OUTPUT_0_NB_OUTPUTS 64
#define _63_MUL_17_OUTPUT_0_OUT_HEIGHT 20
#define _63_MUL_17_OUTPUT_0_OUT_WIDTH 20
#define _63_MUL_17_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _63_MUL_17_OUTPUT_0_MEM_SIZE 256
#define _63_MUL_17_OUTPUT_0_MEM_OFFSET 512000
#define _63_MUL_17_OUTPUT_0_MEM_STRIDE 256
#define _63_MUL_17_OUTPUT_0_MEM_LENGTH 20
#define _63_MUL_17_OUTPUT_0_MEM_CONT_OFFSET 0
#define _63_MUL_17_OUTPUT_0_MEM_CONT_SIZE 0
#define _63_MUL_17_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _63_MUL_17_OUTPUT_0_MEM_WRAP_SIZE 0


#define _63_MUL_17_NB_MAT 1
#define _63_MUL_17_INPUT1_CONT_SIZE 25600
#define _63_MUL_17_INPUT2_CONT_SIZE 25600
#define _63_MUL_17_OUTPUT_CONT_SIZE 25600

constexpr size_t _63_MUL_17_OFFSET_IN1[] =  { 0 };
constexpr size_t _63_MUL_17_OFFSET_IN2[] = { 0 };

#define _63_MUL_17_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _63_MUL_17_ELEM_OP export_cpp::ElemWise_T::Mul

#define _63_MUL_17_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _63_MUL_17_RESCALING = {};

#endif /* _63_MUL_17_LAYER_H */
