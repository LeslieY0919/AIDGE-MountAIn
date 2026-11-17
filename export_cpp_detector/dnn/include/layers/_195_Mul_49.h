#ifndef _195_MUL_49_LAYER_H
#define _195_MUL_49_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _190_CONV2D_26_OUTPUT_0_IN_BATCH 1
#define _190_CONV2D_26_OUTPUT_0_NB_CHANNELS 128
#define _190_CONV2D_26_OUTPUT_0_IN_HEIGHT 40
#define _190_CONV2D_26_OUTPUT_0_IN_WIDTH 40

// OUTPUT CONF
#define _195_MUL_49_OUTPUT_0_NB_OUTPUTS 128
#define _195_MUL_49_OUTPUT_0_OUT_HEIGHT 40
#define _195_MUL_49_OUTPUT_0_OUT_WIDTH 40
#define _195_MUL_49_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _195_MUL_49_OUTPUT_0_MEM_SIZE 512
#define _195_MUL_49_OUTPUT_0_MEM_OFFSET 6144000
#define _195_MUL_49_OUTPUT_0_MEM_STRIDE 512
#define _195_MUL_49_OUTPUT_0_MEM_LENGTH 40
#define _195_MUL_49_OUTPUT_0_MEM_CONT_OFFSET 0
#define _195_MUL_49_OUTPUT_0_MEM_CONT_SIZE 0
#define _195_MUL_49_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _195_MUL_49_OUTPUT_0_MEM_WRAP_SIZE 0


#define _195_MUL_49_NB_MAT 1
#define _195_MUL_49_INPUT1_CONT_SIZE 204800
#define _195_MUL_49_INPUT2_CONT_SIZE 204800
#define _195_MUL_49_OUTPUT_CONT_SIZE 204800

constexpr size_t _195_MUL_49_OFFSET_IN1[] =  { 0 };
constexpr size_t _195_MUL_49_OFFSET_IN2[] = { 0 };

#define _195_MUL_49_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _195_MUL_49_ELEM_OP export_cpp::ElemWise_T::Mul

#define _195_MUL_49_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _195_MUL_49_RESCALING = {};

#endif /* _195_MUL_49_LAYER_H */
