#ifndef _48_MUL_13_LAYER_H
#define _48_MUL_13_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _46_CONV2D_4_OUTPUT_0_IN_BATCH 1
#define _46_CONV2D_4_OUTPUT_0_NB_CHANNELS 128
#define _46_CONV2D_4_OUTPUT_0_IN_HEIGHT 80
#define _46_CONV2D_4_OUTPUT_0_IN_WIDTH 80

// OUTPUT CONF
#define _48_MUL_13_OUTPUT_0_NB_OUTPUTS 128
#define _48_MUL_13_OUTPUT_0_OUT_HEIGHT 80
#define _48_MUL_13_OUTPUT_0_OUT_WIDTH 80
#define _48_MUL_13_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _48_MUL_13_OUTPUT_0_MEM_SIZE 512
#define _48_MUL_13_OUTPUT_0_MEM_OFFSET 8192000
#define _48_MUL_13_OUTPUT_0_MEM_STRIDE 512
#define _48_MUL_13_OUTPUT_0_MEM_LENGTH 80
#define _48_MUL_13_OUTPUT_0_MEM_CONT_OFFSET 0
#define _48_MUL_13_OUTPUT_0_MEM_CONT_SIZE 0
#define _48_MUL_13_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _48_MUL_13_OUTPUT_0_MEM_WRAP_SIZE 0


#define _48_MUL_13_NB_MAT 1
#define _48_MUL_13_INPUT1_CONT_SIZE 819200
#define _48_MUL_13_INPUT2_CONT_SIZE 819200
#define _48_MUL_13_OUTPUT_CONT_SIZE 819200

constexpr size_t _48_MUL_13_OFFSET_IN1[] =  { 0 };
constexpr size_t _48_MUL_13_OFFSET_IN2[] = { 0 };

#define _48_MUL_13_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _48_MUL_13_ELEM_OP export_cpp::ElemWise_T::Mul

#define _48_MUL_13_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _48_MUL_13_RESCALING = {};

#endif /* _48_MUL_13_LAYER_H */
