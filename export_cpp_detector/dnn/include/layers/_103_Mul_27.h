#ifndef _103_MUL_27_LAYER_H
#define _103_MUL_27_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _101_CONV2D_12_OUTPUT_0_IN_BATCH 1
#define _101_CONV2D_12_OUTPUT_0_NB_CHANNELS 256
#define _101_CONV2D_12_OUTPUT_0_IN_HEIGHT 20
#define _101_CONV2D_12_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _103_MUL_27_OUTPUT_0_NB_OUTPUTS 256
#define _103_MUL_27_OUTPUT_0_OUT_HEIGHT 20
#define _103_MUL_27_OUTPUT_0_OUT_WIDTH 20
#define _103_MUL_27_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _103_MUL_27_OUTPUT_0_MEM_SIZE 1024
#define _103_MUL_27_OUTPUT_0_MEM_OFFSET 4915200
#define _103_MUL_27_OUTPUT_0_MEM_STRIDE 1024
#define _103_MUL_27_OUTPUT_0_MEM_LENGTH 20
#define _103_MUL_27_OUTPUT_0_MEM_CONT_OFFSET 0
#define _103_MUL_27_OUTPUT_0_MEM_CONT_SIZE 0
#define _103_MUL_27_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _103_MUL_27_OUTPUT_0_MEM_WRAP_SIZE 0


#define _103_MUL_27_NB_MAT 1
#define _103_MUL_27_INPUT1_CONT_SIZE 102400
#define _103_MUL_27_INPUT2_CONT_SIZE 102400
#define _103_MUL_27_OUTPUT_CONT_SIZE 102400

constexpr size_t _103_MUL_27_OFFSET_IN1[] =  { 0 };
constexpr size_t _103_MUL_27_OFFSET_IN2[] = { 0 };

#define _103_MUL_27_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _103_MUL_27_ELEM_OP export_cpp::ElemWise_T::Mul

#define _103_MUL_27_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _103_MUL_27_RESCALING = {};

#endif /* _103_MUL_27_LAYER_H */
