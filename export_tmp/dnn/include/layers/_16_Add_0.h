#ifndef _16_ADD_0_LAYER_H
#define _16_ADD_0_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _9_SPLIT_0_OUTPUT_1_IN_BATCH 1
#define _9_SPLIT_0_OUTPUT_1_NB_CHANNELS 16
#define _9_SPLIT_0_OUTPUT_1_IN_HEIGHT 80
#define _9_SPLIT_0_OUTPUT_1_IN_WIDTH 80

// OUTPUT CONF
#define _16_ADD_0_OUTPUT_0_NB_OUTPUTS 16
#define _16_ADD_0_OUTPUT_0_OUT_HEIGHT 80
#define _16_ADD_0_OUTPUT_0_OUT_WIDTH 80
#define _16_ADD_0_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _16_ADD_0_OUTPUT_0_MEM_SIZE 64
#define _16_ADD_0_OUTPUT_0_MEM_OFFSET 819200
#define _16_ADD_0_OUTPUT_0_MEM_STRIDE 64
#define _16_ADD_0_OUTPUT_0_MEM_LENGTH 80
#define _16_ADD_0_OUTPUT_0_MEM_CONT_OFFSET 0
#define _16_ADD_0_OUTPUT_0_MEM_CONT_SIZE 0
#define _16_ADD_0_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _16_ADD_0_OUTPUT_0_MEM_WRAP_SIZE 0


#define _16_ADD_0_NB_MAT 1
#define _16_ADD_0_INPUT1_CONT_SIZE 102400
#define _16_ADD_0_INPUT2_CONT_SIZE 102400
#define _16_ADD_0_OUTPUT_CONT_SIZE 102400

constexpr size_t _16_ADD_0_OFFSET_IN1[] =  { 0 };
constexpr size_t _16_ADD_0_OFFSET_IN2[] = { 0 };

#define _16_ADD_0_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _16_ADD_0_ELEM_OP export_cpp::ElemWise_T::Add

#define _16_ADD_0_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _16_ADD_0_RESCALING = {};

#endif /* _16_ADD_0_LAYER_H */
