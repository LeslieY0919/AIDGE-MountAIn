#ifndef _41_ADD_2_LAYER_H
#define _41_ADD_2_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _34_ADD_1_OUTPUT_0_IN_BATCH 1
#define _34_ADD_1_OUTPUT_0_NB_CHANNELS 32
#define _34_ADD_1_OUTPUT_0_IN_HEIGHT 40
#define _34_ADD_1_OUTPUT_0_IN_WIDTH 40

// OUTPUT CONF
#define _41_ADD_2_OUTPUT_0_NB_OUTPUTS 32
#define _41_ADD_2_OUTPUT_0_OUT_HEIGHT 40
#define _41_ADD_2_OUTPUT_0_OUT_WIDTH 40
#define _41_ADD_2_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _41_ADD_2_OUTPUT_0_MEM_SIZE 128
#define _41_ADD_2_OUTPUT_0_MEM_OFFSET 614400
#define _41_ADD_2_OUTPUT_0_MEM_STRIDE 128
#define _41_ADD_2_OUTPUT_0_MEM_LENGTH 40
#define _41_ADD_2_OUTPUT_0_MEM_CONT_OFFSET 0
#define _41_ADD_2_OUTPUT_0_MEM_CONT_SIZE 0
#define _41_ADD_2_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _41_ADD_2_OUTPUT_0_MEM_WRAP_SIZE 0


#define _41_ADD_2_NB_MAT 1
#define _41_ADD_2_INPUT1_CONT_SIZE 51200
#define _41_ADD_2_INPUT2_CONT_SIZE 51200
#define _41_ADD_2_OUTPUT_CONT_SIZE 51200

constexpr size_t _41_ADD_2_OFFSET_IN1[] =  { 0 };
constexpr size_t _41_ADD_2_OFFSET_IN2[] = { 0 };

#define _41_ADD_2_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _41_ADD_2_ELEM_OP export_cpp::ElemWise_T::Add

#define _41_ADD_2_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _41_ADD_2_RESCALING = {};

#endif /* _41_ADD_2_LAYER_H */
