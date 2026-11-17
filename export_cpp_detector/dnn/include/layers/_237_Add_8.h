#ifndef _237_ADD_8_LAYER_H
#define _237_ADD_8_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _213_SPLIT_8_OUTPUT_1_IN_BATCH 1
#define _213_SPLIT_8_OUTPUT_1_NB_CHANNELS 128
#define _213_SPLIT_8_OUTPUT_1_IN_HEIGHT 20
#define _213_SPLIT_8_OUTPUT_1_IN_WIDTH 20

// OUTPUT CONF
#define _237_ADD_8_OUTPUT_0_NB_OUTPUTS 128
#define _237_ADD_8_OUTPUT_0_OUT_HEIGHT 20
#define _237_ADD_8_OUTPUT_0_OUT_WIDTH 20
#define _237_ADD_8_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _237_ADD_8_OUTPUT_0_MEM_SIZE 512
#define _237_ADD_8_OUTPUT_0_MEM_OFFSET 1433600
#define _237_ADD_8_OUTPUT_0_MEM_STRIDE 512
#define _237_ADD_8_OUTPUT_0_MEM_LENGTH 20
#define _237_ADD_8_OUTPUT_0_MEM_CONT_OFFSET 0
#define _237_ADD_8_OUTPUT_0_MEM_CONT_SIZE 0
#define _237_ADD_8_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _237_ADD_8_OUTPUT_0_MEM_WRAP_SIZE 0


#define _237_ADD_8_NB_MAT 1
#define _237_ADD_8_INPUT1_CONT_SIZE 51200
#define _237_ADD_8_INPUT2_CONT_SIZE 51200
#define _237_ADD_8_OUTPUT_CONT_SIZE 51200

constexpr size_t _237_ADD_8_OFFSET_IN1[] =  { 0 };
constexpr size_t _237_ADD_8_OFFSET_IN2[] = { 0 };

#define _237_ADD_8_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _237_ADD_8_ELEM_OP export_cpp::ElemWise_T::Add

#define _237_ADD_8_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _237_ADD_8_RESCALING = {};

#endif /* _237_ADD_8_LAYER_H */
