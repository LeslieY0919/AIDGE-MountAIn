#ifndef _60_ADD_3_LAYER_H
#define _60_ADD_3_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _53_SPLIT_2_OUTPUT_1_IN_BATCH 1
#define _53_SPLIT_2_OUTPUT_1_NB_CHANNELS 64
#define _53_SPLIT_2_OUTPUT_1_IN_HEIGHT 40
#define _53_SPLIT_2_OUTPUT_1_IN_WIDTH 40

// OUTPUT CONF
#define _60_ADD_3_OUTPUT_0_NB_OUTPUTS 64
#define _60_ADD_3_OUTPUT_0_OUT_HEIGHT 40
#define _60_ADD_3_OUTPUT_0_OUT_WIDTH 40
#define _60_ADD_3_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _60_ADD_3_OUTPUT_0_MEM_SIZE 256
#define _60_ADD_3_OUTPUT_0_MEM_OFFSET 819200
#define _60_ADD_3_OUTPUT_0_MEM_STRIDE 256
#define _60_ADD_3_OUTPUT_0_MEM_LENGTH 40
#define _60_ADD_3_OUTPUT_0_MEM_CONT_OFFSET 0
#define _60_ADD_3_OUTPUT_0_MEM_CONT_SIZE 0
#define _60_ADD_3_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _60_ADD_3_OUTPUT_0_MEM_WRAP_SIZE 0


#define _60_ADD_3_NB_MAT 1
#define _60_ADD_3_INPUT1_CONT_SIZE 102400
#define _60_ADD_3_INPUT2_CONT_SIZE 102400
#define _60_ADD_3_OUTPUT_CONT_SIZE 102400

constexpr size_t _60_ADD_3_OFFSET_IN1[] =  { 0 };
constexpr size_t _60_ADD_3_OFFSET_IN2[] = { 0 };

#define _60_ADD_3_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _60_ADD_3_ELEM_OP export_cpp::ElemWise_T::Add

#define _60_ADD_3_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _60_ADD_3_RESCALING = {};

#endif /* _60_ADD_3_LAYER_H */
