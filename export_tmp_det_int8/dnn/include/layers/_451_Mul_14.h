#ifndef _451_MUL_14_LAYER_H
#define _451_MUL_14_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _446_QUANTIZER_147_OUTPUT_0_IN_BATCH 1
#define _446_QUANTIZER_147_OUTPUT_0_NB_CHANNELS 128
#define _446_QUANTIZER_147_OUTPUT_0_IN_HEIGHT 40
#define _446_QUANTIZER_147_OUTPUT_0_IN_WIDTH 40

// OUTPUT CONF
#define _451_MUL_14_OUTPUT_0_NB_OUTPUTS 128
#define _451_MUL_14_OUTPUT_0_OUT_HEIGHT 40
#define _451_MUL_14_OUTPUT_0_OUT_WIDTH 40
#define _451_MUL_14_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _451_MUL_14_OUTPUT_0_MEM_SIZE 512
#define _451_MUL_14_OUTPUT_0_MEM_OFFSET 2457600
#define _451_MUL_14_OUTPUT_0_MEM_STRIDE 512
#define _451_MUL_14_OUTPUT_0_MEM_LENGTH 40
#define _451_MUL_14_OUTPUT_0_MEM_CONT_OFFSET 0
#define _451_MUL_14_OUTPUT_0_MEM_CONT_SIZE 0
#define _451_MUL_14_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _451_MUL_14_OUTPUT_0_MEM_WRAP_SIZE 0


#define _451_MUL_14_NB_MAT 1
#define _451_MUL_14_INPUT1_CONT_SIZE 204800
#define _451_MUL_14_INPUT2_CONT_SIZE 204800
#define _451_MUL_14_OUTPUT_CONT_SIZE 204800

constexpr size_t _451_MUL_14_OFFSET_IN1[] =  { 0 };
constexpr size_t _451_MUL_14_OFFSET_IN2[] = { 0 };

#define _451_MUL_14_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _451_MUL_14_ELEM_OP export_cpp::ElemWise_T::Mul

#define _451_MUL_14_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _451_MUL_14_RESCALING = {};

#endif /* _451_MUL_14_LAYER_H */
