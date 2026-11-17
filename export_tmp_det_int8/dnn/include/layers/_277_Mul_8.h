#ifndef _277_MUL_8_LAYER_H
#define _277_MUL_8_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _273_QUANTIZER_91_OUTPUT_0_IN_BATCH 1
#define _273_QUANTIZER_91_OUTPUT_0_NB_CHANNELS 256
#define _273_QUANTIZER_91_OUTPUT_0_IN_HEIGHT 20
#define _273_QUANTIZER_91_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _277_MUL_8_OUTPUT_0_NB_OUTPUTS 256
#define _277_MUL_8_OUTPUT_0_OUT_HEIGHT 20
#define _277_MUL_8_OUTPUT_0_OUT_WIDTH 20
#define _277_MUL_8_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _277_MUL_8_OUTPUT_0_MEM_SIZE 1024
#define _277_MUL_8_OUTPUT_0_MEM_OFFSET 2150400
#define _277_MUL_8_OUTPUT_0_MEM_STRIDE 1024
#define _277_MUL_8_OUTPUT_0_MEM_LENGTH 20
#define _277_MUL_8_OUTPUT_0_MEM_CONT_OFFSET 0
#define _277_MUL_8_OUTPUT_0_MEM_CONT_SIZE 0
#define _277_MUL_8_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _277_MUL_8_OUTPUT_0_MEM_WRAP_SIZE 0


#define _277_MUL_8_NB_MAT 1
#define _277_MUL_8_INPUT1_CONT_SIZE 102400
#define _277_MUL_8_INPUT2_CONT_SIZE 102400
#define _277_MUL_8_OUTPUT_CONT_SIZE 102400

constexpr size_t _277_MUL_8_OFFSET_IN1[] =  { 0 };
constexpr size_t _277_MUL_8_OFFSET_IN2[] = { 0 };

#define _277_MUL_8_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _277_MUL_8_ELEM_OP export_cpp::ElemWise_T::Mul

#define _277_MUL_8_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _277_MUL_8_RESCALING = {};

#endif /* _277_MUL_8_LAYER_H */
