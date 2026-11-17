#ifndef _251_MUL_7_LAYER_H
#define _251_MUL_7_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _247_QUANTIZER_82_OUTPUT_0_IN_BATCH 1
#define _247_QUANTIZER_82_OUTPUT_0_NB_CHANNELS 256
#define _247_QUANTIZER_82_OUTPUT_0_IN_HEIGHT 20
#define _247_QUANTIZER_82_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _251_MUL_7_OUTPUT_0_NB_OUTPUTS 256
#define _251_MUL_7_OUTPUT_0_OUT_HEIGHT 20
#define _251_MUL_7_OUTPUT_0_OUT_WIDTH 20
#define _251_MUL_7_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _251_MUL_7_OUTPUT_0_MEM_SIZE 1024
#define _251_MUL_7_OUTPUT_0_MEM_OFFSET 2048000
#define _251_MUL_7_OUTPUT_0_MEM_STRIDE 1024
#define _251_MUL_7_OUTPUT_0_MEM_LENGTH 20
#define _251_MUL_7_OUTPUT_0_MEM_CONT_OFFSET 0
#define _251_MUL_7_OUTPUT_0_MEM_CONT_SIZE 0
#define _251_MUL_7_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _251_MUL_7_OUTPUT_0_MEM_WRAP_SIZE 0


#define _251_MUL_7_NB_MAT 1
#define _251_MUL_7_INPUT1_CONT_SIZE 102400
#define _251_MUL_7_INPUT2_CONT_SIZE 102400
#define _251_MUL_7_OUTPUT_CONT_SIZE 102400

constexpr size_t _251_MUL_7_OFFSET_IN1[] =  { 0 };
constexpr size_t _251_MUL_7_OFFSET_IN2[] = { 0 };

#define _251_MUL_7_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _251_MUL_7_ELEM_OP export_cpp::ElemWise_T::Mul

#define _251_MUL_7_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _251_MUL_7_RESCALING = {};

#endif /* _251_MUL_7_LAYER_H */
