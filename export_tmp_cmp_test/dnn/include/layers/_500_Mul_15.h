#ifndef _500_MUL_15_LAYER_H
#define _500_MUL_15_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _490_QUANTIZER_160_OUTPUT_0_IN_BATCH 1
#define _490_QUANTIZER_160_OUTPUT_0_NB_CHANNELS 256
#define _490_QUANTIZER_160_OUTPUT_0_IN_HEIGHT 20
#define _490_QUANTIZER_160_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _500_MUL_15_OUTPUT_0_NB_OUTPUTS 256
#define _500_MUL_15_OUTPUT_0_OUT_HEIGHT 20
#define _500_MUL_15_OUTPUT_0_OUT_WIDTH 20
#define _500_MUL_15_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _500_MUL_15_OUTPUT_0_MEM_SIZE 1024
#define _500_MUL_15_OUTPUT_0_MEM_OFFSET 2048000
#define _500_MUL_15_OUTPUT_0_MEM_STRIDE 1024
#define _500_MUL_15_OUTPUT_0_MEM_LENGTH 20
#define _500_MUL_15_OUTPUT_0_MEM_CONT_OFFSET 0
#define _500_MUL_15_OUTPUT_0_MEM_CONT_SIZE 0
#define _500_MUL_15_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _500_MUL_15_OUTPUT_0_MEM_WRAP_SIZE 0


#define _500_MUL_15_NB_MAT 1
#define _500_MUL_15_INPUT1_CONT_SIZE 102400
#define _500_MUL_15_INPUT2_CONT_SIZE 102400
#define _500_MUL_15_OUTPUT_CONT_SIZE 102400

constexpr size_t _500_MUL_15_OFFSET_IN1[] =  { 0 };
constexpr size_t _500_MUL_15_OFFSET_IN2[] = { 0 };

#define _500_MUL_15_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _500_MUL_15_ELEM_OP export_cpp::ElemWise_T::Mul

#define _500_MUL_15_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _500_MUL_15_RESCALING = {};

#endif /* _500_MUL_15_LAYER_H */
