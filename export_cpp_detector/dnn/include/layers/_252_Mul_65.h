#ifndef _252_MUL_65_LAYER_H
#define _252_MUL_65_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _249_PADCONV_27_OUTPUT_0_IN_BATCH 1
#define _249_PADCONV_27_OUTPUT_0_NB_CHANNELS 64
#define _249_PADCONV_27_OUTPUT_0_IN_HEIGHT 20
#define _249_PADCONV_27_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _252_MUL_65_OUTPUT_0_NB_OUTPUTS 64
#define _252_MUL_65_OUTPUT_0_OUT_HEIGHT 20
#define _252_MUL_65_OUTPUT_0_OUT_WIDTH 20
#define _252_MUL_65_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _252_MUL_65_OUTPUT_0_MEM_SIZE 256
#define _252_MUL_65_OUTPUT_0_MEM_OFFSET 1843200
#define _252_MUL_65_OUTPUT_0_MEM_STRIDE 256
#define _252_MUL_65_OUTPUT_0_MEM_LENGTH 20
#define _252_MUL_65_OUTPUT_0_MEM_CONT_OFFSET 0
#define _252_MUL_65_OUTPUT_0_MEM_CONT_SIZE 0
#define _252_MUL_65_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _252_MUL_65_OUTPUT_0_MEM_WRAP_SIZE 0


#define _252_MUL_65_NB_MAT 1
#define _252_MUL_65_INPUT1_CONT_SIZE 25600
#define _252_MUL_65_INPUT2_CONT_SIZE 25600
#define _252_MUL_65_OUTPUT_CONT_SIZE 25600

constexpr size_t _252_MUL_65_OFFSET_IN1[] =  { 0 };
constexpr size_t _252_MUL_65_OFFSET_IN2[] = { 0 };

#define _252_MUL_65_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _252_MUL_65_ELEM_OP export_cpp::ElemWise_T::Mul

#define _252_MUL_65_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _252_MUL_65_RESCALING = {};

#endif /* _252_MUL_65_LAYER_H */
