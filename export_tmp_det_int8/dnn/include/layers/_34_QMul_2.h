#ifndef _34_QMUL_2_LAYER_H
#define _34_QMUL_2_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _29_QUANTIZER_10_OUTPUT_0_IN_BATCH 1
#define _29_QUANTIZER_10_OUTPUT_0_NB_CHANNELS 16
#define _29_QUANTIZER_10_OUTPUT_0_IN_HEIGHT 160
#define _29_QUANTIZER_10_OUTPUT_0_IN_WIDTH 160

// OUTPUT CONF
#define _34_QMUL_2_OUTPUT_0_NB_OUTPUTS 16
#define _34_QMUL_2_OUTPUT_0_OUT_HEIGHT 160
#define _34_QMUL_2_OUTPUT_0_OUT_WIDTH 160
#define _34_QMUL_2_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _34_QMUL_2_OUTPUT_0_MEM_SIZE 16
#define _34_QMUL_2_OUTPUT_0_MEM_OFFSET 409600
#define _34_QMUL_2_OUTPUT_0_MEM_STRIDE 16
#define _34_QMUL_2_OUTPUT_0_MEM_LENGTH 160
#define _34_QMUL_2_OUTPUT_0_MEM_CONT_OFFSET 0
#define _34_QMUL_2_OUTPUT_0_MEM_CONT_SIZE 0
#define _34_QMUL_2_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _34_QMUL_2_OUTPUT_0_MEM_WRAP_SIZE 0


#define _34_QMUL_2_NB_MAT 1
#define _34_QMUL_2_INPUT1_CONT_SIZE 409600
#define _34_QMUL_2_INPUT2_CONT_SIZE 409600
#define _34_QMUL_2_OUTPUT_CONT_SIZE 409600

constexpr size_t _34_QMUL_2_OFFSET_IN1[] =  { 0 };
constexpr size_t _34_QMUL_2_OFFSET_IN2[] = { 0 };

#define _34_QMUL_2_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _34_QMUL_2_ELEM_OP export_cpp::ElemWise_T::Mul

#define _34_QMUL_2_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _34_QMUL_2_RESCALING = {};

#endif /* _34_QMUL_2_LAYER_H */
