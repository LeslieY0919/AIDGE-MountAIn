#ifndef _85_QMUL_7_LAYER_H
#define _85_QMUL_7_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _81_QUANTIZER_26_OUTPUT_0_IN_BATCH 1
#define _81_QUANTIZER_26_OUTPUT_0_NB_CHANNELS 32
#define _81_QUANTIZER_26_OUTPUT_0_IN_HEIGHT 80
#define _81_QUANTIZER_26_OUTPUT_0_IN_WIDTH 80

// OUTPUT CONF
#define _85_QMUL_7_OUTPUT_0_NB_OUTPUTS 32
#define _85_QMUL_7_OUTPUT_0_OUT_HEIGHT 80
#define _85_QMUL_7_OUTPUT_0_OUT_WIDTH 80
#define _85_QMUL_7_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _85_QMUL_7_OUTPUT_0_MEM_SIZE 32
#define _85_QMUL_7_OUTPUT_0_MEM_OFFSET 1638400
#define _85_QMUL_7_OUTPUT_0_MEM_STRIDE 32
#define _85_QMUL_7_OUTPUT_0_MEM_LENGTH 80
#define _85_QMUL_7_OUTPUT_0_MEM_CONT_OFFSET 0
#define _85_QMUL_7_OUTPUT_0_MEM_CONT_SIZE 0
#define _85_QMUL_7_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _85_QMUL_7_OUTPUT_0_MEM_WRAP_SIZE 0


#define _85_QMUL_7_NB_MAT 1
#define _85_QMUL_7_INPUT1_CONT_SIZE 204800
#define _85_QMUL_7_INPUT2_CONT_SIZE 204800
#define _85_QMUL_7_OUTPUT_CONT_SIZE 204800

constexpr size_t _85_QMUL_7_OFFSET_IN1[] =  { 0 };
constexpr size_t _85_QMUL_7_OFFSET_IN2[] = { 0 };

#define _85_QMUL_7_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _85_QMUL_7_ELEM_OP export_cpp::ElemWise_T::Mul

#define _85_QMUL_7_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _85_QMUL_7_RESCALING = {};

#endif /* _85_QMUL_7_LAYER_H */
