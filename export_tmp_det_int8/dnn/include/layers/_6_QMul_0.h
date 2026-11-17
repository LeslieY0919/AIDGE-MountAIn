#ifndef _6_QMUL_0_LAYER_H
#define _6_QMUL_0_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _1_QUANTIZER_0_OUTPUT_0_IN_BATCH 1
#define _1_QUANTIZER_0_OUTPUT_0_NB_CHANNELS 16
#define _1_QUANTIZER_0_OUTPUT_0_IN_HEIGHT 320
#define _1_QUANTIZER_0_OUTPUT_0_IN_WIDTH 320

// OUTPUT CONF
#define _6_QMUL_0_OUTPUT_0_NB_OUTPUTS 16
#define _6_QMUL_0_OUTPUT_0_OUT_HEIGHT 320
#define _6_QMUL_0_OUTPUT_0_OUT_WIDTH 320
#define _6_QMUL_0_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _6_QMUL_0_OUTPUT_0_MEM_SIZE 16
#define _6_QMUL_0_OUTPUT_0_MEM_OFFSET 6553600
#define _6_QMUL_0_OUTPUT_0_MEM_STRIDE 16
#define _6_QMUL_0_OUTPUT_0_MEM_LENGTH 320
#define _6_QMUL_0_OUTPUT_0_MEM_CONT_OFFSET 0
#define _6_QMUL_0_OUTPUT_0_MEM_CONT_SIZE 0
#define _6_QMUL_0_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _6_QMUL_0_OUTPUT_0_MEM_WRAP_SIZE 0


#define _6_QMUL_0_NB_MAT 1
#define _6_QMUL_0_INPUT1_CONT_SIZE 1638400
#define _6_QMUL_0_INPUT2_CONT_SIZE 1638400
#define _6_QMUL_0_OUTPUT_CONT_SIZE 1638400

constexpr size_t _6_QMUL_0_OFFSET_IN1[] =  { 0 };
constexpr size_t _6_QMUL_0_OFFSET_IN2[] = { 0 };

#define _6_QMUL_0_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _6_QMUL_0_ELEM_OP export_cpp::ElemWise_T::Mul

#define _6_QMUL_0_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _6_QMUL_0_RESCALING = {};

#endif /* _6_QMUL_0_LAYER_H */
