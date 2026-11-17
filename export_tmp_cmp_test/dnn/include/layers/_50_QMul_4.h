#ifndef _50_QMUL_4_LAYER_H
#define _50_QMUL_4_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _46_QUANTIZER_14_OUTPUT_0_IN_BATCH 1
#define _46_QUANTIZER_14_OUTPUT_0_NB_CHANNELS 32
#define _46_QUANTIZER_14_OUTPUT_0_IN_HEIGHT 160
#define _46_QUANTIZER_14_OUTPUT_0_IN_WIDTH 160

// OUTPUT CONF
#define _50_QMUL_4_OUTPUT_0_NB_OUTPUTS 32
#define _50_QMUL_4_OUTPUT_0_OUT_HEIGHT 160
#define _50_QMUL_4_OUTPUT_0_OUT_WIDTH 160
#define _50_QMUL_4_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _50_QMUL_4_OUTPUT_0_MEM_SIZE 32
#define _50_QMUL_4_OUTPUT_0_MEM_OFFSET 819200
#define _50_QMUL_4_OUTPUT_0_MEM_STRIDE 32
#define _50_QMUL_4_OUTPUT_0_MEM_LENGTH 160
#define _50_QMUL_4_OUTPUT_0_MEM_CONT_OFFSET 0
#define _50_QMUL_4_OUTPUT_0_MEM_CONT_SIZE 0
#define _50_QMUL_4_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _50_QMUL_4_OUTPUT_0_MEM_WRAP_SIZE 0


#define _50_QMUL_4_NB_MAT 1
#define _50_QMUL_4_INPUT1_CONT_SIZE 819200
#define _50_QMUL_4_INPUT2_CONT_SIZE 819200
#define _50_QMUL_4_OUTPUT_CONT_SIZE 819200

constexpr size_t _50_QMUL_4_OFFSET_IN1[] =  { 0 };
constexpr size_t _50_QMUL_4_OFFSET_IN2[] = { 0 };

#define _50_QMUL_4_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _50_QMUL_4_ELEM_OP export_cpp::ElemWise_T::Mul

#define _50_QMUL_4_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _50_QMUL_4_RESCALING = {};

#endif /* _50_QMUL_4_LAYER_H */
