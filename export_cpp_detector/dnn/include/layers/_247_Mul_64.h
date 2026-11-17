#ifndef _247_MUL_64_LAYER_H
#define _247_MUL_64_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _243_PADCONVDW_10_OUTPUT_0_IN_BATCH 1
#define _243_PADCONVDW_10_OUTPUT_0_NB_CHANNELS 256
#define _243_PADCONVDW_10_OUTPUT_0_IN_HEIGHT 20
#define _243_PADCONVDW_10_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _247_MUL_64_OUTPUT_0_NB_OUTPUTS 256
#define _247_MUL_64_OUTPUT_0_OUT_HEIGHT 20
#define _247_MUL_64_OUTPUT_0_OUT_WIDTH 20
#define _247_MUL_64_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _247_MUL_64_OUTPUT_0_MEM_SIZE 1024
#define _247_MUL_64_OUTPUT_0_MEM_OFFSET 2969600
#define _247_MUL_64_OUTPUT_0_MEM_STRIDE 1024
#define _247_MUL_64_OUTPUT_0_MEM_LENGTH 20
#define _247_MUL_64_OUTPUT_0_MEM_CONT_OFFSET 0
#define _247_MUL_64_OUTPUT_0_MEM_CONT_SIZE 0
#define _247_MUL_64_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _247_MUL_64_OUTPUT_0_MEM_WRAP_SIZE 0


#define _247_MUL_64_NB_MAT 1
#define _247_MUL_64_INPUT1_CONT_SIZE 102400
#define _247_MUL_64_INPUT2_CONT_SIZE 102400
#define _247_MUL_64_OUTPUT_CONT_SIZE 102400

constexpr size_t _247_MUL_64_OFFSET_IN1[] =  { 0 };
constexpr size_t _247_MUL_64_OFFSET_IN2[] = { 0 };

#define _247_MUL_64_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _247_MUL_64_ELEM_OP export_cpp::ElemWise_T::Mul

#define _247_MUL_64_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _247_MUL_64_RESCALING = {};

#endif /* _247_MUL_64_LAYER_H */
