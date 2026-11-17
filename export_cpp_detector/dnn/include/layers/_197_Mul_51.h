#ifndef _197_MUL_51_LAYER_H
#define _197_MUL_51_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _191_PADCONVDW_4_OUTPUT_0_IN_BATCH 1
#define _191_PADCONVDW_4_OUTPUT_0_NB_CHANNELS 128
#define _191_PADCONVDW_4_OUTPUT_0_IN_HEIGHT 40
#define _191_PADCONVDW_4_OUTPUT_0_IN_WIDTH 40

// OUTPUT CONF
#define _197_MUL_51_OUTPUT_0_NB_OUTPUTS 128
#define _197_MUL_51_OUTPUT_0_OUT_HEIGHT 40
#define _197_MUL_51_OUTPUT_0_OUT_WIDTH 40
#define _197_MUL_51_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _197_MUL_51_OUTPUT_0_MEM_SIZE 512
#define _197_MUL_51_OUTPUT_0_MEM_OFFSET 3686400
#define _197_MUL_51_OUTPUT_0_MEM_STRIDE 512
#define _197_MUL_51_OUTPUT_0_MEM_LENGTH 40
#define _197_MUL_51_OUTPUT_0_MEM_CONT_OFFSET 0
#define _197_MUL_51_OUTPUT_0_MEM_CONT_SIZE 0
#define _197_MUL_51_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _197_MUL_51_OUTPUT_0_MEM_WRAP_SIZE 0


#define _197_MUL_51_NB_MAT 1
#define _197_MUL_51_INPUT1_CONT_SIZE 204800
#define _197_MUL_51_INPUT2_CONT_SIZE 204800
#define _197_MUL_51_OUTPUT_CONT_SIZE 204800

constexpr size_t _197_MUL_51_OFFSET_IN1[] =  { 0 };
constexpr size_t _197_MUL_51_OFFSET_IN2[] = { 0 };

#define _197_MUL_51_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _197_MUL_51_ELEM_OP export_cpp::ElemWise_T::Mul

#define _197_MUL_51_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _197_MUL_51_RESCALING = {};

#endif /* _197_MUL_51_LAYER_H */
