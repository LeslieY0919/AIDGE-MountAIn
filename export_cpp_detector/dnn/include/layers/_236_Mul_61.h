#ifndef _236_MUL_61_LAYER_H
#define _236_MUL_61_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _234_PADCONVDW_9_OUTPUT_0_IN_BATCH 1
#define _234_PADCONVDW_9_OUTPUT_0_NB_CHANNELS 128
#define _234_PADCONVDW_9_OUTPUT_0_IN_HEIGHT 20
#define _234_PADCONVDW_9_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _236_MUL_61_OUTPUT_0_NB_OUTPUTS 128
#define _236_MUL_61_OUTPUT_0_OUT_HEIGHT 20
#define _236_MUL_61_OUTPUT_0_OUT_WIDTH 20
#define _236_MUL_61_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _236_MUL_61_OUTPUT_0_MEM_SIZE 512
#define _236_MUL_61_OUTPUT_0_MEM_OFFSET 2252800
#define _236_MUL_61_OUTPUT_0_MEM_STRIDE 512
#define _236_MUL_61_OUTPUT_0_MEM_LENGTH 20
#define _236_MUL_61_OUTPUT_0_MEM_CONT_OFFSET 0
#define _236_MUL_61_OUTPUT_0_MEM_CONT_SIZE 0
#define _236_MUL_61_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _236_MUL_61_OUTPUT_0_MEM_WRAP_SIZE 0


#define _236_MUL_61_NB_MAT 1
#define _236_MUL_61_INPUT1_CONT_SIZE 51200
#define _236_MUL_61_INPUT2_CONT_SIZE 51200
#define _236_MUL_61_OUTPUT_CONT_SIZE 51200

constexpr size_t _236_MUL_61_OFFSET_IN1[] =  { 0 };
constexpr size_t _236_MUL_61_OFFSET_IN2[] = { 0 };

#define _236_MUL_61_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _236_MUL_61_ELEM_OP export_cpp::ElemWise_T::Mul

#define _236_MUL_61_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _236_MUL_61_RESCALING = {};

#endif /* _236_MUL_61_LAYER_H */
