#ifndef _171_MUL_44_LAYER_H
#define _171_MUL_44_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _166_PADCONVDW_3_OUTPUT_0_IN_BATCH 1
#define _166_PADCONVDW_3_OUTPUT_0_NB_CHANNELS 64
#define _166_PADCONVDW_3_OUTPUT_0_IN_HEIGHT 80
#define _166_PADCONVDW_3_OUTPUT_0_IN_WIDTH 80

// OUTPUT CONF
#define _171_MUL_44_OUTPUT_0_NB_OUTPUTS 64
#define _171_MUL_44_OUTPUT_0_OUT_HEIGHT 80
#define _171_MUL_44_OUTPUT_0_OUT_WIDTH 80
#define _171_MUL_44_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _171_MUL_44_OUTPUT_0_MEM_SIZE 256
#define _171_MUL_44_OUTPUT_0_MEM_OFFSET 5324800
#define _171_MUL_44_OUTPUT_0_MEM_STRIDE 256
#define _171_MUL_44_OUTPUT_0_MEM_LENGTH 80
#define _171_MUL_44_OUTPUT_0_MEM_CONT_OFFSET 0
#define _171_MUL_44_OUTPUT_0_MEM_CONT_SIZE 0
#define _171_MUL_44_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _171_MUL_44_OUTPUT_0_MEM_WRAP_SIZE 0


#define _171_MUL_44_NB_MAT 1
#define _171_MUL_44_INPUT1_CONT_SIZE 409600
#define _171_MUL_44_INPUT2_CONT_SIZE 409600
#define _171_MUL_44_OUTPUT_CONT_SIZE 409600

constexpr size_t _171_MUL_44_OFFSET_IN1[] =  { 0 };
constexpr size_t _171_MUL_44_OFFSET_IN2[] = { 0 };

#define _171_MUL_44_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _171_MUL_44_ELEM_OP export_cpp::ElemWise_T::Mul

#define _171_MUL_44_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _171_MUL_44_RESCALING = {};

#endif /* _171_MUL_44_LAYER_H */
