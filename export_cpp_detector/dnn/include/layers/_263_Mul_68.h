#ifndef _263_MUL_68_LAYER_H
#define _263_MUL_68_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _261_CONV2D_37_OUTPUT_0_IN_BATCH 1
#define _261_CONV2D_37_OUTPUT_0_NB_CHANNELS 64
#define _261_CONV2D_37_OUTPUT_0_IN_HEIGHT 20
#define _261_CONV2D_37_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _263_MUL_68_OUTPUT_0_NB_OUTPUTS 64
#define _263_MUL_68_OUTPUT_0_OUT_HEIGHT 20
#define _263_MUL_68_OUTPUT_0_OUT_WIDTH 20
#define _263_MUL_68_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _263_MUL_68_OUTPUT_0_MEM_SIZE 256
#define _263_MUL_68_OUTPUT_0_MEM_OFFSET 204800
#define _263_MUL_68_OUTPUT_0_MEM_STRIDE 256
#define _263_MUL_68_OUTPUT_0_MEM_LENGTH 20
#define _263_MUL_68_OUTPUT_0_MEM_CONT_OFFSET 0
#define _263_MUL_68_OUTPUT_0_MEM_CONT_SIZE 0
#define _263_MUL_68_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _263_MUL_68_OUTPUT_0_MEM_WRAP_SIZE 0


#define _263_MUL_68_NB_MAT 1
#define _263_MUL_68_INPUT1_CONT_SIZE 25600
#define _263_MUL_68_INPUT2_CONT_SIZE 25600
#define _263_MUL_68_OUTPUT_CONT_SIZE 25600

constexpr size_t _263_MUL_68_OFFSET_IN1[] =  { 0 };
constexpr size_t _263_MUL_68_OFFSET_IN2[] = { 0 };

#define _263_MUL_68_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _263_MUL_68_ELEM_OP export_cpp::ElemWise_T::Mul

#define _263_MUL_68_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _263_MUL_68_RESCALING = {};

#endif /* _263_MUL_68_LAYER_H */
