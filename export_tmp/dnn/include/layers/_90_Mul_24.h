#ifndef _90_MUL_24_LAYER_H
#define _90_MUL_24_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _88_CONV2D_9_OUTPUT_0_IN_BATCH 1
#define _88_CONV2D_9_OUTPUT_0_NB_CHANNELS 256
#define _88_CONV2D_9_OUTPUT_0_IN_HEIGHT 10
#define _88_CONV2D_9_OUTPUT_0_IN_WIDTH 10

// OUTPUT CONF
#define _90_MUL_24_OUTPUT_0_NB_OUTPUTS 256
#define _90_MUL_24_OUTPUT_0_OUT_HEIGHT 10
#define _90_MUL_24_OUTPUT_0_OUT_WIDTH 10
#define _90_MUL_24_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _90_MUL_24_OUTPUT_0_MEM_SIZE 1024
#define _90_MUL_24_OUTPUT_0_MEM_OFFSET 204800
#define _90_MUL_24_OUTPUT_0_MEM_STRIDE 1024
#define _90_MUL_24_OUTPUT_0_MEM_LENGTH 10
#define _90_MUL_24_OUTPUT_0_MEM_CONT_OFFSET 0
#define _90_MUL_24_OUTPUT_0_MEM_CONT_SIZE 0
#define _90_MUL_24_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _90_MUL_24_OUTPUT_0_MEM_WRAP_SIZE 0


#define _90_MUL_24_NB_MAT 1
#define _90_MUL_24_INPUT1_CONT_SIZE 25600
#define _90_MUL_24_INPUT2_CONT_SIZE 25600
#define _90_MUL_24_OUTPUT_CONT_SIZE 25600

constexpr size_t _90_MUL_24_OFFSET_IN1[] =  { 0 };
constexpr size_t _90_MUL_24_OFFSET_IN2[] = { 0 };

#define _90_MUL_24_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _90_MUL_24_ELEM_OP export_cpp::ElemWise_T::Mul

#define _90_MUL_24_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _90_MUL_24_RESCALING = {};

#endif /* _90_MUL_24_LAYER_H */
