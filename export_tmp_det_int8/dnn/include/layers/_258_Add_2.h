#ifndef _258_ADD_2_LAYER_H
#define _258_ADD_2_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _257_QUANTIZER_86_OUTPUT_0_IN_BATCH 1
#define _257_QUANTIZER_86_OUTPUT_0_NB_CHANNELS 128
#define _257_QUANTIZER_86_OUTPUT_0_IN_HEIGHT 20
#define _257_QUANTIZER_86_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _258_ADD_2_OUTPUT_0_NB_OUTPUTS 128
#define _258_ADD_2_OUTPUT_0_OUT_HEIGHT 20
#define _258_ADD_2_OUTPUT_0_OUT_WIDTH 20
#define _258_ADD_2_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _258_ADD_2_OUTPUT_0_MEM_SIZE 512
#define _258_ADD_2_OUTPUT_0_MEM_OFFSET 2048000
#define _258_ADD_2_OUTPUT_0_MEM_STRIDE 512
#define _258_ADD_2_OUTPUT_0_MEM_LENGTH 20
#define _258_ADD_2_OUTPUT_0_MEM_CONT_OFFSET 0
#define _258_ADD_2_OUTPUT_0_MEM_CONT_SIZE 0
#define _258_ADD_2_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _258_ADD_2_OUTPUT_0_MEM_WRAP_SIZE 0


#define _258_ADD_2_NB_MAT 1
#define _258_ADD_2_INPUT1_CONT_SIZE 51200
#define _258_ADD_2_INPUT2_CONT_SIZE 51200
#define _258_ADD_2_OUTPUT_CONT_SIZE 51200

constexpr size_t _258_ADD_2_OFFSET_IN1[] =  { 0 };
constexpr size_t _258_ADD_2_OFFSET_IN2[] = { 0 };

#define _258_ADD_2_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _258_ADD_2_ELEM_OP export_cpp::ElemWise_T::Add

#define _258_ADD_2_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _258_ADD_2_RESCALING = {};

#endif /* _258_ADD_2_LAYER_H */
