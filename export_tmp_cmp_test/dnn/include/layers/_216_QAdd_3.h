#ifndef _216_QADD_3_LAYER_H
#define _216_QADD_3_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _200_QUANTIZER_68_OUTPUT_0_IN_BATCH 1
#define _200_QUANTIZER_68_OUTPUT_0_NB_CHANNELS 128
#define _200_QUANTIZER_68_OUTPUT_0_IN_HEIGHT 20
#define _200_QUANTIZER_68_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _216_QADD_3_OUTPUT_0_NB_OUTPUTS 128
#define _216_QADD_3_OUTPUT_0_OUT_HEIGHT 20
#define _216_QADD_3_OUTPUT_0_OUT_WIDTH 20
#define _216_QADD_3_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _216_QADD_3_OUTPUT_0_MEM_SIZE 128
#define _216_QADD_3_OUTPUT_0_MEM_OFFSET 921600
#define _216_QADD_3_OUTPUT_0_MEM_STRIDE 128
#define _216_QADD_3_OUTPUT_0_MEM_LENGTH 20
#define _216_QADD_3_OUTPUT_0_MEM_CONT_OFFSET 0
#define _216_QADD_3_OUTPUT_0_MEM_CONT_SIZE 0
#define _216_QADD_3_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _216_QADD_3_OUTPUT_0_MEM_WRAP_SIZE 0


#define _216_QADD_3_NB_MAT 1
#define _216_QADD_3_INPUT1_CONT_SIZE 51200
#define _216_QADD_3_INPUT2_CONT_SIZE 51200
#define _216_QADD_3_OUTPUT_CONT_SIZE 51200

constexpr size_t _216_QADD_3_OFFSET_IN1[] =  { 0 };
constexpr size_t _216_QADD_3_OFFSET_IN2[] = { 0 };

#define _216_QADD_3_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _216_QADD_3_ELEM_OP export_cpp::ElemWise_T::Add

#define _216_QADD_3_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _216_QADD_3_RESCALING = {};

#endif /* _216_QADD_3_LAYER_H */
