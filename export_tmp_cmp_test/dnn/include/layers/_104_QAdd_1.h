#ifndef _104_QADD_1_LAYER_H
#define _104_QADD_1_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _87_QUANTIZER_28_OUTPUT_0_IN_BATCH 1
#define _87_QUANTIZER_28_OUTPUT_0_NB_CHANNELS 32
#define _87_QUANTIZER_28_OUTPUT_0_IN_HEIGHT 80
#define _87_QUANTIZER_28_OUTPUT_0_IN_WIDTH 80

// OUTPUT CONF
#define _104_QADD_1_OUTPUT_0_NB_OUTPUTS 32
#define _104_QADD_1_OUTPUT_0_OUT_HEIGHT 80
#define _104_QADD_1_OUTPUT_0_OUT_WIDTH 80
#define _104_QADD_1_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _104_QADD_1_OUTPUT_0_MEM_SIZE 32
#define _104_QADD_1_OUTPUT_0_MEM_OFFSET 204800
#define _104_QADD_1_OUTPUT_0_MEM_STRIDE 32
#define _104_QADD_1_OUTPUT_0_MEM_LENGTH 80
#define _104_QADD_1_OUTPUT_0_MEM_CONT_OFFSET 0
#define _104_QADD_1_OUTPUT_0_MEM_CONT_SIZE 0
#define _104_QADD_1_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _104_QADD_1_OUTPUT_0_MEM_WRAP_SIZE 0


#define _104_QADD_1_NB_MAT 1
#define _104_QADD_1_INPUT1_CONT_SIZE 204800
#define _104_QADD_1_INPUT2_CONT_SIZE 204800
#define _104_QADD_1_OUTPUT_CONT_SIZE 204800

constexpr size_t _104_QADD_1_OFFSET_IN1[] =  { 0 };
constexpr size_t _104_QADD_1_OFFSET_IN2[] = { 0 };

#define _104_QADD_1_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _104_QADD_1_ELEM_OP export_cpp::ElemWise_T::Add

#define _104_QADD_1_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _104_QADD_1_RESCALING = {};

#endif /* _104_QADD_1_LAYER_H */
