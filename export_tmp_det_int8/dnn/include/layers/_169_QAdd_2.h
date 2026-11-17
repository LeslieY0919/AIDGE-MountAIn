#ifndef _169_QADD_2_LAYER_H
#define _169_QADD_2_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _152_QUANTIZER_51_OUTPUT_0_IN_BATCH 1
#define _152_QUANTIZER_51_OUTPUT_0_NB_CHANNELS 64
#define _152_QUANTIZER_51_OUTPUT_0_IN_HEIGHT 40
#define _152_QUANTIZER_51_OUTPUT_0_IN_WIDTH 40

// OUTPUT CONF
#define _169_QADD_2_OUTPUT_0_NB_OUTPUTS 64
#define _169_QADD_2_OUTPUT_0_OUT_HEIGHT 40
#define _169_QADD_2_OUTPUT_0_OUT_WIDTH 40
#define _169_QADD_2_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _169_QADD_2_OUTPUT_0_MEM_SIZE 64
#define _169_QADD_2_OUTPUT_0_MEM_OFFSET 512000
#define _169_QADD_2_OUTPUT_0_MEM_STRIDE 64
#define _169_QADD_2_OUTPUT_0_MEM_LENGTH 40
#define _169_QADD_2_OUTPUT_0_MEM_CONT_OFFSET 0
#define _169_QADD_2_OUTPUT_0_MEM_CONT_SIZE 0
#define _169_QADD_2_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _169_QADD_2_OUTPUT_0_MEM_WRAP_SIZE 0


#define _169_QADD_2_NB_MAT 1
#define _169_QADD_2_INPUT1_CONT_SIZE 102400
#define _169_QADD_2_INPUT2_CONT_SIZE 102400
#define _169_QADD_2_OUTPUT_CONT_SIZE 102400

constexpr size_t _169_QADD_2_OFFSET_IN1[] =  { 0 };
constexpr size_t _169_QADD_2_OFFSET_IN2[] = { 0 };

#define _169_QADD_2_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _169_QADD_2_ELEM_OP export_cpp::ElemWise_T::Add

#define _169_QADD_2_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _169_QADD_2_RESCALING = {};

#endif /* _169_QADD_2_LAYER_H */
