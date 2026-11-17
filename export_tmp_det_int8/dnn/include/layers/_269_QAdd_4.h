#ifndef _269_QADD_4_LAYER_H
#define _269_QADD_4_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _260_QUANTIZER_88_OUTPUT_0_IN_BATCH 1
#define _260_QUANTIZER_88_OUTPUT_0_NB_CHANNELS 128
#define _260_QUANTIZER_88_OUTPUT_0_IN_HEIGHT 20
#define _260_QUANTIZER_88_OUTPUT_0_IN_WIDTH 20

// OUTPUT CONF
#define _269_QADD_4_OUTPUT_0_NB_OUTPUTS 128
#define _269_QADD_4_OUTPUT_0_OUT_HEIGHT 20
#define _269_QADD_4_OUTPUT_0_OUT_WIDTH 20
#define _269_QADD_4_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _269_QADD_4_OUTPUT_0_MEM_SIZE 128
#define _269_QADD_4_OUTPUT_0_MEM_OFFSET 716800
#define _269_QADD_4_OUTPUT_0_MEM_STRIDE 128
#define _269_QADD_4_OUTPUT_0_MEM_LENGTH 20
#define _269_QADD_4_OUTPUT_0_MEM_CONT_OFFSET 0
#define _269_QADD_4_OUTPUT_0_MEM_CONT_SIZE 0
#define _269_QADD_4_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _269_QADD_4_OUTPUT_0_MEM_WRAP_SIZE 0


#define _269_QADD_4_NB_MAT 1
#define _269_QADD_4_INPUT1_CONT_SIZE 51200
#define _269_QADD_4_INPUT2_CONT_SIZE 51200
#define _269_QADD_4_OUTPUT_CONT_SIZE 51200

constexpr size_t _269_QADD_4_OFFSET_IN1[] =  { 0 };
constexpr size_t _269_QADD_4_OFFSET_IN2[] = { 0 };

#define _269_QADD_4_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _269_QADD_4_ELEM_OP export_cpp::ElemWise_T::Add

#define _269_QADD_4_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _269_QADD_4_RESCALING = {};

#endif /* _269_QADD_4_LAYER_H */
