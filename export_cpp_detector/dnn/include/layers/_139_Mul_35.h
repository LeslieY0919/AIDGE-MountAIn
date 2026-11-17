#ifndef _139_MUL_35_LAYER_H
#define _139_MUL_35_LAYER_H

#include <cstddef>
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _137_PADCONV_17_OUTPUT_0_IN_BATCH 1
#define _137_PADCONV_17_OUTPUT_0_NB_CHANNELS 32
#define _137_PADCONV_17_OUTPUT_0_IN_HEIGHT 80
#define _137_PADCONV_17_OUTPUT_0_IN_WIDTH 80

// OUTPUT CONF
#define _139_MUL_35_OUTPUT_0_NB_OUTPUTS 32
#define _139_MUL_35_OUTPUT_0_OUT_HEIGHT 80
#define _139_MUL_35_OUTPUT_0_OUT_WIDTH 80
#define _139_MUL_35_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF




#define _139_MUL_35_OUTPUT_0_MEM_SIZE 128
#define _139_MUL_35_OUTPUT_0_MEM_OFFSET 6963200
#define _139_MUL_35_OUTPUT_0_MEM_STRIDE 128
#define _139_MUL_35_OUTPUT_0_MEM_LENGTH 80
#define _139_MUL_35_OUTPUT_0_MEM_CONT_OFFSET 0
#define _139_MUL_35_OUTPUT_0_MEM_CONT_SIZE 0
#define _139_MUL_35_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _139_MUL_35_OUTPUT_0_MEM_WRAP_SIZE 0


#define _139_MUL_35_NB_MAT 1
#define _139_MUL_35_INPUT1_CONT_SIZE 204800
#define _139_MUL_35_INPUT2_CONT_SIZE 204800
#define _139_MUL_35_OUTPUT_CONT_SIZE 204800

constexpr size_t _139_MUL_35_OFFSET_IN1[] =  { 0 };
constexpr size_t _139_MUL_35_OFFSET_IN2[] = { 0 };

#define _139_MUL_35_ACTIVATION export_cpp::ActivationFunction_T::Linear
#define _139_MUL_35_ELEM_OP export_cpp::ElemWise_T::Mul

#define _139_MUL_35_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::NHWC

static const export_cpp::NoScaling _139_MUL_35_RESCALING = {};

#endif /* _139_MUL_35_LAYER_H */
