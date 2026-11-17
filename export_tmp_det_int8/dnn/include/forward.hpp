#ifndef DNN_HPP
#define DNN_HPP

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


void model_forward (const int8_t* _0_PadConv_0_input_0,
    float** _617_Dequantizer_77_output_0, 
    float** _630_Dequantizer_79_output_0, 
    int64_t** _234_PadMaxPool_0_output_1, 
    int64_t** _235_PadMaxPool_1_output_1, 
    int64_t** _236_PadMaxPool_2_output_1);


#ifdef __cplusplus
}
#endif

#endif /* DNN_HPP */
