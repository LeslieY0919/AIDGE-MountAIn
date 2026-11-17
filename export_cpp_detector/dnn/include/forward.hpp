#ifndef DNN_HPP
#define DNN_HPP

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


void model_forward (const float* _0_PadConv_0_input_0,
    float** _260_Concat_14_output_0, 
    float** _267_Concat_15_output_0, 
    int64_t** _94_PadMaxPool_0_output_1, 
    int64_t** _95_PadMaxPool_1_output_1, 
    int64_t** _96_PadMaxPool_2_output_1);


#ifdef __cplusplus
}
#endif

#endif /* DNN_HPP */
