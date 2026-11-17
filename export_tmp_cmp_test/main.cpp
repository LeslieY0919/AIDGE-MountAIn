#include <cstdio>
#include "forward.hpp"
#include "data/_0_PadConv_0_input_0.h"

int main()
{
    // Initialize the output arrays
    float* _617_Dequantizer_77_output_0 = nullptr;
    
    float* _630_Dequantizer_79_output_0 = nullptr;
    
    int64_t* _234_PadMaxPool_0_output_1 = nullptr;
    
    int64_t* _235_PadMaxPool_1_output_1 = nullptr;
    
    int64_t* _236_PadMaxPool_2_output_1 = nullptr;
    

    // Call the forward function
    model_forward(_0_PadConv_0_input_0, &_617_Dequantizer_77_output_0, &_630_Dequantizer_79_output_0, &_234_PadMaxPool_0_output_1, &_235_PadMaxPool_1_output_1, &_236_PadMaxPool_2_output_1);

    // Print the results

    printf("_617_Dequantizer_77_output_0:\n");
    for (int o = 0; o < 100; ++o) {
        printf("%f ", _617_Dequantizer_77_output_0[o]);
    }
    printf("Output 0 is too large and has been truncated.\n");
    printf("\n");

    printf("_630_Dequantizer_79_output_0:\n");
    for (int o = 0; o < 100; ++o) {
        printf("%f ", _630_Dequantizer_79_output_0[o]);
    }
    printf("Output 1 is too large and has been truncated.\n");
    printf("\n");

    printf("_234_PadMaxPool_0_output_1:\n");
    for (int o = 0; o < 100; ++o) {
        printf("%ld ", _234_PadMaxPool_0_output_1[o]);
    }
    printf("Output 2 is too large and has been truncated.\n");
    printf("\n");

    printf("_235_PadMaxPool_1_output_1:\n");
    for (int o = 0; o < 100; ++o) {
        printf("%ld ", _235_PadMaxPool_1_output_1[o]);
    }
    printf("Output 3 is too large and has been truncated.\n");
    printf("\n");

    printf("_236_PadMaxPool_2_output_1:\n");
    for (int o = 0; o < 100; ++o) {
        printf("%ld ", _236_PadMaxPool_2_output_1[o]);
    }
    printf("Output 4 is too large and has been truncated.\n");
    printf("\n");
    return 0;
}
