#include <cstdio>
#include <cstdint>
#include "forward.hpp"
#include "data/_0_PadConv_0_input_0.h"

int main() {
    float* _617_Dequantizer_77_output_0 = nullptr;
    float* _630_Dequantizer_79_output_0 = nullptr;
    int64_t* _234_PadMaxPool_0_output_1 = nullptr;
    int64_t* _235_PadMaxPool_1_output_1 = nullptr;
    int64_t* _236_PadMaxPool_2_output_1 = nullptr;
    model_forward(_0_PadConv_0_input_0, &_617_Dequantizer_77_output_0, &_630_Dequantizer_79_output_0, &_234_PadMaxPool_0_output_1, &_235_PadMaxPool_1_output_1, &_236_PadMaxPool_2_output_1);
    FILE* fb = std::fopen("boxes_out_f32.bin", "wb");
    std::fwrite(_617_Dequantizer_77_output_0, sizeof(float), 1 * 64 * 8400, fb);
    std::fclose(fb);
    FILE* fs = std::fopen("scores_out_f32.bin", "wb");
    std::fwrite(_630_Dequantizer_79_output_0, sizeof(float), 1 * 3 * 8400, fs);
    std::fclose(fs);
    return 0;
}
