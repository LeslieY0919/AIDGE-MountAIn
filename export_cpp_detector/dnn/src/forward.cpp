
#include <stdint.h>
#include <stddef.h>


#include "forward.hpp"

// Layer & memory configurations

#include "layers/_117_Conv2D_16_weights.h"
#include "layers/_14_Sigmoid_4.h"
#include "parameters/_199_PadConv_25_biases.hpp"
#include "kernels/cpp/split.hpp"
#include "parameters/_266_Reshape_5_prod_0.hpp"
#include "parameters/_54_PadConv_9_biases.hpp"
#include "layers/_144_Conv2D_19_weights.h"
#include "layers/_25_Sigmoid_7.h"
#include "layers/_147_PadConvDw_2_weights.h"
#include "parameters/_0_PadConv_0_weights.hpp"
#include "layers/_131_Resize_1_prod_0.h"
#include "layers/_180_Sigmoid_47.h"
#include "parameters/_228_PadConvDw_8_biases.hpp"
#include "layers/_120_Split_5.h"
#include "layers/_116_Concat_6.h"
#include "layers/_106_Conv2D_13_weights.h"
#include "layers/_146_Mul_37.h"
#include "layers/_199_PadConv_25_weights.h"
#include "layers/_91_Conv2D_10.h"
#include "layers/_136_Split_6_prod_0.h"
#include "layers/_133_Conv2D_18.h"
#include "layers/_106_Conv2D_13.h"
#include "parameters/_259_Reshape_4_prod_0.hpp"
#include "layers/_141_Sigmoid_36.h"
#include "layers/_158_Conv2D_20_weights.h"
#include "layers/_33_Mul_9.h"
#include "layers/_199_PadConv_25_biases.h"
#include "layers/_19_Sigmoid_5.h"
#include "layers/_104_Split_4.h"
#include "layers/_200_Conv2D_27.h"
#include "parameters/_167_Conv2D_22_biases.hpp"
#include "parameters/_3_PadConv_1_weights.hpp"
#include "layers/_3_PadConv_1_biases.h"
#include "layers/_206_Conv2D_28_weights.h"
#include "layers/_190_Conv2D_26.h"
#include "layers/_56_Mul_15.h"
#include "kernels/cpp/transpose.hpp"
#include "layers/_242_PadConv_26_biases.h"
#include "layers/_234_PadConvDw_9_weights.h"
#include "layers/_222_Conv2D_31_weights.h"
#include "layers/_149_PadConv_20.h"
#include "layers/_241_Mul_62.h"
#include "layers/_61_PadConv_11.h"
#include "layers/_133_Conv2D_18_biases.h"
#include "layers/_6_Conv2D_0_biases.h"
#include "parameters/_88_Conv2D_9_weights.hpp"
#include "parameters/_199_PadConv_25_weights.hpp"
#include "layers/_263_Mul_68.h"
#include "layers/_214_Mul_55.h"
#include "layers/_131_Resize_1.h"
#include "layers/_238_Concat_13.h"
#include "layers/_188_Mul_48.h"
#include "layers/_239_Conv2D_34_weights.h"
#include "layers/_63_Mul_17.h"
#include "layers/_213_Split_8_prod_0.h"
#include "parameters/_3_PadConv_1_biases.hpp"
#include "layers/_43_Conv2D_3_weights.h"
#include "kernels/cpp/elemwise.hpp"
#include "layers/_54_PadConv_9_biases.h"
#include "parameters/_166_PadConvDw_3_biases.hpp"
#include "layers/_99_Sigmoid_26.h"
#include "parameters/_6_Conv2D_0_biases.hpp"
#include "parameters/_168_Split_7_prod_0.hpp"
#include "parameters/_124_PadConv_16_weights.hpp"
#include "parameters/_76_Conv2D_8_weights.hpp"
#include "layers/_21_PadConv_4_biases.h"
#include "parameters/_69_Conv2D_6_biases.hpp"
#include "layers/_31_PadConv_6_biases.h"
#include "layers/_207_Sigmoid_54.h"
#include "layers/_121_PadConv_15.h"
#include "layers/_218_Sigmoid_56.h"
#include "layers/_58_Sigmoid_16.h"
#include "parameters/_121_PadConv_15_weights.hpp"
#include "layers/_145_Sigmoid_37.h"
#include "layers/_32_Sigmoid_9.h"
#include "layers/_113_Sigmoid_29.h"
#include "layers/_216_Conv2D_30.h"
#include "layers/_24_Conv2D_2.h"
#include "parameters/_198_PadConvDw_5_weights.hpp"
#include "parameters/_144_Conv2D_19_biases.hpp"
#include "parameters/_9_Split_0_prod_0.hpp"
#include "layers/_240_Sigmoid_62.h"
#include "layers/_244_Sigmoid_63.h"
#include "layers/_172_PadConv_22_weights.h"
#include "parameters/_80_PadConv_13_biases.hpp"
#include "parameters/_231_Conv2D_33_biases.hpp"
#include "layers/_169_Sigmoid_44.h"
#include "layers/_39_Sigmoid_11.h"
#include "layers/_121_PadConv_15_weights.h"
#include "parameters/_140_PadConv_18_biases.hpp"
#include "layers/_234_PadConvDw_9_biases.h"
#include "layers/_255_PadConvDw_11.h"
#include "layers/_76_Conv2D_8_weights.h"
#include "parameters/_186_Conv2D_25_biases.hpp"
#include "layers/_75_PadConvDw_1_weights.h"
#include "layers/_139_Mul_35.h"
#include "layers/_213_Split_8.h"
#include "layers/_21_PadConv_4.h"
#include "layers/_124_PadConv_16_biases.h"
#include "layers/_57_PadConv_10.h"
#include "parameters/_35_PadConv_7_biases.hpp"
#include "layers/_256_Sigmoid_67.h"
#include "layers/_18_Conv2D_1_weights.h"
#include "layers/_231_Conv2D_33_weights.h"
#include "layers/_179_PadConv_23_weights.h"
#include "parameters/_72_Conv2D_7_weights.hpp"
#include "layers/_38_PadConv_8_weights.h"
#include "layers/_55_Sigmoid_15.h"
#include "layers/_92_Sigmoid_25.h"
#include "layers/_147_PadConvDw_2_biases.h"
#include "layers/_95_PadMaxPool_1.h"
#include "layers/_64_PadConv_12_weights.h"
#include "parameters/_136_Split_6_prod_0.hpp"
#include "parameters/_38_PadConv_8_biases.hpp"
#include "layers/_61_PadConv_11_weights.h"
#include "layers/_65_Sigmoid_18.h"
#include "parameters/_121_PadConv_15_biases.hpp"
#include "layers/_167_Conv2D_22_weights.h"
#include "layers/_259_Reshape_4_prod_0.h"
#include "layers/_217_PadConvDw_7_weights.h"
#include "layers/_108_Mul_28.h"
#include "layers/_210_Transpose_2.h"
#include "layers/_109_Conv2D_14_biases.h"
#include "layers/_186_Conv2D_25.h"
#include "parameters/_243_PadConvDw_10_weights.hpp"
#include "parameters/_208_Conv2D_29_weights.hpp"
#include "layers/_112_Conv2D_15_weights.h"
#include "kernels/cpp/pooling.hpp"
#include "layers/_212_Sigmoid_55.h"
#include "layers/_258_Mul_67.h"
#include "layers/_1_Sigmoid_0.h"
#include "layers/_250_Sigmoid_65.h"
#include "layers/_125_Sigmoid_32.h"
#include "layers/_203_Sigmoid_53.h"
#include "layers/_228_PadConvDw_8.h"
#include "parameters/_98_Conv2D_11_weights.hpp"
#include "layers/_119_Mul_30.h"
#include "layers/_68_Concat_2.h"
#include "layers/_64_PadConv_12.h"
#include "parameters/_112_Conv2D_15_weights.hpp"
#include "parameters/_255_PadConvDw_11_weights.hpp"
#include "layers/_191_PadConvDw_4_weights.h"
#include "parameters/_83_PadConv_14_biases.hpp"
#include "layers/_209_PadConvDw_6_biases.h"
#include "parameters/_200_Conv2D_27_biases.hpp"
#include "parameters/_181_Conv2D_24_weights.hpp"
#include "layers/_42_Concat_1.h"
#include "layers/_3_PadConv_1.h"
#include "layers/_98_Conv2D_11.h"
#include "parameters/_264_Conv2D_38_biases.hpp"
#include "parameters/_117_Conv2D_16_biases.hpp"
#include "parameters/_91_Conv2D_10_biases.hpp"
#include "layers/_17_Concat_0.h"
#include "parameters/_18_Conv2D_1_biases.hpp"
#include "parameters/_161_Conv2D_21_biases.hpp"
#include "layers/_222_Conv2D_31_biases.h"
#include "parameters/_239_Conv2D_34_biases.hpp"
#include "parameters/_234_PadConvDw_9_weights.hpp"
#include "parameters/_46_Conv2D_4_biases.hpp"
#include "layers/_246_Mul_63.h"
#include "parameters/_31_PadConv_6_biases.hpp"
#include "layers/_80_PadConv_13.h"
#include "parameters/_213_Split_8_prod_0.hpp"
#include "layers/_208_Conv2D_29.h"
#include "layers/_152_Sigmoid_40.h"
#include "layers/_198_PadConvDw_5_weights.h"
#include "layers/_70_Sigmoid_19.h"
#include "parameters/_6_Conv2D_0_weights.hpp"
#include "layers/_140_PadConv_18.h"
#include "layers/_228_PadConvDw_8_biases.h"
#include "parameters/_50_Conv2D_5_biases.hpp"
#include "layers/_89_Sigmoid_24.h"
#include "layers/_30_Mul_8.h"
#include "parameters/_216_Conv2D_30_biases.hpp"
#include "layers/_10_PadConv_2_weights.h"
#include "layers/_88_Conv2D_9.h"
#include "layers/_69_Conv2D_6.h"
#include "layers/_124_PadConv_16_weights.h"
#include "layers/_57_PadConv_10_weights.h"
#include "layers/_78_Mul_21.h"
#include "layers/_148_PadConv_19_weights.h"
#include "layers/_200_Conv2D_27_biases.h"
#include "parameters/_158_Conv2D_20_weights.hpp"
#include "parameters/_231_Conv2D_33_weights.hpp"
#include "parameters/_255_PadConvDw_11_biases.hpp"
#include "layers/_0_PadConv_0.h"
#include "parameters/_222_Conv2D_31_weights.hpp"
#include "parameters/_50_Conv2D_5_weights.hpp"
#include "layers/_186_Conv2D_25_weights.h"
#include "parameters/_198_PadConvDw_5_biases.hpp"
#include "parameters/_21_PadConv_4_biases.hpp"
#include "parameters/_234_PadConvDw_9_biases.hpp"
#include "parameters/_18_Conv2D_1_weights.hpp"
#include "parameters/_49_PadConvDw_0_weights.hpp"
#include "layers/_249_PadConv_27.h"
#include "layers/_138_Sigmoid_35.h"
#include "parameters/_157_PadConv_21_weights.hpp"
#include "parameters/_49_PadConvDw_0_biases.hpp"
#include "layers/_164_Mul_42.h"
#include "parameters/_158_Conv2D_20_biases.hpp"
#include "parameters/_248_Conv2D_35_weights.hpp"
#include "layers/_2_Mul_0.h"
#include "layers/_48_Mul_13.h"
#include "parameters/_57_PadConv_10_biases.hpp"
#include "layers/_206_Conv2D_28.h"
#include "layers/_239_Conv2D_34_biases.h"
#include "layers/_172_PadConv_22_biases.h"
#include "layers/_110_Add_7.h"
#include "layers/_231_Conv2D_33.h"
#include "parameters/_189_PadConv_24_biases.hpp"
#include "layers/_175_Conv2D_23.h"
#include "parameters/_157_PadConv_21_biases.hpp"
#include "layers/_264_Conv2D_38_weights.h"
#include "parameters/_242_PadConv_26_weights.hpp"
#include "layers/_257_Transpose_4.h"
#include "layers/_137_PadConv_17.h"
#include "layers/_86_Add_5.h"
#include "layers/_117_Conv2D_16_biases.h"
#include "layers/_135_Mul_34.h"
#include "layers/_175_Conv2D_23_weights.h"
#include "layers/_49_PadConvDw_0.h"
#include "layers/_183_Transpose_1.h"
#include "layers/_13_PadConv_3.h"
#include "layers/_182_Mul_47.h"
#include "layers/_31_PadConv_6.h"
#include "parameters/_147_PadConvDw_2_weights.hpp"
#include "layers/_127_Concat_7.h"
#include "layers/_205_Mul_53.h"
#include "layers/_85_Mul_23.h"
#include "layers/_69_Conv2D_6_biases.h"
#include "layers/_261_Conv2D_37_weights.h"
#include "layers/_190_Conv2D_26_weights.h"
#include "layers/_161_Conv2D_21_biases.h"
#include "parameters/_249_PadConv_27_weights.hpp"
#include "layers/_156_Concat_10.h"
#include "layers/_243_PadConvDw_10.h"
#include "layers/_233_Mul_60.h"
#include "layers/_15_Mul_4.h"
#include "parameters/_91_Conv2D_10_weights.hpp"
#include "layers/_140_PadConv_18_weights.h"
#include "parameters/_209_PadConvDw_6_biases.hpp"
#include "layers/_38_PadConv_8_biases.h"
#include "layers/_132_Concat_8.h"
#include "layers/_80_PadConv_13_biases.h"
#include "parameters/_101_Conv2D_12_weights.hpp"
#include "layers/_77_Sigmoid_21.h"
#include "layers/_191_PadConvDw_4.h"
#include "layers/_232_Sigmoid_60.h"
#include "parameters/_28_PadConv_5_biases.hpp"
#include "layers/_46_Conv2D_4_weights.h"
#include "parameters/_106_Conv2D_13_biases.hpp"
#include "layers/_60_Add_3.h"
#include "layers/_237_Add_8.h"
#include "layers/_31_PadConv_6_weights.h"
#include "layers/_167_Conv2D_22.h"
#include "layers/_209_PadConvDw_6_weights.h"
#include "layers/_236_Mul_61.h"
#include "layers/_122_Sigmoid_31.h"
#include "layers/_24_Conv2D_2_biases.h"
#include "parameters/_223_Conv2D_32_biases.hpp"
#include "layers/_266_Reshape_5_prod_0.h"
#include "layers/_242_PadConv_26_weights.h"
#include "layers/_181_Conv2D_24_biases.h"
#include "layers/_126_Mul_32.h"
#include "layers/_222_Conv2D_31.h"
#include "parameters/_208_Conv2D_29_biases.hpp"
#include "layers/_50_Conv2D_5_weights.h"
#include "layers/_217_PadConvDw_7.h"
#include "layers/_173_Reshape_0.h"
#include "layers/_36_Sigmoid_10.h"
#include "parameters/_186_Conv2D_25_weights.hpp"
#include "layers/_87_Concat_3.h"
#include "parameters/_217_PadConvDw_7_biases.hpp"
#include "layers/_153_Mul_38.h"
#include "parameters/_10_PadConv_2_biases.hpp"
#include "layers/_193_Sigmoid_50.h"
#include "kernels/cpp/resize.hpp"
#include "layers/_12_Mul_3.h"
#include "layers/_35_PadConv_7_biases.h"
#include "parameters/_179_PadConv_23_biases.hpp"
#include "layers/_229_Sigmoid_59.h"
#include "layers/_216_Conv2D_30_biases.h"
#include "layers/_13_PadConv_3_biases.h"
#include "layers/_128_Conv2D_17_biases.h"
#include "layers/_228_PadConvDw_8_weights.h"
#include "layers/_37_Mul_10.h"
#include "layers/_50_Conv2D_5_biases.h"
#include "parameters/_133_Conv2D_18_biases.hpp"
#include "parameters/_223_Conv2D_32_weights.hpp"
#include "layers/_101_Conv2D_12_biases.h"
#include "layers/_200_Conv2D_27_weights.h"
#include "layers/_220_Mul_56.h"
#include "parameters/_181_Conv2D_24_biases.hpp"
#include "layers/_255_PadConvDw_11_weights.h"
#include "layers/_49_PadConvDw_0_weights.h"
#include "layers/_51_Sigmoid_14.h"
#include "layers/_230_Mul_59.h"
#include "kernels/cpp/convolution_depthwise.hpp"
#include "layers/_97_Concat_4.h"
#include "layers/_88_Conv2D_9_biases.h"
#include "layers/_216_Conv2D_30_weights.h"
#include "layers/_10_PadConv_2.h"
#include "layers/_7_Sigmoid_2.h"
#include "layers/_194_Sigmoid_51.h"
#include "parameters/_83_PadConv_14_weights.hpp"
#include "layers/_251_Sigmoid_66.h"
#include "layers/_198_PadConvDw_5_biases.h"
#include "layers/_148_PadConv_19_biases.h"
#include "parameters/_28_PadConv_5_weights.hpp"
#include "parameters/_172_PadConv_22_weights.hpp"
#include "layers/_118_Sigmoid_30.h"
#include "layers/_81_Sigmoid_22.h"
#include "parameters/_254_Conv2D_36_biases.hpp"
#include "layers/_72_Conv2D_7_biases.h"
#include "layers/_45_Mul_12.h"
#include "parameters/_148_PadConv_19_biases.hpp"
#include "layers/_149_PadConv_20_biases.h"
#include "layers/_103_Mul_27.h"
#include "parameters/_64_PadConv_12_biases.hpp"
#include "layers/_62_Sigmoid_17.h"
#include "parameters/_206_Conv2D_28_weights.hpp"
#include "layers/_221_Mul_57.h"
#include "layers/_243_PadConvDw_10_weights.h"
#include "layers/_83_PadConv_14.h"
#include "parameters/_200_Conv2D_27_weights.hpp"
#include "layers/_192_Sigmoid_49.h"
#include "layers/_84_Sigmoid_23.h"
#include "layers/_52_Mul_14.h"
#include "layers/_249_PadConv_27_weights.h"
#include "parameters/_128_Conv2D_17_weights.hpp"
#include "layers/_11_Sigmoid_3.h"
#include "layers/_75_PadConvDw_1.h"
#include "parameters/_13_PadConv_3_weights.hpp"
#include "layers/_223_Conv2D_32_weights.h"
#include "parameters/_144_Conv2D_19_weights.hpp"
#include "layers/_91_Conv2D_10_weights.h"
#include "layers/_98_Conv2D_11_biases.h"
#include "layers/_247_Mul_64.h"
#include "layers/_170_Transpose_0.h"
#include "layers/_179_PadConv_23_biases.h"
#include "layers/_259_Reshape_4.h"
#include "parameters/_216_Conv2D_30_weights.hpp"
#include "parameters/_228_PadConvDw_8_weights.hpp"
#include "parameters/_57_PadConv_10_weights.hpp"
#include "layers/_181_Conv2D_24_weights.h"
#include "layers/_158_Conv2D_20_biases.h"
#include "layers/_235_Sigmoid_61.h"
#include "layers/_181_Conv2D_24.h"
#include "parameters/_242_PadConv_26_biases.hpp"
#include "layers/_72_Conv2D_7.h"
#include "layers/_101_Conv2D_12.h"
#include "layers/_231_Conv2D_33_biases.h"
#include "layers/_254_Conv2D_36_weights.h"
#include "layers/_154_Mul_39.h"
#include "parameters/_61_PadConv_11_weights.hpp"
#include "layers/_27_Split_1.h"
#include "layers/_189_PadConv_24_biases.h"
#include "layers/_234_PadConvDw_9.h"
#include "layers/_82_Mul_22.h"
#include "layers/_112_Conv2D_15.h"
#include "parameters/_106_Conv2D_13_weights.hpp"
#include "layers/_149_PadConv_20_weights.h"
#include "layers/_24_Conv2D_2_weights.h"
#include "layers/_226_Mul_58.h"
#include "layers/_105_Add_6.h"
#include "layers/_61_PadConv_11_biases.h"
#include "parameters/_222_Conv2D_31_biases.hpp"
#include "parameters/_101_Conv2D_12_biases.hpp"
#include "parameters/_54_PadConv_9_weights.hpp"
#include "parameters/_80_PadConv_13_weights.hpp"
#include "layers/_18_Conv2D_1_biases.h"
#include "layers/_224_Sigmoid_58.h"
#include "layers/_130_Mul_33.h"
#include "layers/_163_Mul_41.h"
#include "layers/_242_PadConv_26.h"
#include "layers/_157_PadConv_21_biases.h"
#include "layers/_128_Conv2D_17.h"
#include "layers/_211_Mul_54.h"
#include "layers/_191_PadConvDw_4_biases.h"
#include "layers/_248_Conv2D_35_biases.h"
#include "parameters/_35_PadConv_7_weights.hpp"
#include "parameters/_191_PadConvDw_4_weights.hpp"
#include "layers/_9_Split_0.h"
#include "layers/_121_PadConv_15_biases.h"
#include "layers/_158_Conv2D_20.h"
#include "parameters/_43_Conv2D_3_biases.hpp"
#include "layers/_6_Conv2D_0.h"
#include "layers/_41_Add_2.h"
#include "layers/_4_Sigmoid_1.h"
#include "layers/_21_PadConv_4_weights.h"
#include "parameters/_31_PadConv_6_weights.hpp"
#include "layers/_106_Conv2D_13_biases.h"
#include "layers/_79_Split_3.h"
#include "layers/_46_Conv2D_4.h"
#include "layers/_0_PadConv_0_weights.h"
#include "layers/_83_PadConv_14_biases.h"
#include "parameters/_137_PadConv_17_biases.hpp"
#include "parameters/_69_Conv2D_6_weights.hpp"
#include "layers/_197_Mul_51.h"
#include "layers/_13_PadConv_3_weights.h"
#include "parameters/_248_Conv2D_35_biases.hpp"
#include "layers/_71_Mul_19.h"
#include "parameters/_24_Conv2D_2_weights.hpp"
#include "layers/_223_Conv2D_32.h"
#include "layers/_178_Mul_46.h"
#include "layers/_50_Conv2D_5.h"
#include "layers/_117_Conv2D_16.h"
#include "layers/_265_Transpose_5.h"
#include "layers/_29_Sigmoid_8.h"
#include "kernels/cpp/identity.hpp"
#include "parameters/_179_PadConv_23_weights.hpp"
#include "layers/_47_Sigmoid_13.h"
#include "layers/_172_PadConv_22.h"
#include "parameters/_75_PadConvDw_1_biases.hpp"
#include "layers/_142_Mul_36.h"
#include "layers/_67_Add_4.h"
#include "layers/_252_Mul_65.h"
#include "parameters/_161_Conv2D_21_weights.hpp"
#include "parameters/_133_Conv2D_18_weights.hpp"
#include "parameters/_109_Conv2D_14_biases.hpp"
#include "layers/_124_PadConv_16.h"
#include "layers/_112_Conv2D_15_biases.h"
#include "parameters/_261_Conv2D_37_weights.hpp"
#include "layers/_75_PadConvDw_1_biases.h"
#include "layers/_202_Sigmoid_52.h"
#include "layers/_261_Conv2D_37.h"
#include "layers/_69_Conv2D_6_weights.h"
#include "layers/_66_Mul_18.h"
#include "layers/_128_Conv2D_17_weights.h"
#include "layers/_264_Conv2D_38.h"
#include "layers/_167_Conv2D_22_biases.h"
#include "layers/_94_PadMaxPool_0.h"
#include "layers/_199_PadConv_25.h"
#include "kernels/cpp/convolution.hpp"
#include "parameters/_264_Conv2D_38_weights.hpp"
#include "layers/_254_Conv2D_36.h"
#include "layers/_44_Sigmoid_12.h"
#include "layers/_201_Concat_12.h"
#include "layers/_72_Conv2D_7_weights.h"
#include "layers/_43_Conv2D_3_biases.h"
#include "layers/_196_Mul_50.h"
#include "parameters/_140_PadConv_18_weights.hpp"
#include "parameters/_190_Conv2D_26_weights.hpp"
#include "layers/_10_PadConv_2_biases.h"
#include "layers/_101_Conv2D_12_weights.h"
#include "parameters/_61_PadConv_11_biases.hpp"
#include "layers/_206_Conv2D_28_biases.h"
#include "layers/_186_Conv2D_25_biases.h"
#include "parameters/_149_PadConv_20_biases.hpp"
#include "parameters/_46_Conv2D_4_weights.hpp"
#include "layers/_175_Conv2D_23_biases.h"
#include "layers/_73_Sigmoid_20.h"
#include "layers/_57_PadConv_10_biases.h"
#include "layers/_264_Conv2D_38_biases.h"
#include "layers/_18_Conv2D_1.h"
#include "layers/_249_PadConv_27_biases.h"
#include "layers/_160_Sigmoid_42.h"
#include "layers/_219_Sigmoid_57.h"
#include "kernels/cpp/concat.hpp"
#include "layers/_162_Sigmoid_43.h"
#include "layers/_255_PadConvDw_11_biases.h"
#include "parameters/_175_Conv2D_23_biases.hpp"
#include "parameters/_239_Conv2D_34_weights.hpp"
#include "layers/_147_PadConvDw_2.h"
#include "layers/_179_PadConv_23.h"
#include "layers/_28_PadConv_5_biases.h"
#include "layers/_34_Add_1.h"
#include "parameters/_117_Conv2D_16_weights.hpp"
#include "layers/_155_Mul_40.h"
#include "layers/_90_Mul_24.h"
#include "parameters/_149_PadConv_20_weights.hpp"
#include "parameters/_206_Conv2D_28_biases.hpp"
#include "layers/_159_Sigmoid_41.h"
#include "layers/_151_Sigmoid_39.h"
#include "layers/_133_Conv2D_18_weights.h"
#include "layers/_5_Mul_1.h"
#include "parameters/_10_PadConv_2_weights.hpp"
#include "parameters/_131_Resize_1_prod_0.hpp"
#include "parameters/_189_PadConv_24_weights.hpp"
#include "layers/_109_Conv2D_14_weights.h"
#include "layers/_208_Conv2D_29_biases.h"
#include "layers/_223_Conv2D_32_biases.h"
#include "layers/_22_Sigmoid_6.h"
#include "parameters/_43_Conv2D_3_weights.hpp"
#include "layers/_161_Conv2D_21.h"
#include "layers/_166_PadConvDw_3_biases.h"
#include "layers/_185_Reshape_1.h"
#include "layers/_46_Conv2D_4_biases.h"
#include "layers/_96_PadMaxPool_2.h"
#include "parameters/_191_PadConvDw_4_biases.hpp"
#include "layers/_107_Sigmoid_28.h"
#include "parameters/_64_PadConv_12_weights.hpp"
#include "layers/_150_Sigmoid_38.h"
#include "layers/_8_Mul_2.h"
#include "layers/_93_Mul_25.h"
#include "parameters/_88_Conv2D_9_biases.hpp"
#include "layers/_190_Conv2D_26_biases.h"
#include "layers/_168_Split_7.h"
#include "layers/_28_PadConv_5_weights.h"
#include "parameters/_190_Conv2D_26_biases.hpp"
#include "layers/_53_Split_2.h"
#include "layers/_243_PadConvDw_10_biases.h"
#include "layers/_189_PadConv_24_weights.h"
#include "layers/_166_PadConvDw_3_weights.h"
#include "layers/_144_Conv2D_19_biases.h"
#include "layers/_260_Concat_14.h"
#include "parameters/_21_PadConv_4_weights.hpp"
#include "layers/_157_PadConv_21.h"
#include "layers/_88_Conv2D_9_weights.h"
#include "layers/_215_Reshape_2.h"
#include "layers/_209_PadConvDw_6.h"
#include "layers/_109_Conv2D_14.h"
#include "layers/_245_Sigmoid_64.h"
#include "layers/_0_PadConv_0_biases.h"
#include "parameters/_137_PadConv_17_weights.hpp"
#include "parameters/_112_Conv2D_15_biases.hpp"
#include "parameters/_98_Conv2D_11_biases.hpp"
#include "layers/_64_PadConv_12_biases.h"
#include "parameters/_243_PadConvDw_10_biases.hpp"
#include "layers/_80_PadConv_13_weights.h"
#include "layers/_187_Sigmoid_48.h"
#include "layers/_76_Conv2D_8_biases.h"
#include "layers/_217_PadConvDw_7_biases.h"
#include "parameters/_167_Conv2D_22_weights.hpp"
#include "layers/_189_PadConv_24.h"
#include "layers/_83_PadConv_14_weights.h"
#include "parameters/_172_PadConv_22_biases.hpp"
#include "layers/_123_Mul_31.h"
#include "parameters/_13_PadConv_3_biases.hpp"
#include "parameters/_166_PadConvDw_3_weights.hpp"
#include "layers/_176_Mul_45.h"
#include "layers/_198_PadConvDw_5.h"
#include "layers/_184_Concat_11.h"
#include "parameters/_175_Conv2D_23_weights.hpp"
#include "layers/_148_PadConv_19.h"
#include "layers/_76_Conv2D_8.h"
#include "parameters/_76_Conv2D_8_biases.hpp"
#include "layers/_168_Split_7_prod_0.h"
#include "layers/_35_PadConv_7.h"
#include "parameters/_124_PadConv_16_biases.hpp"
#include "layers/_54_PadConv_9.h"
#include "layers/_74_Mul_20.h"
#include "layers/_6_Conv2D_0_weights.h"
#include "layers/_102_Sigmoid_27.h"
#include "layers/_20_Mul_5.h"
#include "layers/_262_Sigmoid_68.h"
#include "layers/_59_Mul_16.h"
#include "layers/_43_Conv2D_3.h"
#include "layers/_100_Mul_26.h"
#include "layers/_136_Split_6.h"
#include "layers/_253_Mul_66.h"
#include "layers/_23_Mul_6.h"
#include "parameters/_261_Conv2D_37_biases.hpp"
#include "layers/_49_PadConvDw_0_biases.h"
#include "layers/_28_PadConv_5.h"
#include "layers/_195_Mul_49.h"
#include "layers/_114_Mul_29.h"
#include "parameters/_24_Conv2D_2_biases.hpp"
#include "layers/_3_PadConv_1_weights.h"
#include "layers/_261_Conv2D_37_biases.h"
#include "layers/_208_Conv2D_29_weights.h"
#include "layers/_140_PadConv_18_biases.h"
#include "parameters/_147_PadConvDw_2_biases.hpp"
#include "layers/_239_Conv2D_34.h"
#include "layers/_40_Mul_11.h"
#include "layers/_143_Concat_9.h"
#include "layers/_16_Add_0.h"
#include "parameters/_38_PadConv_8_weights.hpp"
#include "layers/_248_Conv2D_35_weights.h"
#include "layers/_54_PadConv_9_weights.h"
#include "layers/_98_Conv2D_11_weights.h"
#include "layers/_137_PadConv_17_biases.h"
#include "layers/_161_Conv2D_21_weights.h"
#include "layers/_111_Concat_5.h"
#include "layers/_166_PadConvDw_3.h"
#include "layers/_266_Reshape_5.h"
#include "parameters/_249_PadConv_27_biases.hpp"
#include "layers/_171_Mul_44.h"
#include "kernels/cpp/sigmoid.hpp"
#include "layers/_129_Sigmoid_33.h"
#include "parameters/_148_PadConv_19_weights.hpp"
#include "layers/_115_Resize_0.h"
#include "layers/_137_PadConv_17_weights.h"
#include "layers/_267_Concat_15.h"
#include "parameters/_217_PadConvDw_7_weights.hpp"
#include "layers/_38_PadConv_8.h"
#include "parameters/_109_Conv2D_14_weights.hpp"
#include "parameters/_128_Conv2D_17_biases.hpp"
#include "layers/_26_Mul_7.h"
#include "parameters/_72_Conv2D_7_biases.hpp"
#include "layers/_134_Sigmoid_34.h"
#include "layers/_225_Transpose_3.h"
#include "parameters/_209_PadConvDw_6_weights.hpp"
#include "layers/_177_Sigmoid_46.h"
#include "layers/_174_Sigmoid_45.h"
#include "parameters/_75_PadConvDw_1_weights.hpp"
#include "layers/_35_PadConv_7_weights.h"
#include "layers/_165_Mul_43.h"
#include "parameters/_254_Conv2D_36_weights.hpp"
#include "layers/_144_Conv2D_19.h"
#include "layers/_254_Conv2D_36_biases.h"
#include "layers/_91_Conv2D_10_biases.h"
#include "parameters/_0_PadConv_0_biases.hpp"
#include "layers/_248_Conv2D_35.h"
#include "layers/_227_Reshape_3.h"
#include "layers/_157_PadConv_21_weights.h"
#include "layers/_204_Mul_52.h"
#include "layers/_9_Split_0_prod_0.h"

// Memory block
static unsigned char mem_0[19660800];


void model_forward (const float* _0_PadConv_0_input_0,float** _260_Concat_14_output_0_ptr, float** _267_Concat_15_output_0_ptr, int64_t** _94_PadMaxPool_0_output_1_ptr, int64_t** _95_PadMaxPool_1_output_1_ptr, int64_t** _96_PadMaxPool_2_output_1_ptr)
{
    
    

    float* _0_PadConv_0_output_0 = (float*) (mem_0 + _0_PADCONV_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_0_PADCONV_0_INPUT_0_IN_BATCH,
                           _0_PADCONV_0_NB_CHANNELS,
                           _0_PADCONV_0_INPUT_0_IN_HEIGHT,
                           _0_PADCONV_0_INPUT_0_IN_WIDTH,
                           _0_PADCONV_0_NB_OUTPUTS,
                           _0_PADCONV_0_OUTPUT_0_OUT_HEIGHT,
                           _0_PADCONV_0_OUTPUT_0_OUT_WIDTH,
                           _0_PADCONV_0_PADDING_Y,
                           _0_PADCONV_0_PADDING_X,
                           _0_PADCONV_0_STRIDE_Y,
                           _0_PADCONV_0_STRIDE_X,
                           _0_PADCONV_0_DILATION_Y,
                           _0_PADCONV_0_DILATION_X,
                           _0_PADCONV_0_KERNEL_HEIGHT,
                           _0_PADCONV_0_KERNEL_WIDTH,
                           _0_PADCONV_0_ACTIVATION,
                           _0_PADCONV_0_INPUT_0_MEM_CONT_OFFSET,
                           _0_PADCONV_0_INPUT_0_MEM_CONT_SIZE,
                           _0_PADCONV_0_INPUT_0_MEM_WRAP_OFFSET,
                           _0_PADCONV_0_INPUT_0_MEM_WRAP_SIZE,
                           _0_PADCONV_0_INPUT_0_MEM_STRIDE,
                           _0_PADCONV_0_OUTPUT_0_MEM_CONT_OFFSET,
                           _0_PADCONV_0_OUTPUT_0_MEM_CONT_SIZE,
                           _0_PADCONV_0_OUTPUT_0_MEM_WRAP_OFFSET,
                           _0_PADCONV_0_OUTPUT_0_MEM_WRAP_SIZE,
                           _0_PADCONV_0_OUTPUT_0_MEM_STRIDE>
                           (_0_PadConv_0_input_0, _0_PadConv_0_output_0, _0_PadConv_0_weights_output_0, _0_PadConv_0_biases_output_0, _0_PADCONV_0_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _0_PadConv_0\n");
    FILE* _0_PADCONV_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_0_PadConv_0_output_0.txt", "w");
    export_cpp::saveOutputs<_0_PADCONV_0_OUTPUT_0_NB_OUTPUTS,
                            _0_PADCONV_0_OUTPUT_0_OUT_HEIGHT,
                            _0_PADCONV_0_OUTPUT_0_OUT_WIDTH,
                            _0_PADCONV_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _0_PADCONV_0_OUTPUT_0_MEM_CONT_SIZE,
                            _0_PADCONV_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _0_PADCONV_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _0_PADCONV_0_OUTPUT_0_DEV_FMT>
                            (_0_PadConv_0_output_0,
                            _0_PADCONV_0_OUTPUT_0_STREAM);
    fclose(_0_PADCONV_0_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _1_Sigmoid_0_output_0 = (float*) (mem_0 + _1_SIGMOID_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_1_SIGMOID_0_NB_ELTS>
                               (_0_PadConv_0_output_0, _1_Sigmoid_0_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _1_Sigmoid_0\n");
    FILE* _1_SIGMOID_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_1_Sigmoid_0_output_0.txt", "w");
    export_cpp::saveOutputs<_1_SIGMOID_0_OUTPUT_0_NB_OUTPUTS,
                            _1_SIGMOID_0_OUTPUT_0_OUT_HEIGHT,
                            _1_SIGMOID_0_OUTPUT_0_OUT_WIDTH,
                            _1_SIGMOID_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _1_SIGMOID_0_OUTPUT_0_MEM_CONT_SIZE,
                            _1_SIGMOID_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _1_SIGMOID_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _1_SIGMOID_0_OUTPUT_0_DEV_FMT>
                            (_1_Sigmoid_0_output_0,
                            _1_SIGMOID_0_OUTPUT_0_STREAM);
    fclose(_1_SIGMOID_0_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _2_Mul_0_output_0 = (float*) (mem_0 + _2_MUL_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_2_MUL_0_NB_MAT,
                                 _2_MUL_0_ELEM_OP,
                                 _2_MUL_0_INPUT1_CONT_SIZE,
                                 _2_MUL_0_INPUT2_CONT_SIZE,
                                 _2_MUL_0_OUTPUT_CONT_SIZE,
                                 _2_MUL_0_OFFSET_IN1,
                                 _2_MUL_0_OFFSET_IN2,
                                 _2_MUL_0_ACTIVATION,
                                 _0_PADCONV_0_OUTPUT_0_MEM_CONT_OFFSET,
                                 _0_PADCONV_0_OUTPUT_0_MEM_CONT_SIZE,
                                 _0_PADCONV_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _0_PADCONV_0_OUTPUT_0_MEM_WRAP_SIZE,
                                 _0_PADCONV_0_OUTPUT_0_MEM_STRIDE,
                                 _1_SIGMOID_0_OUTPUT_0_MEM_CONT_OFFSET,
                                 _1_SIGMOID_0_OUTPUT_0_MEM_CONT_SIZE,
                                 _1_SIGMOID_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _1_SIGMOID_0_OUTPUT_0_MEM_WRAP_SIZE,
                                 _1_SIGMOID_0_OUTPUT_0_MEM_STRIDE,
                                 _2_MUL_0_OUTPUT_0_MEM_CONT_OFFSET,
                                 _2_MUL_0_OUTPUT_0_MEM_CONT_SIZE,
                                 _2_MUL_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _2_MUL_0_OUTPUT_0_MEM_WRAP_SIZE,
                                 _2_MUL_0_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_2_Mul_0_output_0, _2_MUL_0_RESCALING, _0_PadConv_0_output_0, _1_Sigmoid_0_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _2_Mul_0\n");
    FILE* _2_MUL_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_2_Mul_0_output_0.txt", "w");
    export_cpp::saveOutputs<_2_MUL_0_OUTPUT_0_NB_OUTPUTS,
                            _2_MUL_0_OUTPUT_0_OUT_HEIGHT,
                            _2_MUL_0_OUTPUT_0_OUT_WIDTH,
                            _2_MUL_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _2_MUL_0_OUTPUT_0_MEM_CONT_SIZE,
                            _2_MUL_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _2_MUL_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _2_MUL_0_OUTPUT_0_DEV_FMT>
                            (_2_Mul_0_output_0,
                            _2_MUL_0_OUTPUT_0_STREAM);
    fclose(_2_MUL_0_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _3_PadConv_1_output_0 = (float*) (mem_0 + _3_PADCONV_1_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_2_MUL_0_OUTPUT_0_IN_BATCH,
                           _3_PADCONV_1_NB_CHANNELS,
                           _2_MUL_0_OUTPUT_0_IN_HEIGHT,
                           _2_MUL_0_OUTPUT_0_IN_WIDTH,
                           _3_PADCONV_1_NB_OUTPUTS,
                           _3_PADCONV_1_OUTPUT_0_OUT_HEIGHT,
                           _3_PADCONV_1_OUTPUT_0_OUT_WIDTH,
                           _3_PADCONV_1_PADDING_Y,
                           _3_PADCONV_1_PADDING_X,
                           _3_PADCONV_1_STRIDE_Y,
                           _3_PADCONV_1_STRIDE_X,
                           _3_PADCONV_1_DILATION_Y,
                           _3_PADCONV_1_DILATION_X,
                           _3_PADCONV_1_KERNEL_HEIGHT,
                           _3_PADCONV_1_KERNEL_WIDTH,
                           _3_PADCONV_1_ACTIVATION,
                           _2_MUL_0_OUTPUT_0_MEM_CONT_OFFSET,
                           _2_MUL_0_OUTPUT_0_MEM_CONT_SIZE,
                           _2_MUL_0_OUTPUT_0_MEM_WRAP_OFFSET,
                           _2_MUL_0_OUTPUT_0_MEM_WRAP_SIZE,
                           _2_MUL_0_OUTPUT_0_MEM_STRIDE,
                           _3_PADCONV_1_OUTPUT_0_MEM_CONT_OFFSET,
                           _3_PADCONV_1_OUTPUT_0_MEM_CONT_SIZE,
                           _3_PADCONV_1_OUTPUT_0_MEM_WRAP_OFFSET,
                           _3_PADCONV_1_OUTPUT_0_MEM_WRAP_SIZE,
                           _3_PADCONV_1_OUTPUT_0_MEM_STRIDE>
                           (_2_Mul_0_output_0, _3_PadConv_1_output_0, _3_PadConv_1_weights_output_0, _3_PadConv_1_biases_output_0, _3_PADCONV_1_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _3_PadConv_1\n");
    FILE* _3_PADCONV_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_3_PadConv_1_output_0.txt", "w");
    export_cpp::saveOutputs<_3_PADCONV_1_OUTPUT_0_NB_OUTPUTS,
                            _3_PADCONV_1_OUTPUT_0_OUT_HEIGHT,
                            _3_PADCONV_1_OUTPUT_0_OUT_WIDTH,
                            _3_PADCONV_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _3_PADCONV_1_OUTPUT_0_MEM_CONT_SIZE,
                            _3_PADCONV_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _3_PADCONV_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _3_PADCONV_1_OUTPUT_0_DEV_FMT>
                            (_3_PadConv_1_output_0,
                            _3_PADCONV_1_OUTPUT_0_STREAM);
    fclose(_3_PADCONV_1_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _4_Sigmoid_1_output_0 = (float*) (mem_0 + _4_SIGMOID_1_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_4_SIGMOID_1_NB_ELTS>
                               (_3_PadConv_1_output_0, _4_Sigmoid_1_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _4_Sigmoid_1\n");
    FILE* _4_SIGMOID_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_4_Sigmoid_1_output_0.txt", "w");
    export_cpp::saveOutputs<_4_SIGMOID_1_OUTPUT_0_NB_OUTPUTS,
                            _4_SIGMOID_1_OUTPUT_0_OUT_HEIGHT,
                            _4_SIGMOID_1_OUTPUT_0_OUT_WIDTH,
                            _4_SIGMOID_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _4_SIGMOID_1_OUTPUT_0_MEM_CONT_SIZE,
                            _4_SIGMOID_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _4_SIGMOID_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _4_SIGMOID_1_OUTPUT_0_DEV_FMT>
                            (_4_Sigmoid_1_output_0,
                            _4_SIGMOID_1_OUTPUT_0_STREAM);
    fclose(_4_SIGMOID_1_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _5_Mul_1_output_0 = (float*) (mem_0 + _5_MUL_1_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_5_MUL_1_NB_MAT,
                                 _5_MUL_1_ELEM_OP,
                                 _5_MUL_1_INPUT1_CONT_SIZE,
                                 _5_MUL_1_INPUT2_CONT_SIZE,
                                 _5_MUL_1_OUTPUT_CONT_SIZE,
                                 _5_MUL_1_OFFSET_IN1,
                                 _5_MUL_1_OFFSET_IN2,
                                 _5_MUL_1_ACTIVATION,
                                 _3_PADCONV_1_OUTPUT_0_MEM_CONT_OFFSET,
                                 _3_PADCONV_1_OUTPUT_0_MEM_CONT_SIZE,
                                 _3_PADCONV_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _3_PADCONV_1_OUTPUT_0_MEM_WRAP_SIZE,
                                 _3_PADCONV_1_OUTPUT_0_MEM_STRIDE,
                                 _4_SIGMOID_1_OUTPUT_0_MEM_CONT_OFFSET,
                                 _4_SIGMOID_1_OUTPUT_0_MEM_CONT_SIZE,
                                 _4_SIGMOID_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _4_SIGMOID_1_OUTPUT_0_MEM_WRAP_SIZE,
                                 _4_SIGMOID_1_OUTPUT_0_MEM_STRIDE,
                                 _5_MUL_1_OUTPUT_0_MEM_CONT_OFFSET,
                                 _5_MUL_1_OUTPUT_0_MEM_CONT_SIZE,
                                 _5_MUL_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _5_MUL_1_OUTPUT_0_MEM_WRAP_SIZE,
                                 _5_MUL_1_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_5_Mul_1_output_0, _5_MUL_1_RESCALING, _3_PadConv_1_output_0, _4_Sigmoid_1_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _5_Mul_1\n");
    FILE* _5_MUL_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_5_Mul_1_output_0.txt", "w");
    export_cpp::saveOutputs<_5_MUL_1_OUTPUT_0_NB_OUTPUTS,
                            _5_MUL_1_OUTPUT_0_OUT_HEIGHT,
                            _5_MUL_1_OUTPUT_0_OUT_WIDTH,
                            _5_MUL_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _5_MUL_1_OUTPUT_0_MEM_CONT_SIZE,
                            _5_MUL_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _5_MUL_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _5_MUL_1_OUTPUT_0_DEV_FMT>
                            (_5_Mul_1_output_0,
                            _5_MUL_1_OUTPUT_0_STREAM);
    fclose(_5_MUL_1_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _6_Conv2D_0_output_0 = (float*) (mem_0 + _6_CONV2D_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_5_MUL_1_OUTPUT_0_IN_BATCH,
                           _6_CONV2D_0_NB_CHANNELS,
                           _5_MUL_1_OUTPUT_0_IN_HEIGHT,
                           _5_MUL_1_OUTPUT_0_IN_WIDTH,
                           _6_CONV2D_0_NB_OUTPUTS,
                           _6_CONV2D_0_OUTPUT_0_OUT_HEIGHT,
                           _6_CONV2D_0_OUTPUT_0_OUT_WIDTH,
                           _6_CONV2D_0_PADDING_Y,
                           _6_CONV2D_0_PADDING_X,
                           _6_CONV2D_0_STRIDE_Y,
                           _6_CONV2D_0_STRIDE_X,
                           _6_CONV2D_0_DILATION_Y,
                           _6_CONV2D_0_DILATION_X,
                           _6_CONV2D_0_KERNEL_HEIGHT,
                           _6_CONV2D_0_KERNEL_WIDTH,
                           _6_CONV2D_0_ACTIVATION,
                           _5_MUL_1_OUTPUT_0_MEM_CONT_OFFSET,
                           _5_MUL_1_OUTPUT_0_MEM_CONT_SIZE,
                           _5_MUL_1_OUTPUT_0_MEM_WRAP_OFFSET,
                           _5_MUL_1_OUTPUT_0_MEM_WRAP_SIZE,
                           _5_MUL_1_OUTPUT_0_MEM_STRIDE,
                           _6_CONV2D_0_OUTPUT_0_MEM_CONT_OFFSET,
                           _6_CONV2D_0_OUTPUT_0_MEM_CONT_SIZE,
                           _6_CONV2D_0_OUTPUT_0_MEM_WRAP_OFFSET,
                           _6_CONV2D_0_OUTPUT_0_MEM_WRAP_SIZE,
                           _6_CONV2D_0_OUTPUT_0_MEM_STRIDE>
                           (_5_Mul_1_output_0, _6_Conv2D_0_output_0, _6_Conv2D_0_weights_output_0, _6_Conv2D_0_biases_output_0, _6_CONV2D_0_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _6_Conv2D_0\n");
    FILE* _6_CONV2D_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_6_Conv2D_0_output_0.txt", "w");
    export_cpp::saveOutputs<_6_CONV2D_0_OUTPUT_0_NB_OUTPUTS,
                            _6_CONV2D_0_OUTPUT_0_OUT_HEIGHT,
                            _6_CONV2D_0_OUTPUT_0_OUT_WIDTH,
                            _6_CONV2D_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _6_CONV2D_0_OUTPUT_0_MEM_CONT_SIZE,
                            _6_CONV2D_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _6_CONV2D_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _6_CONV2D_0_OUTPUT_0_DEV_FMT>
                            (_6_Conv2D_0_output_0,
                            _6_CONV2D_0_OUTPUT_0_STREAM);
    fclose(_6_CONV2D_0_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _7_Sigmoid_2_output_0 = (float*) (mem_0 + _7_SIGMOID_2_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_7_SIGMOID_2_NB_ELTS>
                               (_6_Conv2D_0_output_0, _7_Sigmoid_2_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _7_Sigmoid_2\n");
    FILE* _7_SIGMOID_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_7_Sigmoid_2_output_0.txt", "w");
    export_cpp::saveOutputs<_7_SIGMOID_2_OUTPUT_0_NB_OUTPUTS,
                            _7_SIGMOID_2_OUTPUT_0_OUT_HEIGHT,
                            _7_SIGMOID_2_OUTPUT_0_OUT_WIDTH,
                            _7_SIGMOID_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _7_SIGMOID_2_OUTPUT_0_MEM_CONT_SIZE,
                            _7_SIGMOID_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _7_SIGMOID_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _7_SIGMOID_2_OUTPUT_0_DEV_FMT>
                            (_7_Sigmoid_2_output_0,
                            _7_SIGMOID_2_OUTPUT_0_STREAM);
    fclose(_7_SIGMOID_2_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _8_Mul_2_output_0 = (float*) (mem_0 + _8_MUL_2_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_8_MUL_2_NB_MAT,
                                 _8_MUL_2_ELEM_OP,
                                 _8_MUL_2_INPUT1_CONT_SIZE,
                                 _8_MUL_2_INPUT2_CONT_SIZE,
                                 _8_MUL_2_OUTPUT_CONT_SIZE,
                                 _8_MUL_2_OFFSET_IN1,
                                 _8_MUL_2_OFFSET_IN2,
                                 _8_MUL_2_ACTIVATION,
                                 _6_CONV2D_0_OUTPUT_0_MEM_CONT_OFFSET,
                                 _6_CONV2D_0_OUTPUT_0_MEM_CONT_SIZE,
                                 _6_CONV2D_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _6_CONV2D_0_OUTPUT_0_MEM_WRAP_SIZE,
                                 _6_CONV2D_0_OUTPUT_0_MEM_STRIDE,
                                 _7_SIGMOID_2_OUTPUT_0_MEM_CONT_OFFSET,
                                 _7_SIGMOID_2_OUTPUT_0_MEM_CONT_SIZE,
                                 _7_SIGMOID_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _7_SIGMOID_2_OUTPUT_0_MEM_WRAP_SIZE,
                                 _7_SIGMOID_2_OUTPUT_0_MEM_STRIDE,
                                 _8_MUL_2_OUTPUT_0_MEM_CONT_OFFSET,
                                 _8_MUL_2_OUTPUT_0_MEM_CONT_SIZE,
                                 _8_MUL_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _8_MUL_2_OUTPUT_0_MEM_WRAP_SIZE,
                                 _8_MUL_2_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_8_Mul_2_output_0, _8_MUL_2_RESCALING, _6_Conv2D_0_output_0, _7_Sigmoid_2_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _8_Mul_2\n");
    FILE* _8_MUL_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_8_Mul_2_output_0.txt", "w");
    export_cpp::saveOutputs<_8_MUL_2_OUTPUT_0_NB_OUTPUTS,
                            _8_MUL_2_OUTPUT_0_OUT_HEIGHT,
                            _8_MUL_2_OUTPUT_0_OUT_WIDTH,
                            _8_MUL_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _8_MUL_2_OUTPUT_0_MEM_CONT_SIZE,
                            _8_MUL_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _8_MUL_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _8_MUL_2_OUTPUT_0_DEV_FMT>
                            (_8_Mul_2_output_0,
                            _8_MUL_2_OUTPUT_0_STREAM);
    fclose(_8_MUL_2_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _9_Split_0_output_0 = (float*) (mem_0 + _9_SPLIT_0_OUTPUT_0_MEM_OFFSET);
    float* _9_Split_0_output_1 = (float*) (mem_0 + _9_SPLIT_0_OUTPUT_1_MEM_OFFSET);



    float* __9_Split_0_outputs[_9_SPLIT_0_NB_OUTPUTS] = {
        
        _9_Split_0_output_0,
        
        _9_Split_0_output_1
        
    };

    export_cpp::split_forward<_9_SPLIT_0_NB_OUTPUTS,
                              _9_SPLIT_0_AXIS_DIM,
                              _9_SPLIT_0_PRE_AXIS_ELEMS,
                              _9_SPLIT_0_POST_AXIS_ELEMS,
                              _9_SPLIT_0_SPLITS,
                              float>
                              (_8_Mul_2_output_0,
                               __9_Split_0_outputs);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _9_Split_0\n");
    FILE* _9_SPLIT_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_9_Split_0_output_0.txt", "w");
    export_cpp::saveOutputs<_9_SPLIT_0_OUTPUT_0_NB_OUTPUTS,
                            _9_SPLIT_0_OUTPUT_0_OUT_HEIGHT,
                            _9_SPLIT_0_OUTPUT_0_OUT_WIDTH,
                            _9_SPLIT_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _9_SPLIT_0_OUTPUT_0_MEM_CONT_SIZE,
                            _9_SPLIT_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _9_SPLIT_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _9_SPLIT_0_OUTPUT_0_DEV_FMT>
                            (_9_Split_0_output_0,
                            _9_SPLIT_0_OUTPUT_0_STREAM);
    fclose(_9_SPLIT_0_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _10_PadConv_2_output_0 = (float*) (mem_0 + _10_PADCONV_2_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_9_SPLIT_0_OUTPUT_1_IN_BATCH,
                           _10_PADCONV_2_NB_CHANNELS,
                           _9_SPLIT_0_OUTPUT_1_IN_HEIGHT,
                           _9_SPLIT_0_OUTPUT_1_IN_WIDTH,
                           _10_PADCONV_2_NB_OUTPUTS,
                           _10_PADCONV_2_OUTPUT_0_OUT_HEIGHT,
                           _10_PADCONV_2_OUTPUT_0_OUT_WIDTH,
                           _10_PADCONV_2_PADDING_Y,
                           _10_PADCONV_2_PADDING_X,
                           _10_PADCONV_2_STRIDE_Y,
                           _10_PADCONV_2_STRIDE_X,
                           _10_PADCONV_2_DILATION_Y,
                           _10_PADCONV_2_DILATION_X,
                           _10_PADCONV_2_KERNEL_HEIGHT,
                           _10_PADCONV_2_KERNEL_WIDTH,
                           _10_PADCONV_2_ACTIVATION,
                           _9_SPLIT_0_OUTPUT_1_MEM_CONT_OFFSET,
                           _9_SPLIT_0_OUTPUT_1_MEM_CONT_SIZE,
                           _9_SPLIT_0_OUTPUT_1_MEM_WRAP_OFFSET,
                           _9_SPLIT_0_OUTPUT_1_MEM_WRAP_SIZE,
                           _9_SPLIT_0_OUTPUT_1_MEM_STRIDE,
                           _10_PADCONV_2_OUTPUT_0_MEM_CONT_OFFSET,
                           _10_PADCONV_2_OUTPUT_0_MEM_CONT_SIZE,
                           _10_PADCONV_2_OUTPUT_0_MEM_WRAP_OFFSET,
                           _10_PADCONV_2_OUTPUT_0_MEM_WRAP_SIZE,
                           _10_PADCONV_2_OUTPUT_0_MEM_STRIDE>
                           (_9_Split_0_output_1, _10_PadConv_2_output_0, _10_PadConv_2_weights_output_0, _10_PadConv_2_biases_output_0, _10_PADCONV_2_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _10_PadConv_2\n");
    FILE* _10_PADCONV_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_10_PadConv_2_output_0.txt", "w");
    export_cpp::saveOutputs<_10_PADCONV_2_OUTPUT_0_NB_OUTPUTS,
                            _10_PADCONV_2_OUTPUT_0_OUT_HEIGHT,
                            _10_PADCONV_2_OUTPUT_0_OUT_WIDTH,
                            _10_PADCONV_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _10_PADCONV_2_OUTPUT_0_MEM_CONT_SIZE,
                            _10_PADCONV_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _10_PADCONV_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _10_PADCONV_2_OUTPUT_0_DEV_FMT>
                            (_10_PadConv_2_output_0,
                            _10_PADCONV_2_OUTPUT_0_STREAM);
    fclose(_10_PADCONV_2_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _11_Sigmoid_3_output_0 = (float*) (mem_0 + _11_SIGMOID_3_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_11_SIGMOID_3_NB_ELTS>
                               (_10_PadConv_2_output_0, _11_Sigmoid_3_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _11_Sigmoid_3\n");
    FILE* _11_SIGMOID_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_11_Sigmoid_3_output_0.txt", "w");
    export_cpp::saveOutputs<_11_SIGMOID_3_OUTPUT_0_NB_OUTPUTS,
                            _11_SIGMOID_3_OUTPUT_0_OUT_HEIGHT,
                            _11_SIGMOID_3_OUTPUT_0_OUT_WIDTH,
                            _11_SIGMOID_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _11_SIGMOID_3_OUTPUT_0_MEM_CONT_SIZE,
                            _11_SIGMOID_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _11_SIGMOID_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _11_SIGMOID_3_OUTPUT_0_DEV_FMT>
                            (_11_Sigmoid_3_output_0,
                            _11_SIGMOID_3_OUTPUT_0_STREAM);
    fclose(_11_SIGMOID_3_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _12_Mul_3_output_0 = (float*) (mem_0 + _12_MUL_3_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_12_MUL_3_NB_MAT,
                                 _12_MUL_3_ELEM_OP,
                                 _12_MUL_3_INPUT1_CONT_SIZE,
                                 _12_MUL_3_INPUT2_CONT_SIZE,
                                 _12_MUL_3_OUTPUT_CONT_SIZE,
                                 _12_MUL_3_OFFSET_IN1,
                                 _12_MUL_3_OFFSET_IN2,
                                 _12_MUL_3_ACTIVATION,
                                 _10_PADCONV_2_OUTPUT_0_MEM_CONT_OFFSET,
                                 _10_PADCONV_2_OUTPUT_0_MEM_CONT_SIZE,
                                 _10_PADCONV_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _10_PADCONV_2_OUTPUT_0_MEM_WRAP_SIZE,
                                 _10_PADCONV_2_OUTPUT_0_MEM_STRIDE,
                                 _11_SIGMOID_3_OUTPUT_0_MEM_CONT_OFFSET,
                                 _11_SIGMOID_3_OUTPUT_0_MEM_CONT_SIZE,
                                 _11_SIGMOID_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _11_SIGMOID_3_OUTPUT_0_MEM_WRAP_SIZE,
                                 _11_SIGMOID_3_OUTPUT_0_MEM_STRIDE,
                                 _12_MUL_3_OUTPUT_0_MEM_CONT_OFFSET,
                                 _12_MUL_3_OUTPUT_0_MEM_CONT_SIZE,
                                 _12_MUL_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _12_MUL_3_OUTPUT_0_MEM_WRAP_SIZE,
                                 _12_MUL_3_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_12_Mul_3_output_0, _12_MUL_3_RESCALING, _10_PadConv_2_output_0, _11_Sigmoid_3_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _12_Mul_3\n");
    FILE* _12_MUL_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_12_Mul_3_output_0.txt", "w");
    export_cpp::saveOutputs<_12_MUL_3_OUTPUT_0_NB_OUTPUTS,
                            _12_MUL_3_OUTPUT_0_OUT_HEIGHT,
                            _12_MUL_3_OUTPUT_0_OUT_WIDTH,
                            _12_MUL_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _12_MUL_3_OUTPUT_0_MEM_CONT_SIZE,
                            _12_MUL_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _12_MUL_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _12_MUL_3_OUTPUT_0_DEV_FMT>
                            (_12_Mul_3_output_0,
                            _12_MUL_3_OUTPUT_0_STREAM);
    fclose(_12_MUL_3_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _13_PadConv_3_output_0 = (float*) (mem_0 + _13_PADCONV_3_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_12_MUL_3_OUTPUT_0_IN_BATCH,
                           _13_PADCONV_3_NB_CHANNELS,
                           _12_MUL_3_OUTPUT_0_IN_HEIGHT,
                           _12_MUL_3_OUTPUT_0_IN_WIDTH,
                           _13_PADCONV_3_NB_OUTPUTS,
                           _13_PADCONV_3_OUTPUT_0_OUT_HEIGHT,
                           _13_PADCONV_3_OUTPUT_0_OUT_WIDTH,
                           _13_PADCONV_3_PADDING_Y,
                           _13_PADCONV_3_PADDING_X,
                           _13_PADCONV_3_STRIDE_Y,
                           _13_PADCONV_3_STRIDE_X,
                           _13_PADCONV_3_DILATION_Y,
                           _13_PADCONV_3_DILATION_X,
                           _13_PADCONV_3_KERNEL_HEIGHT,
                           _13_PADCONV_3_KERNEL_WIDTH,
                           _13_PADCONV_3_ACTIVATION,
                           _12_MUL_3_OUTPUT_0_MEM_CONT_OFFSET,
                           _12_MUL_3_OUTPUT_0_MEM_CONT_SIZE,
                           _12_MUL_3_OUTPUT_0_MEM_WRAP_OFFSET,
                           _12_MUL_3_OUTPUT_0_MEM_WRAP_SIZE,
                           _12_MUL_3_OUTPUT_0_MEM_STRIDE,
                           _13_PADCONV_3_OUTPUT_0_MEM_CONT_OFFSET,
                           _13_PADCONV_3_OUTPUT_0_MEM_CONT_SIZE,
                           _13_PADCONV_3_OUTPUT_0_MEM_WRAP_OFFSET,
                           _13_PADCONV_3_OUTPUT_0_MEM_WRAP_SIZE,
                           _13_PADCONV_3_OUTPUT_0_MEM_STRIDE>
                           (_12_Mul_3_output_0, _13_PadConv_3_output_0, _13_PadConv_3_weights_output_0, _13_PadConv_3_biases_output_0, _13_PADCONV_3_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _13_PadConv_3\n");
    FILE* _13_PADCONV_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_13_PadConv_3_output_0.txt", "w");
    export_cpp::saveOutputs<_13_PADCONV_3_OUTPUT_0_NB_OUTPUTS,
                            _13_PADCONV_3_OUTPUT_0_OUT_HEIGHT,
                            _13_PADCONV_3_OUTPUT_0_OUT_WIDTH,
                            _13_PADCONV_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _13_PADCONV_3_OUTPUT_0_MEM_CONT_SIZE,
                            _13_PADCONV_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _13_PADCONV_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _13_PADCONV_3_OUTPUT_0_DEV_FMT>
                            (_13_PadConv_3_output_0,
                            _13_PADCONV_3_OUTPUT_0_STREAM);
    fclose(_13_PADCONV_3_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _14_Sigmoid_4_output_0 = (float*) (mem_0 + _14_SIGMOID_4_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_14_SIGMOID_4_NB_ELTS>
                               (_13_PadConv_3_output_0, _14_Sigmoid_4_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _14_Sigmoid_4\n");
    FILE* _14_SIGMOID_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_14_Sigmoid_4_output_0.txt", "w");
    export_cpp::saveOutputs<_14_SIGMOID_4_OUTPUT_0_NB_OUTPUTS,
                            _14_SIGMOID_4_OUTPUT_0_OUT_HEIGHT,
                            _14_SIGMOID_4_OUTPUT_0_OUT_WIDTH,
                            _14_SIGMOID_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _14_SIGMOID_4_OUTPUT_0_MEM_CONT_SIZE,
                            _14_SIGMOID_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _14_SIGMOID_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _14_SIGMOID_4_OUTPUT_0_DEV_FMT>
                            (_14_Sigmoid_4_output_0,
                            _14_SIGMOID_4_OUTPUT_0_STREAM);
    fclose(_14_SIGMOID_4_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _15_Mul_4_output_0 = (float*) (mem_0 + _15_MUL_4_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_15_MUL_4_NB_MAT,
                                 _15_MUL_4_ELEM_OP,
                                 _15_MUL_4_INPUT1_CONT_SIZE,
                                 _15_MUL_4_INPUT2_CONT_SIZE,
                                 _15_MUL_4_OUTPUT_CONT_SIZE,
                                 _15_MUL_4_OFFSET_IN1,
                                 _15_MUL_4_OFFSET_IN2,
                                 _15_MUL_4_ACTIVATION,
                                 _13_PADCONV_3_OUTPUT_0_MEM_CONT_OFFSET,
                                 _13_PADCONV_3_OUTPUT_0_MEM_CONT_SIZE,
                                 _13_PADCONV_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _13_PADCONV_3_OUTPUT_0_MEM_WRAP_SIZE,
                                 _13_PADCONV_3_OUTPUT_0_MEM_STRIDE,
                                 _14_SIGMOID_4_OUTPUT_0_MEM_CONT_OFFSET,
                                 _14_SIGMOID_4_OUTPUT_0_MEM_CONT_SIZE,
                                 _14_SIGMOID_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _14_SIGMOID_4_OUTPUT_0_MEM_WRAP_SIZE,
                                 _14_SIGMOID_4_OUTPUT_0_MEM_STRIDE,
                                 _15_MUL_4_OUTPUT_0_MEM_CONT_OFFSET,
                                 _15_MUL_4_OUTPUT_0_MEM_CONT_SIZE,
                                 _15_MUL_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _15_MUL_4_OUTPUT_0_MEM_WRAP_SIZE,
                                 _15_MUL_4_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_15_Mul_4_output_0, _15_MUL_4_RESCALING, _13_PadConv_3_output_0, _14_Sigmoid_4_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _15_Mul_4\n");
    FILE* _15_MUL_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_15_Mul_4_output_0.txt", "w");
    export_cpp::saveOutputs<_15_MUL_4_OUTPUT_0_NB_OUTPUTS,
                            _15_MUL_4_OUTPUT_0_OUT_HEIGHT,
                            _15_MUL_4_OUTPUT_0_OUT_WIDTH,
                            _15_MUL_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _15_MUL_4_OUTPUT_0_MEM_CONT_SIZE,
                            _15_MUL_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _15_MUL_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _15_MUL_4_OUTPUT_0_DEV_FMT>
                            (_15_Mul_4_output_0,
                            _15_MUL_4_OUTPUT_0_STREAM);
    fclose(_15_MUL_4_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _16_Add_0_output_0 = (float*) (mem_0 + _16_ADD_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_16_ADD_0_NB_MAT,
                                 _16_ADD_0_ELEM_OP,
                                 _16_ADD_0_INPUT1_CONT_SIZE,
                                 _16_ADD_0_INPUT2_CONT_SIZE,
                                 _16_ADD_0_OUTPUT_CONT_SIZE,
                                 _16_ADD_0_OFFSET_IN1,
                                 _16_ADD_0_OFFSET_IN2,
                                 _16_ADD_0_ACTIVATION,
                                 _9_SPLIT_0_OUTPUT_1_MEM_CONT_OFFSET,
                                 _9_SPLIT_0_OUTPUT_1_MEM_CONT_SIZE,
                                 _9_SPLIT_0_OUTPUT_1_MEM_WRAP_OFFSET,
                                 _9_SPLIT_0_OUTPUT_1_MEM_WRAP_SIZE,
                                 _9_SPLIT_0_OUTPUT_1_MEM_STRIDE,
                                 _15_MUL_4_OUTPUT_0_MEM_CONT_OFFSET,
                                 _15_MUL_4_OUTPUT_0_MEM_CONT_SIZE,
                                 _15_MUL_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _15_MUL_4_OUTPUT_0_MEM_WRAP_SIZE,
                                 _15_MUL_4_OUTPUT_0_MEM_STRIDE,
                                 _16_ADD_0_OUTPUT_0_MEM_CONT_OFFSET,
                                 _16_ADD_0_OUTPUT_0_MEM_CONT_SIZE,
                                 _16_ADD_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _16_ADD_0_OUTPUT_0_MEM_WRAP_SIZE,
                                 _16_ADD_0_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_16_Add_0_output_0, _16_ADD_0_RESCALING, _9_Split_0_output_1, _15_Mul_4_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _16_Add_0\n");
    FILE* _16_ADD_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_16_Add_0_output_0.txt", "w");
    export_cpp::saveOutputs<_16_ADD_0_OUTPUT_0_NB_OUTPUTS,
                            _16_ADD_0_OUTPUT_0_OUT_HEIGHT,
                            _16_ADD_0_OUTPUT_0_OUT_WIDTH,
                            _16_ADD_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _16_ADD_0_OUTPUT_0_MEM_CONT_SIZE,
                            _16_ADD_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _16_ADD_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _16_ADD_0_OUTPUT_0_DEV_FMT>
                            (_16_Add_0_output_0,
                            _16_ADD_0_OUTPUT_0_STREAM);
    fclose(_16_ADD_0_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _17_Concat_0_output_0 = (float*) (mem_0 + _17_CONCAT_0_OUTPUT_0_MEM_OFFSET);

    const float* _17_Concat_0_inputs[] = { _9_Split_0_output_0, _9_Split_0_output_1, _16_Add_0_output_0 };
    export_cpp::concat_forward<_17_CONCAT_0_AXIS_SIZE_POST,
                               _17_CONCAT_0_AXIS_SIZE_PRE,
                               _17_CONCAT_0_AXIS_SIZE,
                               _17_CONCAT_0_TOTAL_AXIS_SIZE,
                               3> 
                              (_17_Concat_0_inputs,
                               _17_Concat_0_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _17_Concat_0\n");
    FILE* _17_CONCAT_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_17_Concat_0_output_0.txt", "w");
    export_cpp::saveOutputs<_17_CONCAT_0_OUTPUT_0_NB_OUTPUTS,
                            _17_CONCAT_0_OUTPUT_0_OUT_HEIGHT,
                            _17_CONCAT_0_OUTPUT_0_OUT_WIDTH,
                            _17_CONCAT_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _17_CONCAT_0_OUTPUT_0_MEM_CONT_SIZE,
                            _17_CONCAT_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _17_CONCAT_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _17_CONCAT_0_OUTPUT_0_DEV_FMT>
                            (_17_Concat_0_output_0,
                            _17_CONCAT_0_OUTPUT_0_STREAM);
    fclose(_17_CONCAT_0_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _18_Conv2D_1_output_0 = (float*) (mem_0 + _18_CONV2D_1_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_17_CONCAT_0_OUTPUT_0_IN_BATCH,
                           _18_CONV2D_1_NB_CHANNELS,
                           _17_CONCAT_0_OUTPUT_0_IN_HEIGHT,
                           _17_CONCAT_0_OUTPUT_0_IN_WIDTH,
                           _18_CONV2D_1_NB_OUTPUTS,
                           _18_CONV2D_1_OUTPUT_0_OUT_HEIGHT,
                           _18_CONV2D_1_OUTPUT_0_OUT_WIDTH,
                           _18_CONV2D_1_PADDING_Y,
                           _18_CONV2D_1_PADDING_X,
                           _18_CONV2D_1_STRIDE_Y,
                           _18_CONV2D_1_STRIDE_X,
                           _18_CONV2D_1_DILATION_Y,
                           _18_CONV2D_1_DILATION_X,
                           _18_CONV2D_1_KERNEL_HEIGHT,
                           _18_CONV2D_1_KERNEL_WIDTH,
                           _18_CONV2D_1_ACTIVATION,
                           _17_CONCAT_0_OUTPUT_0_MEM_CONT_OFFSET,
                           _17_CONCAT_0_OUTPUT_0_MEM_CONT_SIZE,
                           _17_CONCAT_0_OUTPUT_0_MEM_WRAP_OFFSET,
                           _17_CONCAT_0_OUTPUT_0_MEM_WRAP_SIZE,
                           _17_CONCAT_0_OUTPUT_0_MEM_STRIDE,
                           _18_CONV2D_1_OUTPUT_0_MEM_CONT_OFFSET,
                           _18_CONV2D_1_OUTPUT_0_MEM_CONT_SIZE,
                           _18_CONV2D_1_OUTPUT_0_MEM_WRAP_OFFSET,
                           _18_CONV2D_1_OUTPUT_0_MEM_WRAP_SIZE,
                           _18_CONV2D_1_OUTPUT_0_MEM_STRIDE>
                           (_17_Concat_0_output_0, _18_Conv2D_1_output_0, _18_Conv2D_1_weights_output_0, _18_Conv2D_1_biases_output_0, _18_CONV2D_1_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _18_Conv2D_1\n");
    FILE* _18_CONV2D_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_18_Conv2D_1_output_0.txt", "w");
    export_cpp::saveOutputs<_18_CONV2D_1_OUTPUT_0_NB_OUTPUTS,
                            _18_CONV2D_1_OUTPUT_0_OUT_HEIGHT,
                            _18_CONV2D_1_OUTPUT_0_OUT_WIDTH,
                            _18_CONV2D_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _18_CONV2D_1_OUTPUT_0_MEM_CONT_SIZE,
                            _18_CONV2D_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _18_CONV2D_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _18_CONV2D_1_OUTPUT_0_DEV_FMT>
                            (_18_Conv2D_1_output_0,
                            _18_CONV2D_1_OUTPUT_0_STREAM);
    fclose(_18_CONV2D_1_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _19_Sigmoid_5_output_0 = (float*) (mem_0 + _19_SIGMOID_5_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_19_SIGMOID_5_NB_ELTS>
                               (_18_Conv2D_1_output_0, _19_Sigmoid_5_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _19_Sigmoid_5\n");
    FILE* _19_SIGMOID_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_19_Sigmoid_5_output_0.txt", "w");
    export_cpp::saveOutputs<_19_SIGMOID_5_OUTPUT_0_NB_OUTPUTS,
                            _19_SIGMOID_5_OUTPUT_0_OUT_HEIGHT,
                            _19_SIGMOID_5_OUTPUT_0_OUT_WIDTH,
                            _19_SIGMOID_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _19_SIGMOID_5_OUTPUT_0_MEM_CONT_SIZE,
                            _19_SIGMOID_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _19_SIGMOID_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _19_SIGMOID_5_OUTPUT_0_DEV_FMT>
                            (_19_Sigmoid_5_output_0,
                            _19_SIGMOID_5_OUTPUT_0_STREAM);
    fclose(_19_SIGMOID_5_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _20_Mul_5_output_0 = (float*) (mem_0 + _20_MUL_5_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_20_MUL_5_NB_MAT,
                                 _20_MUL_5_ELEM_OP,
                                 _20_MUL_5_INPUT1_CONT_SIZE,
                                 _20_MUL_5_INPUT2_CONT_SIZE,
                                 _20_MUL_5_OUTPUT_CONT_SIZE,
                                 _20_MUL_5_OFFSET_IN1,
                                 _20_MUL_5_OFFSET_IN2,
                                 _20_MUL_5_ACTIVATION,
                                 _18_CONV2D_1_OUTPUT_0_MEM_CONT_OFFSET,
                                 _18_CONV2D_1_OUTPUT_0_MEM_CONT_SIZE,
                                 _18_CONV2D_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _18_CONV2D_1_OUTPUT_0_MEM_WRAP_SIZE,
                                 _18_CONV2D_1_OUTPUT_0_MEM_STRIDE,
                                 _19_SIGMOID_5_OUTPUT_0_MEM_CONT_OFFSET,
                                 _19_SIGMOID_5_OUTPUT_0_MEM_CONT_SIZE,
                                 _19_SIGMOID_5_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _19_SIGMOID_5_OUTPUT_0_MEM_WRAP_SIZE,
                                 _19_SIGMOID_5_OUTPUT_0_MEM_STRIDE,
                                 _20_MUL_5_OUTPUT_0_MEM_CONT_OFFSET,
                                 _20_MUL_5_OUTPUT_0_MEM_CONT_SIZE,
                                 _20_MUL_5_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _20_MUL_5_OUTPUT_0_MEM_WRAP_SIZE,
                                 _20_MUL_5_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_20_Mul_5_output_0, _20_MUL_5_RESCALING, _18_Conv2D_1_output_0, _19_Sigmoid_5_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _20_Mul_5\n");
    FILE* _20_MUL_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_20_Mul_5_output_0.txt", "w");
    export_cpp::saveOutputs<_20_MUL_5_OUTPUT_0_NB_OUTPUTS,
                            _20_MUL_5_OUTPUT_0_OUT_HEIGHT,
                            _20_MUL_5_OUTPUT_0_OUT_WIDTH,
                            _20_MUL_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _20_MUL_5_OUTPUT_0_MEM_CONT_SIZE,
                            _20_MUL_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _20_MUL_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _20_MUL_5_OUTPUT_0_DEV_FMT>
                            (_20_Mul_5_output_0,
                            _20_MUL_5_OUTPUT_0_STREAM);
    fclose(_20_MUL_5_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _21_PadConv_4_output_0 = (float*) (mem_0 + _21_PADCONV_4_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_20_MUL_5_OUTPUT_0_IN_BATCH,
                           _21_PADCONV_4_NB_CHANNELS,
                           _20_MUL_5_OUTPUT_0_IN_HEIGHT,
                           _20_MUL_5_OUTPUT_0_IN_WIDTH,
                           _21_PADCONV_4_NB_OUTPUTS,
                           _21_PADCONV_4_OUTPUT_0_OUT_HEIGHT,
                           _21_PADCONV_4_OUTPUT_0_OUT_WIDTH,
                           _21_PADCONV_4_PADDING_Y,
                           _21_PADCONV_4_PADDING_X,
                           _21_PADCONV_4_STRIDE_Y,
                           _21_PADCONV_4_STRIDE_X,
                           _21_PADCONV_4_DILATION_Y,
                           _21_PADCONV_4_DILATION_X,
                           _21_PADCONV_4_KERNEL_HEIGHT,
                           _21_PADCONV_4_KERNEL_WIDTH,
                           _21_PADCONV_4_ACTIVATION,
                           _20_MUL_5_OUTPUT_0_MEM_CONT_OFFSET,
                           _20_MUL_5_OUTPUT_0_MEM_CONT_SIZE,
                           _20_MUL_5_OUTPUT_0_MEM_WRAP_OFFSET,
                           _20_MUL_5_OUTPUT_0_MEM_WRAP_SIZE,
                           _20_MUL_5_OUTPUT_0_MEM_STRIDE,
                           _21_PADCONV_4_OUTPUT_0_MEM_CONT_OFFSET,
                           _21_PADCONV_4_OUTPUT_0_MEM_CONT_SIZE,
                           _21_PADCONV_4_OUTPUT_0_MEM_WRAP_OFFSET,
                           _21_PADCONV_4_OUTPUT_0_MEM_WRAP_SIZE,
                           _21_PADCONV_4_OUTPUT_0_MEM_STRIDE>
                           (_20_Mul_5_output_0, _21_PadConv_4_output_0, _21_PadConv_4_weights_output_0, _21_PadConv_4_biases_output_0, _21_PADCONV_4_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _21_PadConv_4\n");
    FILE* _21_PADCONV_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_21_PadConv_4_output_0.txt", "w");
    export_cpp::saveOutputs<_21_PADCONV_4_OUTPUT_0_NB_OUTPUTS,
                            _21_PADCONV_4_OUTPUT_0_OUT_HEIGHT,
                            _21_PADCONV_4_OUTPUT_0_OUT_WIDTH,
                            _21_PADCONV_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _21_PADCONV_4_OUTPUT_0_MEM_CONT_SIZE,
                            _21_PADCONV_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _21_PADCONV_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _21_PADCONV_4_OUTPUT_0_DEV_FMT>
                            (_21_PadConv_4_output_0,
                            _21_PADCONV_4_OUTPUT_0_STREAM);
    fclose(_21_PADCONV_4_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _22_Sigmoid_6_output_0 = (float*) (mem_0 + _22_SIGMOID_6_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_22_SIGMOID_6_NB_ELTS>
                               (_21_PadConv_4_output_0, _22_Sigmoid_6_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _22_Sigmoid_6\n");
    FILE* _22_SIGMOID_6_OUTPUT_0_STREAM = fopen("data/export_outputs/_22_Sigmoid_6_output_0.txt", "w");
    export_cpp::saveOutputs<_22_SIGMOID_6_OUTPUT_0_NB_OUTPUTS,
                            _22_SIGMOID_6_OUTPUT_0_OUT_HEIGHT,
                            _22_SIGMOID_6_OUTPUT_0_OUT_WIDTH,
                            _22_SIGMOID_6_OUTPUT_0_MEM_CONT_OFFSET,
                            _22_SIGMOID_6_OUTPUT_0_MEM_CONT_SIZE,
                            _22_SIGMOID_6_OUTPUT_0_MEM_WRAP_OFFSET,
                            _22_SIGMOID_6_OUTPUT_0_MEM_WRAP_SIZE,
                            _22_SIGMOID_6_OUTPUT_0_DEV_FMT>
                            (_22_Sigmoid_6_output_0,
                            _22_SIGMOID_6_OUTPUT_0_STREAM);
    fclose(_22_SIGMOID_6_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _23_Mul_6_output_0 = (float*) (mem_0 + _23_MUL_6_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_23_MUL_6_NB_MAT,
                                 _23_MUL_6_ELEM_OP,
                                 _23_MUL_6_INPUT1_CONT_SIZE,
                                 _23_MUL_6_INPUT2_CONT_SIZE,
                                 _23_MUL_6_OUTPUT_CONT_SIZE,
                                 _23_MUL_6_OFFSET_IN1,
                                 _23_MUL_6_OFFSET_IN2,
                                 _23_MUL_6_ACTIVATION,
                                 _21_PADCONV_4_OUTPUT_0_MEM_CONT_OFFSET,
                                 _21_PADCONV_4_OUTPUT_0_MEM_CONT_SIZE,
                                 _21_PADCONV_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _21_PADCONV_4_OUTPUT_0_MEM_WRAP_SIZE,
                                 _21_PADCONV_4_OUTPUT_0_MEM_STRIDE,
                                 _22_SIGMOID_6_OUTPUT_0_MEM_CONT_OFFSET,
                                 _22_SIGMOID_6_OUTPUT_0_MEM_CONT_SIZE,
                                 _22_SIGMOID_6_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _22_SIGMOID_6_OUTPUT_0_MEM_WRAP_SIZE,
                                 _22_SIGMOID_6_OUTPUT_0_MEM_STRIDE,
                                 _23_MUL_6_OUTPUT_0_MEM_CONT_OFFSET,
                                 _23_MUL_6_OUTPUT_0_MEM_CONT_SIZE,
                                 _23_MUL_6_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _23_MUL_6_OUTPUT_0_MEM_WRAP_SIZE,
                                 _23_MUL_6_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_23_Mul_6_output_0, _23_MUL_6_RESCALING, _21_PadConv_4_output_0, _22_Sigmoid_6_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _23_Mul_6\n");
    FILE* _23_MUL_6_OUTPUT_0_STREAM = fopen("data/export_outputs/_23_Mul_6_output_0.txt", "w");
    export_cpp::saveOutputs<_23_MUL_6_OUTPUT_0_NB_OUTPUTS,
                            _23_MUL_6_OUTPUT_0_OUT_HEIGHT,
                            _23_MUL_6_OUTPUT_0_OUT_WIDTH,
                            _23_MUL_6_OUTPUT_0_MEM_CONT_OFFSET,
                            _23_MUL_6_OUTPUT_0_MEM_CONT_SIZE,
                            _23_MUL_6_OUTPUT_0_MEM_WRAP_OFFSET,
                            _23_MUL_6_OUTPUT_0_MEM_WRAP_SIZE,
                            _23_MUL_6_OUTPUT_0_DEV_FMT>
                            (_23_Mul_6_output_0,
                            _23_MUL_6_OUTPUT_0_STREAM);
    fclose(_23_MUL_6_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _24_Conv2D_2_output_0 = (float*) (mem_0 + _24_CONV2D_2_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_23_MUL_6_OUTPUT_0_IN_BATCH,
                           _24_CONV2D_2_NB_CHANNELS,
                           _23_MUL_6_OUTPUT_0_IN_HEIGHT,
                           _23_MUL_6_OUTPUT_0_IN_WIDTH,
                           _24_CONV2D_2_NB_OUTPUTS,
                           _24_CONV2D_2_OUTPUT_0_OUT_HEIGHT,
                           _24_CONV2D_2_OUTPUT_0_OUT_WIDTH,
                           _24_CONV2D_2_PADDING_Y,
                           _24_CONV2D_2_PADDING_X,
                           _24_CONV2D_2_STRIDE_Y,
                           _24_CONV2D_2_STRIDE_X,
                           _24_CONV2D_2_DILATION_Y,
                           _24_CONV2D_2_DILATION_X,
                           _24_CONV2D_2_KERNEL_HEIGHT,
                           _24_CONV2D_2_KERNEL_WIDTH,
                           _24_CONV2D_2_ACTIVATION,
                           _23_MUL_6_OUTPUT_0_MEM_CONT_OFFSET,
                           _23_MUL_6_OUTPUT_0_MEM_CONT_SIZE,
                           _23_MUL_6_OUTPUT_0_MEM_WRAP_OFFSET,
                           _23_MUL_6_OUTPUT_0_MEM_WRAP_SIZE,
                           _23_MUL_6_OUTPUT_0_MEM_STRIDE,
                           _24_CONV2D_2_OUTPUT_0_MEM_CONT_OFFSET,
                           _24_CONV2D_2_OUTPUT_0_MEM_CONT_SIZE,
                           _24_CONV2D_2_OUTPUT_0_MEM_WRAP_OFFSET,
                           _24_CONV2D_2_OUTPUT_0_MEM_WRAP_SIZE,
                           _24_CONV2D_2_OUTPUT_0_MEM_STRIDE>
                           (_23_Mul_6_output_0, _24_Conv2D_2_output_0, _24_Conv2D_2_weights_output_0, _24_Conv2D_2_biases_output_0, _24_CONV2D_2_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _24_Conv2D_2\n");
    FILE* _24_CONV2D_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_24_Conv2D_2_output_0.txt", "w");
    export_cpp::saveOutputs<_24_CONV2D_2_OUTPUT_0_NB_OUTPUTS,
                            _24_CONV2D_2_OUTPUT_0_OUT_HEIGHT,
                            _24_CONV2D_2_OUTPUT_0_OUT_WIDTH,
                            _24_CONV2D_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _24_CONV2D_2_OUTPUT_0_MEM_CONT_SIZE,
                            _24_CONV2D_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _24_CONV2D_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _24_CONV2D_2_OUTPUT_0_DEV_FMT>
                            (_24_Conv2D_2_output_0,
                            _24_CONV2D_2_OUTPUT_0_STREAM);
    fclose(_24_CONV2D_2_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _25_Sigmoid_7_output_0 = (float*) (mem_0 + _25_SIGMOID_7_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_25_SIGMOID_7_NB_ELTS>
                               (_24_Conv2D_2_output_0, _25_Sigmoid_7_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _25_Sigmoid_7\n");
    FILE* _25_SIGMOID_7_OUTPUT_0_STREAM = fopen("data/export_outputs/_25_Sigmoid_7_output_0.txt", "w");
    export_cpp::saveOutputs<_25_SIGMOID_7_OUTPUT_0_NB_OUTPUTS,
                            _25_SIGMOID_7_OUTPUT_0_OUT_HEIGHT,
                            _25_SIGMOID_7_OUTPUT_0_OUT_WIDTH,
                            _25_SIGMOID_7_OUTPUT_0_MEM_CONT_OFFSET,
                            _25_SIGMOID_7_OUTPUT_0_MEM_CONT_SIZE,
                            _25_SIGMOID_7_OUTPUT_0_MEM_WRAP_OFFSET,
                            _25_SIGMOID_7_OUTPUT_0_MEM_WRAP_SIZE,
                            _25_SIGMOID_7_OUTPUT_0_DEV_FMT>
                            (_25_Sigmoid_7_output_0,
                            _25_SIGMOID_7_OUTPUT_0_STREAM);
    fclose(_25_SIGMOID_7_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _26_Mul_7_output_0 = (float*) (mem_0 + _26_MUL_7_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_26_MUL_7_NB_MAT,
                                 _26_MUL_7_ELEM_OP,
                                 _26_MUL_7_INPUT1_CONT_SIZE,
                                 _26_MUL_7_INPUT2_CONT_SIZE,
                                 _26_MUL_7_OUTPUT_CONT_SIZE,
                                 _26_MUL_7_OFFSET_IN1,
                                 _26_MUL_7_OFFSET_IN2,
                                 _26_MUL_7_ACTIVATION,
                                 _24_CONV2D_2_OUTPUT_0_MEM_CONT_OFFSET,
                                 _24_CONV2D_2_OUTPUT_0_MEM_CONT_SIZE,
                                 _24_CONV2D_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _24_CONV2D_2_OUTPUT_0_MEM_WRAP_SIZE,
                                 _24_CONV2D_2_OUTPUT_0_MEM_STRIDE,
                                 _25_SIGMOID_7_OUTPUT_0_MEM_CONT_OFFSET,
                                 _25_SIGMOID_7_OUTPUT_0_MEM_CONT_SIZE,
                                 _25_SIGMOID_7_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _25_SIGMOID_7_OUTPUT_0_MEM_WRAP_SIZE,
                                 _25_SIGMOID_7_OUTPUT_0_MEM_STRIDE,
                                 _26_MUL_7_OUTPUT_0_MEM_CONT_OFFSET,
                                 _26_MUL_7_OUTPUT_0_MEM_CONT_SIZE,
                                 _26_MUL_7_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _26_MUL_7_OUTPUT_0_MEM_WRAP_SIZE,
                                 _26_MUL_7_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_26_Mul_7_output_0, _26_MUL_7_RESCALING, _24_Conv2D_2_output_0, _25_Sigmoid_7_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _26_Mul_7\n");
    FILE* _26_MUL_7_OUTPUT_0_STREAM = fopen("data/export_outputs/_26_Mul_7_output_0.txt", "w");
    export_cpp::saveOutputs<_26_MUL_7_OUTPUT_0_NB_OUTPUTS,
                            _26_MUL_7_OUTPUT_0_OUT_HEIGHT,
                            _26_MUL_7_OUTPUT_0_OUT_WIDTH,
                            _26_MUL_7_OUTPUT_0_MEM_CONT_OFFSET,
                            _26_MUL_7_OUTPUT_0_MEM_CONT_SIZE,
                            _26_MUL_7_OUTPUT_0_MEM_WRAP_OFFSET,
                            _26_MUL_7_OUTPUT_0_MEM_WRAP_SIZE,
                            _26_MUL_7_OUTPUT_0_DEV_FMT>
                            (_26_Mul_7_output_0,
                            _26_MUL_7_OUTPUT_0_STREAM);
    fclose(_26_MUL_7_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _27_Split_1_output_0 = (float*) (mem_0 + _27_SPLIT_1_OUTPUT_0_MEM_OFFSET);
    float* _27_Split_1_output_1 = (float*) (mem_0 + _27_SPLIT_1_OUTPUT_1_MEM_OFFSET);



    float* __27_Split_1_outputs[_27_SPLIT_1_NB_OUTPUTS] = {
        
        _27_Split_1_output_0,
        
        _27_Split_1_output_1
        
    };

    export_cpp::split_forward<_27_SPLIT_1_NB_OUTPUTS,
                              _27_SPLIT_1_AXIS_DIM,
                              _27_SPLIT_1_PRE_AXIS_ELEMS,
                              _27_SPLIT_1_POST_AXIS_ELEMS,
                              _27_SPLIT_1_SPLITS,
                              float>
                              (_26_Mul_7_output_0,
                               __27_Split_1_outputs);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _27_Split_1\n");
    FILE* _27_SPLIT_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_27_Split_1_output_0.txt", "w");
    export_cpp::saveOutputs<_27_SPLIT_1_OUTPUT_0_NB_OUTPUTS,
                            _27_SPLIT_1_OUTPUT_0_OUT_HEIGHT,
                            _27_SPLIT_1_OUTPUT_0_OUT_WIDTH,
                            _27_SPLIT_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _27_SPLIT_1_OUTPUT_0_MEM_CONT_SIZE,
                            _27_SPLIT_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _27_SPLIT_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _27_SPLIT_1_OUTPUT_0_DEV_FMT>
                            (_27_Split_1_output_0,
                            _27_SPLIT_1_OUTPUT_0_STREAM);
    fclose(_27_SPLIT_1_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _28_PadConv_5_output_0 = (float*) (mem_0 + _28_PADCONV_5_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_27_SPLIT_1_OUTPUT_1_IN_BATCH,
                           _28_PADCONV_5_NB_CHANNELS,
                           _27_SPLIT_1_OUTPUT_1_IN_HEIGHT,
                           _27_SPLIT_1_OUTPUT_1_IN_WIDTH,
                           _28_PADCONV_5_NB_OUTPUTS,
                           _28_PADCONV_5_OUTPUT_0_OUT_HEIGHT,
                           _28_PADCONV_5_OUTPUT_0_OUT_WIDTH,
                           _28_PADCONV_5_PADDING_Y,
                           _28_PADCONV_5_PADDING_X,
                           _28_PADCONV_5_STRIDE_Y,
                           _28_PADCONV_5_STRIDE_X,
                           _28_PADCONV_5_DILATION_Y,
                           _28_PADCONV_5_DILATION_X,
                           _28_PADCONV_5_KERNEL_HEIGHT,
                           _28_PADCONV_5_KERNEL_WIDTH,
                           _28_PADCONV_5_ACTIVATION,
                           _27_SPLIT_1_OUTPUT_1_MEM_CONT_OFFSET,
                           _27_SPLIT_1_OUTPUT_1_MEM_CONT_SIZE,
                           _27_SPLIT_1_OUTPUT_1_MEM_WRAP_OFFSET,
                           _27_SPLIT_1_OUTPUT_1_MEM_WRAP_SIZE,
                           _27_SPLIT_1_OUTPUT_1_MEM_STRIDE,
                           _28_PADCONV_5_OUTPUT_0_MEM_CONT_OFFSET,
                           _28_PADCONV_5_OUTPUT_0_MEM_CONT_SIZE,
                           _28_PADCONV_5_OUTPUT_0_MEM_WRAP_OFFSET,
                           _28_PADCONV_5_OUTPUT_0_MEM_WRAP_SIZE,
                           _28_PADCONV_5_OUTPUT_0_MEM_STRIDE>
                           (_27_Split_1_output_1, _28_PadConv_5_output_0, _28_PadConv_5_weights_output_0, _28_PadConv_5_biases_output_0, _28_PADCONV_5_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _28_PadConv_5\n");
    FILE* _28_PADCONV_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_28_PadConv_5_output_0.txt", "w");
    export_cpp::saveOutputs<_28_PADCONV_5_OUTPUT_0_NB_OUTPUTS,
                            _28_PADCONV_5_OUTPUT_0_OUT_HEIGHT,
                            _28_PADCONV_5_OUTPUT_0_OUT_WIDTH,
                            _28_PADCONV_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _28_PADCONV_5_OUTPUT_0_MEM_CONT_SIZE,
                            _28_PADCONV_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _28_PADCONV_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _28_PADCONV_5_OUTPUT_0_DEV_FMT>
                            (_28_PadConv_5_output_0,
                            _28_PADCONV_5_OUTPUT_0_STREAM);
    fclose(_28_PADCONV_5_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _29_Sigmoid_8_output_0 = (float*) (mem_0 + _29_SIGMOID_8_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_29_SIGMOID_8_NB_ELTS>
                               (_28_PadConv_5_output_0, _29_Sigmoid_8_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _29_Sigmoid_8\n");
    FILE* _29_SIGMOID_8_OUTPUT_0_STREAM = fopen("data/export_outputs/_29_Sigmoid_8_output_0.txt", "w");
    export_cpp::saveOutputs<_29_SIGMOID_8_OUTPUT_0_NB_OUTPUTS,
                            _29_SIGMOID_8_OUTPUT_0_OUT_HEIGHT,
                            _29_SIGMOID_8_OUTPUT_0_OUT_WIDTH,
                            _29_SIGMOID_8_OUTPUT_0_MEM_CONT_OFFSET,
                            _29_SIGMOID_8_OUTPUT_0_MEM_CONT_SIZE,
                            _29_SIGMOID_8_OUTPUT_0_MEM_WRAP_OFFSET,
                            _29_SIGMOID_8_OUTPUT_0_MEM_WRAP_SIZE,
                            _29_SIGMOID_8_OUTPUT_0_DEV_FMT>
                            (_29_Sigmoid_8_output_0,
                            _29_SIGMOID_8_OUTPUT_0_STREAM);
    fclose(_29_SIGMOID_8_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _30_Mul_8_output_0 = (float*) (mem_0 + _30_MUL_8_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_30_MUL_8_NB_MAT,
                                 _30_MUL_8_ELEM_OP,
                                 _30_MUL_8_INPUT1_CONT_SIZE,
                                 _30_MUL_8_INPUT2_CONT_SIZE,
                                 _30_MUL_8_OUTPUT_CONT_SIZE,
                                 _30_MUL_8_OFFSET_IN1,
                                 _30_MUL_8_OFFSET_IN2,
                                 _30_MUL_8_ACTIVATION,
                                 _28_PADCONV_5_OUTPUT_0_MEM_CONT_OFFSET,
                                 _28_PADCONV_5_OUTPUT_0_MEM_CONT_SIZE,
                                 _28_PADCONV_5_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _28_PADCONV_5_OUTPUT_0_MEM_WRAP_SIZE,
                                 _28_PADCONV_5_OUTPUT_0_MEM_STRIDE,
                                 _29_SIGMOID_8_OUTPUT_0_MEM_CONT_OFFSET,
                                 _29_SIGMOID_8_OUTPUT_0_MEM_CONT_SIZE,
                                 _29_SIGMOID_8_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _29_SIGMOID_8_OUTPUT_0_MEM_WRAP_SIZE,
                                 _29_SIGMOID_8_OUTPUT_0_MEM_STRIDE,
                                 _30_MUL_8_OUTPUT_0_MEM_CONT_OFFSET,
                                 _30_MUL_8_OUTPUT_0_MEM_CONT_SIZE,
                                 _30_MUL_8_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _30_MUL_8_OUTPUT_0_MEM_WRAP_SIZE,
                                 _30_MUL_8_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_30_Mul_8_output_0, _30_MUL_8_RESCALING, _28_PadConv_5_output_0, _29_Sigmoid_8_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _30_Mul_8\n");
    FILE* _30_MUL_8_OUTPUT_0_STREAM = fopen("data/export_outputs/_30_Mul_8_output_0.txt", "w");
    export_cpp::saveOutputs<_30_MUL_8_OUTPUT_0_NB_OUTPUTS,
                            _30_MUL_8_OUTPUT_0_OUT_HEIGHT,
                            _30_MUL_8_OUTPUT_0_OUT_WIDTH,
                            _30_MUL_8_OUTPUT_0_MEM_CONT_OFFSET,
                            _30_MUL_8_OUTPUT_0_MEM_CONT_SIZE,
                            _30_MUL_8_OUTPUT_0_MEM_WRAP_OFFSET,
                            _30_MUL_8_OUTPUT_0_MEM_WRAP_SIZE,
                            _30_MUL_8_OUTPUT_0_DEV_FMT>
                            (_30_Mul_8_output_0,
                            _30_MUL_8_OUTPUT_0_STREAM);
    fclose(_30_MUL_8_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _31_PadConv_6_output_0 = (float*) (mem_0 + _31_PADCONV_6_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_30_MUL_8_OUTPUT_0_IN_BATCH,
                           _31_PADCONV_6_NB_CHANNELS,
                           _30_MUL_8_OUTPUT_0_IN_HEIGHT,
                           _30_MUL_8_OUTPUT_0_IN_WIDTH,
                           _31_PADCONV_6_NB_OUTPUTS,
                           _31_PADCONV_6_OUTPUT_0_OUT_HEIGHT,
                           _31_PADCONV_6_OUTPUT_0_OUT_WIDTH,
                           _31_PADCONV_6_PADDING_Y,
                           _31_PADCONV_6_PADDING_X,
                           _31_PADCONV_6_STRIDE_Y,
                           _31_PADCONV_6_STRIDE_X,
                           _31_PADCONV_6_DILATION_Y,
                           _31_PADCONV_6_DILATION_X,
                           _31_PADCONV_6_KERNEL_HEIGHT,
                           _31_PADCONV_6_KERNEL_WIDTH,
                           _31_PADCONV_6_ACTIVATION,
                           _30_MUL_8_OUTPUT_0_MEM_CONT_OFFSET,
                           _30_MUL_8_OUTPUT_0_MEM_CONT_SIZE,
                           _30_MUL_8_OUTPUT_0_MEM_WRAP_OFFSET,
                           _30_MUL_8_OUTPUT_0_MEM_WRAP_SIZE,
                           _30_MUL_8_OUTPUT_0_MEM_STRIDE,
                           _31_PADCONV_6_OUTPUT_0_MEM_CONT_OFFSET,
                           _31_PADCONV_6_OUTPUT_0_MEM_CONT_SIZE,
                           _31_PADCONV_6_OUTPUT_0_MEM_WRAP_OFFSET,
                           _31_PADCONV_6_OUTPUT_0_MEM_WRAP_SIZE,
                           _31_PADCONV_6_OUTPUT_0_MEM_STRIDE>
                           (_30_Mul_8_output_0, _31_PadConv_6_output_0, _31_PadConv_6_weights_output_0, _31_PadConv_6_biases_output_0, _31_PADCONV_6_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _31_PadConv_6\n");
    FILE* _31_PADCONV_6_OUTPUT_0_STREAM = fopen("data/export_outputs/_31_PadConv_6_output_0.txt", "w");
    export_cpp::saveOutputs<_31_PADCONV_6_OUTPUT_0_NB_OUTPUTS,
                            _31_PADCONV_6_OUTPUT_0_OUT_HEIGHT,
                            _31_PADCONV_6_OUTPUT_0_OUT_WIDTH,
                            _31_PADCONV_6_OUTPUT_0_MEM_CONT_OFFSET,
                            _31_PADCONV_6_OUTPUT_0_MEM_CONT_SIZE,
                            _31_PADCONV_6_OUTPUT_0_MEM_WRAP_OFFSET,
                            _31_PADCONV_6_OUTPUT_0_MEM_WRAP_SIZE,
                            _31_PADCONV_6_OUTPUT_0_DEV_FMT>
                            (_31_PadConv_6_output_0,
                            _31_PADCONV_6_OUTPUT_0_STREAM);
    fclose(_31_PADCONV_6_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _32_Sigmoid_9_output_0 = (float*) (mem_0 + _32_SIGMOID_9_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_32_SIGMOID_9_NB_ELTS>
                               (_31_PadConv_6_output_0, _32_Sigmoid_9_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _32_Sigmoid_9\n");
    FILE* _32_SIGMOID_9_OUTPUT_0_STREAM = fopen("data/export_outputs/_32_Sigmoid_9_output_0.txt", "w");
    export_cpp::saveOutputs<_32_SIGMOID_9_OUTPUT_0_NB_OUTPUTS,
                            _32_SIGMOID_9_OUTPUT_0_OUT_HEIGHT,
                            _32_SIGMOID_9_OUTPUT_0_OUT_WIDTH,
                            _32_SIGMOID_9_OUTPUT_0_MEM_CONT_OFFSET,
                            _32_SIGMOID_9_OUTPUT_0_MEM_CONT_SIZE,
                            _32_SIGMOID_9_OUTPUT_0_MEM_WRAP_OFFSET,
                            _32_SIGMOID_9_OUTPUT_0_MEM_WRAP_SIZE,
                            _32_SIGMOID_9_OUTPUT_0_DEV_FMT>
                            (_32_Sigmoid_9_output_0,
                            _32_SIGMOID_9_OUTPUT_0_STREAM);
    fclose(_32_SIGMOID_9_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _33_Mul_9_output_0 = (float*) (mem_0 + _33_MUL_9_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_33_MUL_9_NB_MAT,
                                 _33_MUL_9_ELEM_OP,
                                 _33_MUL_9_INPUT1_CONT_SIZE,
                                 _33_MUL_9_INPUT2_CONT_SIZE,
                                 _33_MUL_9_OUTPUT_CONT_SIZE,
                                 _33_MUL_9_OFFSET_IN1,
                                 _33_MUL_9_OFFSET_IN2,
                                 _33_MUL_9_ACTIVATION,
                                 _31_PADCONV_6_OUTPUT_0_MEM_CONT_OFFSET,
                                 _31_PADCONV_6_OUTPUT_0_MEM_CONT_SIZE,
                                 _31_PADCONV_6_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _31_PADCONV_6_OUTPUT_0_MEM_WRAP_SIZE,
                                 _31_PADCONV_6_OUTPUT_0_MEM_STRIDE,
                                 _32_SIGMOID_9_OUTPUT_0_MEM_CONT_OFFSET,
                                 _32_SIGMOID_9_OUTPUT_0_MEM_CONT_SIZE,
                                 _32_SIGMOID_9_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _32_SIGMOID_9_OUTPUT_0_MEM_WRAP_SIZE,
                                 _32_SIGMOID_9_OUTPUT_0_MEM_STRIDE,
                                 _33_MUL_9_OUTPUT_0_MEM_CONT_OFFSET,
                                 _33_MUL_9_OUTPUT_0_MEM_CONT_SIZE,
                                 _33_MUL_9_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _33_MUL_9_OUTPUT_0_MEM_WRAP_SIZE,
                                 _33_MUL_9_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_33_Mul_9_output_0, _33_MUL_9_RESCALING, _31_PadConv_6_output_0, _32_Sigmoid_9_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _33_Mul_9\n");
    FILE* _33_MUL_9_OUTPUT_0_STREAM = fopen("data/export_outputs/_33_Mul_9_output_0.txt", "w");
    export_cpp::saveOutputs<_33_MUL_9_OUTPUT_0_NB_OUTPUTS,
                            _33_MUL_9_OUTPUT_0_OUT_HEIGHT,
                            _33_MUL_9_OUTPUT_0_OUT_WIDTH,
                            _33_MUL_9_OUTPUT_0_MEM_CONT_OFFSET,
                            _33_MUL_9_OUTPUT_0_MEM_CONT_SIZE,
                            _33_MUL_9_OUTPUT_0_MEM_WRAP_OFFSET,
                            _33_MUL_9_OUTPUT_0_MEM_WRAP_SIZE,
                            _33_MUL_9_OUTPUT_0_DEV_FMT>
                            (_33_Mul_9_output_0,
                            _33_MUL_9_OUTPUT_0_STREAM);
    fclose(_33_MUL_9_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _34_Add_1_output_0 = (float*) (mem_0 + _34_ADD_1_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_34_ADD_1_NB_MAT,
                                 _34_ADD_1_ELEM_OP,
                                 _34_ADD_1_INPUT1_CONT_SIZE,
                                 _34_ADD_1_INPUT2_CONT_SIZE,
                                 _34_ADD_1_OUTPUT_CONT_SIZE,
                                 _34_ADD_1_OFFSET_IN1,
                                 _34_ADD_1_OFFSET_IN2,
                                 _34_ADD_1_ACTIVATION,
                                 _27_SPLIT_1_OUTPUT_1_MEM_CONT_OFFSET,
                                 _27_SPLIT_1_OUTPUT_1_MEM_CONT_SIZE,
                                 _27_SPLIT_1_OUTPUT_1_MEM_WRAP_OFFSET,
                                 _27_SPLIT_1_OUTPUT_1_MEM_WRAP_SIZE,
                                 _27_SPLIT_1_OUTPUT_1_MEM_STRIDE,
                                 _33_MUL_9_OUTPUT_0_MEM_CONT_OFFSET,
                                 _33_MUL_9_OUTPUT_0_MEM_CONT_SIZE,
                                 _33_MUL_9_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _33_MUL_9_OUTPUT_0_MEM_WRAP_SIZE,
                                 _33_MUL_9_OUTPUT_0_MEM_STRIDE,
                                 _34_ADD_1_OUTPUT_0_MEM_CONT_OFFSET,
                                 _34_ADD_1_OUTPUT_0_MEM_CONT_SIZE,
                                 _34_ADD_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _34_ADD_1_OUTPUT_0_MEM_WRAP_SIZE,
                                 _34_ADD_1_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_34_Add_1_output_0, _34_ADD_1_RESCALING, _27_Split_1_output_1, _33_Mul_9_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _34_Add_1\n");
    FILE* _34_ADD_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_34_Add_1_output_0.txt", "w");
    export_cpp::saveOutputs<_34_ADD_1_OUTPUT_0_NB_OUTPUTS,
                            _34_ADD_1_OUTPUT_0_OUT_HEIGHT,
                            _34_ADD_1_OUTPUT_0_OUT_WIDTH,
                            _34_ADD_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _34_ADD_1_OUTPUT_0_MEM_CONT_SIZE,
                            _34_ADD_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _34_ADD_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _34_ADD_1_OUTPUT_0_DEV_FMT>
                            (_34_Add_1_output_0,
                            _34_ADD_1_OUTPUT_0_STREAM);
    fclose(_34_ADD_1_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _35_PadConv_7_output_0 = (float*) (mem_0 + _35_PADCONV_7_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_34_ADD_1_OUTPUT_0_IN_BATCH,
                           _35_PADCONV_7_NB_CHANNELS,
                           _34_ADD_1_OUTPUT_0_IN_HEIGHT,
                           _34_ADD_1_OUTPUT_0_IN_WIDTH,
                           _35_PADCONV_7_NB_OUTPUTS,
                           _35_PADCONV_7_OUTPUT_0_OUT_HEIGHT,
                           _35_PADCONV_7_OUTPUT_0_OUT_WIDTH,
                           _35_PADCONV_7_PADDING_Y,
                           _35_PADCONV_7_PADDING_X,
                           _35_PADCONV_7_STRIDE_Y,
                           _35_PADCONV_7_STRIDE_X,
                           _35_PADCONV_7_DILATION_Y,
                           _35_PADCONV_7_DILATION_X,
                           _35_PADCONV_7_KERNEL_HEIGHT,
                           _35_PADCONV_7_KERNEL_WIDTH,
                           _35_PADCONV_7_ACTIVATION,
                           _34_ADD_1_OUTPUT_0_MEM_CONT_OFFSET,
                           _34_ADD_1_OUTPUT_0_MEM_CONT_SIZE,
                           _34_ADD_1_OUTPUT_0_MEM_WRAP_OFFSET,
                           _34_ADD_1_OUTPUT_0_MEM_WRAP_SIZE,
                           _34_ADD_1_OUTPUT_0_MEM_STRIDE,
                           _35_PADCONV_7_OUTPUT_0_MEM_CONT_OFFSET,
                           _35_PADCONV_7_OUTPUT_0_MEM_CONT_SIZE,
                           _35_PADCONV_7_OUTPUT_0_MEM_WRAP_OFFSET,
                           _35_PADCONV_7_OUTPUT_0_MEM_WRAP_SIZE,
                           _35_PADCONV_7_OUTPUT_0_MEM_STRIDE>
                           (_34_Add_1_output_0, _35_PadConv_7_output_0, _35_PadConv_7_weights_output_0, _35_PadConv_7_biases_output_0, _35_PADCONV_7_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _35_PadConv_7\n");
    FILE* _35_PADCONV_7_OUTPUT_0_STREAM = fopen("data/export_outputs/_35_PadConv_7_output_0.txt", "w");
    export_cpp::saveOutputs<_35_PADCONV_7_OUTPUT_0_NB_OUTPUTS,
                            _35_PADCONV_7_OUTPUT_0_OUT_HEIGHT,
                            _35_PADCONV_7_OUTPUT_0_OUT_WIDTH,
                            _35_PADCONV_7_OUTPUT_0_MEM_CONT_OFFSET,
                            _35_PADCONV_7_OUTPUT_0_MEM_CONT_SIZE,
                            _35_PADCONV_7_OUTPUT_0_MEM_WRAP_OFFSET,
                            _35_PADCONV_7_OUTPUT_0_MEM_WRAP_SIZE,
                            _35_PADCONV_7_OUTPUT_0_DEV_FMT>
                            (_35_PadConv_7_output_0,
                            _35_PADCONV_7_OUTPUT_0_STREAM);
    fclose(_35_PADCONV_7_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _36_Sigmoid_10_output_0 = (float*) (mem_0 + _36_SIGMOID_10_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_36_SIGMOID_10_NB_ELTS>
                               (_35_PadConv_7_output_0, _36_Sigmoid_10_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _36_Sigmoid_10\n");
    FILE* _36_SIGMOID_10_OUTPUT_0_STREAM = fopen("data/export_outputs/_36_Sigmoid_10_output_0.txt", "w");
    export_cpp::saveOutputs<_36_SIGMOID_10_OUTPUT_0_NB_OUTPUTS,
                            _36_SIGMOID_10_OUTPUT_0_OUT_HEIGHT,
                            _36_SIGMOID_10_OUTPUT_0_OUT_WIDTH,
                            _36_SIGMOID_10_OUTPUT_0_MEM_CONT_OFFSET,
                            _36_SIGMOID_10_OUTPUT_0_MEM_CONT_SIZE,
                            _36_SIGMOID_10_OUTPUT_0_MEM_WRAP_OFFSET,
                            _36_SIGMOID_10_OUTPUT_0_MEM_WRAP_SIZE,
                            _36_SIGMOID_10_OUTPUT_0_DEV_FMT>
                            (_36_Sigmoid_10_output_0,
                            _36_SIGMOID_10_OUTPUT_0_STREAM);
    fclose(_36_SIGMOID_10_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _37_Mul_10_output_0 = (float*) (mem_0 + _37_MUL_10_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_37_MUL_10_NB_MAT,
                                 _37_MUL_10_ELEM_OP,
                                 _37_MUL_10_INPUT1_CONT_SIZE,
                                 _37_MUL_10_INPUT2_CONT_SIZE,
                                 _37_MUL_10_OUTPUT_CONT_SIZE,
                                 _37_MUL_10_OFFSET_IN1,
                                 _37_MUL_10_OFFSET_IN2,
                                 _37_MUL_10_ACTIVATION,
                                 _35_PADCONV_7_OUTPUT_0_MEM_CONT_OFFSET,
                                 _35_PADCONV_7_OUTPUT_0_MEM_CONT_SIZE,
                                 _35_PADCONV_7_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _35_PADCONV_7_OUTPUT_0_MEM_WRAP_SIZE,
                                 _35_PADCONV_7_OUTPUT_0_MEM_STRIDE,
                                 _36_SIGMOID_10_OUTPUT_0_MEM_CONT_OFFSET,
                                 _36_SIGMOID_10_OUTPUT_0_MEM_CONT_SIZE,
                                 _36_SIGMOID_10_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _36_SIGMOID_10_OUTPUT_0_MEM_WRAP_SIZE,
                                 _36_SIGMOID_10_OUTPUT_0_MEM_STRIDE,
                                 _37_MUL_10_OUTPUT_0_MEM_CONT_OFFSET,
                                 _37_MUL_10_OUTPUT_0_MEM_CONT_SIZE,
                                 _37_MUL_10_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _37_MUL_10_OUTPUT_0_MEM_WRAP_SIZE,
                                 _37_MUL_10_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_37_Mul_10_output_0, _37_MUL_10_RESCALING, _35_PadConv_7_output_0, _36_Sigmoid_10_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _37_Mul_10\n");
    FILE* _37_MUL_10_OUTPUT_0_STREAM = fopen("data/export_outputs/_37_Mul_10_output_0.txt", "w");
    export_cpp::saveOutputs<_37_MUL_10_OUTPUT_0_NB_OUTPUTS,
                            _37_MUL_10_OUTPUT_0_OUT_HEIGHT,
                            _37_MUL_10_OUTPUT_0_OUT_WIDTH,
                            _37_MUL_10_OUTPUT_0_MEM_CONT_OFFSET,
                            _37_MUL_10_OUTPUT_0_MEM_CONT_SIZE,
                            _37_MUL_10_OUTPUT_0_MEM_WRAP_OFFSET,
                            _37_MUL_10_OUTPUT_0_MEM_WRAP_SIZE,
                            _37_MUL_10_OUTPUT_0_DEV_FMT>
                            (_37_Mul_10_output_0,
                            _37_MUL_10_OUTPUT_0_STREAM);
    fclose(_37_MUL_10_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _38_PadConv_8_output_0 = (float*) (mem_0 + _38_PADCONV_8_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_37_MUL_10_OUTPUT_0_IN_BATCH,
                           _38_PADCONV_8_NB_CHANNELS,
                           _37_MUL_10_OUTPUT_0_IN_HEIGHT,
                           _37_MUL_10_OUTPUT_0_IN_WIDTH,
                           _38_PADCONV_8_NB_OUTPUTS,
                           _38_PADCONV_8_OUTPUT_0_OUT_HEIGHT,
                           _38_PADCONV_8_OUTPUT_0_OUT_WIDTH,
                           _38_PADCONV_8_PADDING_Y,
                           _38_PADCONV_8_PADDING_X,
                           _38_PADCONV_8_STRIDE_Y,
                           _38_PADCONV_8_STRIDE_X,
                           _38_PADCONV_8_DILATION_Y,
                           _38_PADCONV_8_DILATION_X,
                           _38_PADCONV_8_KERNEL_HEIGHT,
                           _38_PADCONV_8_KERNEL_WIDTH,
                           _38_PADCONV_8_ACTIVATION,
                           _37_MUL_10_OUTPUT_0_MEM_CONT_OFFSET,
                           _37_MUL_10_OUTPUT_0_MEM_CONT_SIZE,
                           _37_MUL_10_OUTPUT_0_MEM_WRAP_OFFSET,
                           _37_MUL_10_OUTPUT_0_MEM_WRAP_SIZE,
                           _37_MUL_10_OUTPUT_0_MEM_STRIDE,
                           _38_PADCONV_8_OUTPUT_0_MEM_CONT_OFFSET,
                           _38_PADCONV_8_OUTPUT_0_MEM_CONT_SIZE,
                           _38_PADCONV_8_OUTPUT_0_MEM_WRAP_OFFSET,
                           _38_PADCONV_8_OUTPUT_0_MEM_WRAP_SIZE,
                           _38_PADCONV_8_OUTPUT_0_MEM_STRIDE>
                           (_37_Mul_10_output_0, _38_PadConv_8_output_0, _38_PadConv_8_weights_output_0, _38_PadConv_8_biases_output_0, _38_PADCONV_8_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _38_PadConv_8\n");
    FILE* _38_PADCONV_8_OUTPUT_0_STREAM = fopen("data/export_outputs/_38_PadConv_8_output_0.txt", "w");
    export_cpp::saveOutputs<_38_PADCONV_8_OUTPUT_0_NB_OUTPUTS,
                            _38_PADCONV_8_OUTPUT_0_OUT_HEIGHT,
                            _38_PADCONV_8_OUTPUT_0_OUT_WIDTH,
                            _38_PADCONV_8_OUTPUT_0_MEM_CONT_OFFSET,
                            _38_PADCONV_8_OUTPUT_0_MEM_CONT_SIZE,
                            _38_PADCONV_8_OUTPUT_0_MEM_WRAP_OFFSET,
                            _38_PADCONV_8_OUTPUT_0_MEM_WRAP_SIZE,
                            _38_PADCONV_8_OUTPUT_0_DEV_FMT>
                            (_38_PadConv_8_output_0,
                            _38_PADCONV_8_OUTPUT_0_STREAM);
    fclose(_38_PADCONV_8_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _39_Sigmoid_11_output_0 = (float*) (mem_0 + _39_SIGMOID_11_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_39_SIGMOID_11_NB_ELTS>
                               (_38_PadConv_8_output_0, _39_Sigmoid_11_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _39_Sigmoid_11\n");
    FILE* _39_SIGMOID_11_OUTPUT_0_STREAM = fopen("data/export_outputs/_39_Sigmoid_11_output_0.txt", "w");
    export_cpp::saveOutputs<_39_SIGMOID_11_OUTPUT_0_NB_OUTPUTS,
                            _39_SIGMOID_11_OUTPUT_0_OUT_HEIGHT,
                            _39_SIGMOID_11_OUTPUT_0_OUT_WIDTH,
                            _39_SIGMOID_11_OUTPUT_0_MEM_CONT_OFFSET,
                            _39_SIGMOID_11_OUTPUT_0_MEM_CONT_SIZE,
                            _39_SIGMOID_11_OUTPUT_0_MEM_WRAP_OFFSET,
                            _39_SIGMOID_11_OUTPUT_0_MEM_WRAP_SIZE,
                            _39_SIGMOID_11_OUTPUT_0_DEV_FMT>
                            (_39_Sigmoid_11_output_0,
                            _39_SIGMOID_11_OUTPUT_0_STREAM);
    fclose(_39_SIGMOID_11_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _40_Mul_11_output_0 = (float*) (mem_0 + _40_MUL_11_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_40_MUL_11_NB_MAT,
                                 _40_MUL_11_ELEM_OP,
                                 _40_MUL_11_INPUT1_CONT_SIZE,
                                 _40_MUL_11_INPUT2_CONT_SIZE,
                                 _40_MUL_11_OUTPUT_CONT_SIZE,
                                 _40_MUL_11_OFFSET_IN1,
                                 _40_MUL_11_OFFSET_IN2,
                                 _40_MUL_11_ACTIVATION,
                                 _38_PADCONV_8_OUTPUT_0_MEM_CONT_OFFSET,
                                 _38_PADCONV_8_OUTPUT_0_MEM_CONT_SIZE,
                                 _38_PADCONV_8_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _38_PADCONV_8_OUTPUT_0_MEM_WRAP_SIZE,
                                 _38_PADCONV_8_OUTPUT_0_MEM_STRIDE,
                                 _39_SIGMOID_11_OUTPUT_0_MEM_CONT_OFFSET,
                                 _39_SIGMOID_11_OUTPUT_0_MEM_CONT_SIZE,
                                 _39_SIGMOID_11_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _39_SIGMOID_11_OUTPUT_0_MEM_WRAP_SIZE,
                                 _39_SIGMOID_11_OUTPUT_0_MEM_STRIDE,
                                 _40_MUL_11_OUTPUT_0_MEM_CONT_OFFSET,
                                 _40_MUL_11_OUTPUT_0_MEM_CONT_SIZE,
                                 _40_MUL_11_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _40_MUL_11_OUTPUT_0_MEM_WRAP_SIZE,
                                 _40_MUL_11_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_40_Mul_11_output_0, _40_MUL_11_RESCALING, _38_PadConv_8_output_0, _39_Sigmoid_11_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _40_Mul_11\n");
    FILE* _40_MUL_11_OUTPUT_0_STREAM = fopen("data/export_outputs/_40_Mul_11_output_0.txt", "w");
    export_cpp::saveOutputs<_40_MUL_11_OUTPUT_0_NB_OUTPUTS,
                            _40_MUL_11_OUTPUT_0_OUT_HEIGHT,
                            _40_MUL_11_OUTPUT_0_OUT_WIDTH,
                            _40_MUL_11_OUTPUT_0_MEM_CONT_OFFSET,
                            _40_MUL_11_OUTPUT_0_MEM_CONT_SIZE,
                            _40_MUL_11_OUTPUT_0_MEM_WRAP_OFFSET,
                            _40_MUL_11_OUTPUT_0_MEM_WRAP_SIZE,
                            _40_MUL_11_OUTPUT_0_DEV_FMT>
                            (_40_Mul_11_output_0,
                            _40_MUL_11_OUTPUT_0_STREAM);
    fclose(_40_MUL_11_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _41_Add_2_output_0 = (float*) (mem_0 + _41_ADD_2_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_41_ADD_2_NB_MAT,
                                 _41_ADD_2_ELEM_OP,
                                 _41_ADD_2_INPUT1_CONT_SIZE,
                                 _41_ADD_2_INPUT2_CONT_SIZE,
                                 _41_ADD_2_OUTPUT_CONT_SIZE,
                                 _41_ADD_2_OFFSET_IN1,
                                 _41_ADD_2_OFFSET_IN2,
                                 _41_ADD_2_ACTIVATION,
                                 _34_ADD_1_OUTPUT_0_MEM_CONT_OFFSET,
                                 _34_ADD_1_OUTPUT_0_MEM_CONT_SIZE,
                                 _34_ADD_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _34_ADD_1_OUTPUT_0_MEM_WRAP_SIZE,
                                 _34_ADD_1_OUTPUT_0_MEM_STRIDE,
                                 _40_MUL_11_OUTPUT_0_MEM_CONT_OFFSET,
                                 _40_MUL_11_OUTPUT_0_MEM_CONT_SIZE,
                                 _40_MUL_11_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _40_MUL_11_OUTPUT_0_MEM_WRAP_SIZE,
                                 _40_MUL_11_OUTPUT_0_MEM_STRIDE,
                                 _41_ADD_2_OUTPUT_0_MEM_CONT_OFFSET,
                                 _41_ADD_2_OUTPUT_0_MEM_CONT_SIZE,
                                 _41_ADD_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _41_ADD_2_OUTPUT_0_MEM_WRAP_SIZE,
                                 _41_ADD_2_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_41_Add_2_output_0, _41_ADD_2_RESCALING, _34_Add_1_output_0, _40_Mul_11_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _41_Add_2\n");
    FILE* _41_ADD_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_41_Add_2_output_0.txt", "w");
    export_cpp::saveOutputs<_41_ADD_2_OUTPUT_0_NB_OUTPUTS,
                            _41_ADD_2_OUTPUT_0_OUT_HEIGHT,
                            _41_ADD_2_OUTPUT_0_OUT_WIDTH,
                            _41_ADD_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _41_ADD_2_OUTPUT_0_MEM_CONT_SIZE,
                            _41_ADD_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _41_ADD_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _41_ADD_2_OUTPUT_0_DEV_FMT>
                            (_41_Add_2_output_0,
                            _41_ADD_2_OUTPUT_0_STREAM);
    fclose(_41_ADD_2_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _42_Concat_1_output_0 = (float*) (mem_0 + _42_CONCAT_1_OUTPUT_0_MEM_OFFSET);

    const float* _42_Concat_1_inputs[] = { _27_Split_1_output_0, _27_Split_1_output_1, _34_Add_1_output_0, _41_Add_2_output_0 };
    export_cpp::concat_forward<_42_CONCAT_1_AXIS_SIZE_POST,
                               _42_CONCAT_1_AXIS_SIZE_PRE,
                               _42_CONCAT_1_AXIS_SIZE,
                               _42_CONCAT_1_TOTAL_AXIS_SIZE,
                               4> 
                              (_42_Concat_1_inputs,
                               _42_Concat_1_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _42_Concat_1\n");
    FILE* _42_CONCAT_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_42_Concat_1_output_0.txt", "w");
    export_cpp::saveOutputs<_42_CONCAT_1_OUTPUT_0_NB_OUTPUTS,
                            _42_CONCAT_1_OUTPUT_0_OUT_HEIGHT,
                            _42_CONCAT_1_OUTPUT_0_OUT_WIDTH,
                            _42_CONCAT_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _42_CONCAT_1_OUTPUT_0_MEM_CONT_SIZE,
                            _42_CONCAT_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _42_CONCAT_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _42_CONCAT_1_OUTPUT_0_DEV_FMT>
                            (_42_Concat_1_output_0,
                            _42_CONCAT_1_OUTPUT_0_STREAM);
    fclose(_42_CONCAT_1_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _43_Conv2D_3_output_0 = (float*) (mem_0 + _43_CONV2D_3_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_42_CONCAT_1_OUTPUT_0_IN_BATCH,
                           _43_CONV2D_3_NB_CHANNELS,
                           _42_CONCAT_1_OUTPUT_0_IN_HEIGHT,
                           _42_CONCAT_1_OUTPUT_0_IN_WIDTH,
                           _43_CONV2D_3_NB_OUTPUTS,
                           _43_CONV2D_3_OUTPUT_0_OUT_HEIGHT,
                           _43_CONV2D_3_OUTPUT_0_OUT_WIDTH,
                           _43_CONV2D_3_PADDING_Y,
                           _43_CONV2D_3_PADDING_X,
                           _43_CONV2D_3_STRIDE_Y,
                           _43_CONV2D_3_STRIDE_X,
                           _43_CONV2D_3_DILATION_Y,
                           _43_CONV2D_3_DILATION_X,
                           _43_CONV2D_3_KERNEL_HEIGHT,
                           _43_CONV2D_3_KERNEL_WIDTH,
                           _43_CONV2D_3_ACTIVATION,
                           _42_CONCAT_1_OUTPUT_0_MEM_CONT_OFFSET,
                           _42_CONCAT_1_OUTPUT_0_MEM_CONT_SIZE,
                           _42_CONCAT_1_OUTPUT_0_MEM_WRAP_OFFSET,
                           _42_CONCAT_1_OUTPUT_0_MEM_WRAP_SIZE,
                           _42_CONCAT_1_OUTPUT_0_MEM_STRIDE,
                           _43_CONV2D_3_OUTPUT_0_MEM_CONT_OFFSET,
                           _43_CONV2D_3_OUTPUT_0_MEM_CONT_SIZE,
                           _43_CONV2D_3_OUTPUT_0_MEM_WRAP_OFFSET,
                           _43_CONV2D_3_OUTPUT_0_MEM_WRAP_SIZE,
                           _43_CONV2D_3_OUTPUT_0_MEM_STRIDE>
                           (_42_Concat_1_output_0, _43_Conv2D_3_output_0, _43_Conv2D_3_weights_output_0, _43_Conv2D_3_biases_output_0, _43_CONV2D_3_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _43_Conv2D_3\n");
    FILE* _43_CONV2D_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_43_Conv2D_3_output_0.txt", "w");
    export_cpp::saveOutputs<_43_CONV2D_3_OUTPUT_0_NB_OUTPUTS,
                            _43_CONV2D_3_OUTPUT_0_OUT_HEIGHT,
                            _43_CONV2D_3_OUTPUT_0_OUT_WIDTH,
                            _43_CONV2D_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _43_CONV2D_3_OUTPUT_0_MEM_CONT_SIZE,
                            _43_CONV2D_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _43_CONV2D_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _43_CONV2D_3_OUTPUT_0_DEV_FMT>
                            (_43_Conv2D_3_output_0,
                            _43_CONV2D_3_OUTPUT_0_STREAM);
    fclose(_43_CONV2D_3_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _44_Sigmoid_12_output_0 = (float*) (mem_0 + _44_SIGMOID_12_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_44_SIGMOID_12_NB_ELTS>
                               (_43_Conv2D_3_output_0, _44_Sigmoid_12_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _44_Sigmoid_12\n");
    FILE* _44_SIGMOID_12_OUTPUT_0_STREAM = fopen("data/export_outputs/_44_Sigmoid_12_output_0.txt", "w");
    export_cpp::saveOutputs<_44_SIGMOID_12_OUTPUT_0_NB_OUTPUTS,
                            _44_SIGMOID_12_OUTPUT_0_OUT_HEIGHT,
                            _44_SIGMOID_12_OUTPUT_0_OUT_WIDTH,
                            _44_SIGMOID_12_OUTPUT_0_MEM_CONT_OFFSET,
                            _44_SIGMOID_12_OUTPUT_0_MEM_CONT_SIZE,
                            _44_SIGMOID_12_OUTPUT_0_MEM_WRAP_OFFSET,
                            _44_SIGMOID_12_OUTPUT_0_MEM_WRAP_SIZE,
                            _44_SIGMOID_12_OUTPUT_0_DEV_FMT>
                            (_44_Sigmoid_12_output_0,
                            _44_SIGMOID_12_OUTPUT_0_STREAM);
    fclose(_44_SIGMOID_12_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _45_Mul_12_output_0 = (float*) (mem_0 + _45_MUL_12_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_45_MUL_12_NB_MAT,
                                 _45_MUL_12_ELEM_OP,
                                 _45_MUL_12_INPUT1_CONT_SIZE,
                                 _45_MUL_12_INPUT2_CONT_SIZE,
                                 _45_MUL_12_OUTPUT_CONT_SIZE,
                                 _45_MUL_12_OFFSET_IN1,
                                 _45_MUL_12_OFFSET_IN2,
                                 _45_MUL_12_ACTIVATION,
                                 _43_CONV2D_3_OUTPUT_0_MEM_CONT_OFFSET,
                                 _43_CONV2D_3_OUTPUT_0_MEM_CONT_SIZE,
                                 _43_CONV2D_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _43_CONV2D_3_OUTPUT_0_MEM_WRAP_SIZE,
                                 _43_CONV2D_3_OUTPUT_0_MEM_STRIDE,
                                 _44_SIGMOID_12_OUTPUT_0_MEM_CONT_OFFSET,
                                 _44_SIGMOID_12_OUTPUT_0_MEM_CONT_SIZE,
                                 _44_SIGMOID_12_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _44_SIGMOID_12_OUTPUT_0_MEM_WRAP_SIZE,
                                 _44_SIGMOID_12_OUTPUT_0_MEM_STRIDE,
                                 _45_MUL_12_OUTPUT_0_MEM_CONT_OFFSET,
                                 _45_MUL_12_OUTPUT_0_MEM_CONT_SIZE,
                                 _45_MUL_12_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _45_MUL_12_OUTPUT_0_MEM_WRAP_SIZE,
                                 _45_MUL_12_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_45_Mul_12_output_0, _45_MUL_12_RESCALING, _43_Conv2D_3_output_0, _44_Sigmoid_12_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _45_Mul_12\n");
    FILE* _45_MUL_12_OUTPUT_0_STREAM = fopen("data/export_outputs/_45_Mul_12_output_0.txt", "w");
    export_cpp::saveOutputs<_45_MUL_12_OUTPUT_0_NB_OUTPUTS,
                            _45_MUL_12_OUTPUT_0_OUT_HEIGHT,
                            _45_MUL_12_OUTPUT_0_OUT_WIDTH,
                            _45_MUL_12_OUTPUT_0_MEM_CONT_OFFSET,
                            _45_MUL_12_OUTPUT_0_MEM_CONT_SIZE,
                            _45_MUL_12_OUTPUT_0_MEM_WRAP_OFFSET,
                            _45_MUL_12_OUTPUT_0_MEM_WRAP_SIZE,
                            _45_MUL_12_OUTPUT_0_DEV_FMT>
                            (_45_Mul_12_output_0,
                            _45_MUL_12_OUTPUT_0_STREAM);
    fclose(_45_MUL_12_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _46_Conv2D_4_output_0 = (float*) (mem_0 + _46_CONV2D_4_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_45_MUL_12_OUTPUT_0_IN_BATCH,
                           _46_CONV2D_4_NB_CHANNELS,
                           _45_MUL_12_OUTPUT_0_IN_HEIGHT,
                           _45_MUL_12_OUTPUT_0_IN_WIDTH,
                           _46_CONV2D_4_NB_OUTPUTS,
                           _46_CONV2D_4_OUTPUT_0_OUT_HEIGHT,
                           _46_CONV2D_4_OUTPUT_0_OUT_WIDTH,
                           _46_CONV2D_4_PADDING_Y,
                           _46_CONV2D_4_PADDING_X,
                           _46_CONV2D_4_STRIDE_Y,
                           _46_CONV2D_4_STRIDE_X,
                           _46_CONV2D_4_DILATION_Y,
                           _46_CONV2D_4_DILATION_X,
                           _46_CONV2D_4_KERNEL_HEIGHT,
                           _46_CONV2D_4_KERNEL_WIDTH,
                           _46_CONV2D_4_ACTIVATION,
                           _45_MUL_12_OUTPUT_0_MEM_CONT_OFFSET,
                           _45_MUL_12_OUTPUT_0_MEM_CONT_SIZE,
                           _45_MUL_12_OUTPUT_0_MEM_WRAP_OFFSET,
                           _45_MUL_12_OUTPUT_0_MEM_WRAP_SIZE,
                           _45_MUL_12_OUTPUT_0_MEM_STRIDE,
                           _46_CONV2D_4_OUTPUT_0_MEM_CONT_OFFSET,
                           _46_CONV2D_4_OUTPUT_0_MEM_CONT_SIZE,
                           _46_CONV2D_4_OUTPUT_0_MEM_WRAP_OFFSET,
                           _46_CONV2D_4_OUTPUT_0_MEM_WRAP_SIZE,
                           _46_CONV2D_4_OUTPUT_0_MEM_STRIDE>
                           (_45_Mul_12_output_0, _46_Conv2D_4_output_0, _46_Conv2D_4_weights_output_0, _46_Conv2D_4_biases_output_0, _46_CONV2D_4_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _46_Conv2D_4\n");
    FILE* _46_CONV2D_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_46_Conv2D_4_output_0.txt", "w");
    export_cpp::saveOutputs<_46_CONV2D_4_OUTPUT_0_NB_OUTPUTS,
                            _46_CONV2D_4_OUTPUT_0_OUT_HEIGHT,
                            _46_CONV2D_4_OUTPUT_0_OUT_WIDTH,
                            _46_CONV2D_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _46_CONV2D_4_OUTPUT_0_MEM_CONT_SIZE,
                            _46_CONV2D_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _46_CONV2D_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _46_CONV2D_4_OUTPUT_0_DEV_FMT>
                            (_46_Conv2D_4_output_0,
                            _46_CONV2D_4_OUTPUT_0_STREAM);
    fclose(_46_CONV2D_4_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _47_Sigmoid_13_output_0 = (float*) (mem_0 + _47_SIGMOID_13_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_47_SIGMOID_13_NB_ELTS>
                               (_46_Conv2D_4_output_0, _47_Sigmoid_13_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _47_Sigmoid_13\n");
    FILE* _47_SIGMOID_13_OUTPUT_0_STREAM = fopen("data/export_outputs/_47_Sigmoid_13_output_0.txt", "w");
    export_cpp::saveOutputs<_47_SIGMOID_13_OUTPUT_0_NB_OUTPUTS,
                            _47_SIGMOID_13_OUTPUT_0_OUT_HEIGHT,
                            _47_SIGMOID_13_OUTPUT_0_OUT_WIDTH,
                            _47_SIGMOID_13_OUTPUT_0_MEM_CONT_OFFSET,
                            _47_SIGMOID_13_OUTPUT_0_MEM_CONT_SIZE,
                            _47_SIGMOID_13_OUTPUT_0_MEM_WRAP_OFFSET,
                            _47_SIGMOID_13_OUTPUT_0_MEM_WRAP_SIZE,
                            _47_SIGMOID_13_OUTPUT_0_DEV_FMT>
                            (_47_Sigmoid_13_output_0,
                            _47_SIGMOID_13_OUTPUT_0_STREAM);
    fclose(_47_SIGMOID_13_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _48_Mul_13_output_0 = (float*) (mem_0 + _48_MUL_13_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_48_MUL_13_NB_MAT,
                                 _48_MUL_13_ELEM_OP,
                                 _48_MUL_13_INPUT1_CONT_SIZE,
                                 _48_MUL_13_INPUT2_CONT_SIZE,
                                 _48_MUL_13_OUTPUT_CONT_SIZE,
                                 _48_MUL_13_OFFSET_IN1,
                                 _48_MUL_13_OFFSET_IN2,
                                 _48_MUL_13_ACTIVATION,
                                 _46_CONV2D_4_OUTPUT_0_MEM_CONT_OFFSET,
                                 _46_CONV2D_4_OUTPUT_0_MEM_CONT_SIZE,
                                 _46_CONV2D_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _46_CONV2D_4_OUTPUT_0_MEM_WRAP_SIZE,
                                 _46_CONV2D_4_OUTPUT_0_MEM_STRIDE,
                                 _47_SIGMOID_13_OUTPUT_0_MEM_CONT_OFFSET,
                                 _47_SIGMOID_13_OUTPUT_0_MEM_CONT_SIZE,
                                 _47_SIGMOID_13_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _47_SIGMOID_13_OUTPUT_0_MEM_WRAP_SIZE,
                                 _47_SIGMOID_13_OUTPUT_0_MEM_STRIDE,
                                 _48_MUL_13_OUTPUT_0_MEM_CONT_OFFSET,
                                 _48_MUL_13_OUTPUT_0_MEM_CONT_SIZE,
                                 _48_MUL_13_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _48_MUL_13_OUTPUT_0_MEM_WRAP_SIZE,
                                 _48_MUL_13_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_48_Mul_13_output_0, _48_MUL_13_RESCALING, _46_Conv2D_4_output_0, _47_Sigmoid_13_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _48_Mul_13\n");
    FILE* _48_MUL_13_OUTPUT_0_STREAM = fopen("data/export_outputs/_48_Mul_13_output_0.txt", "w");
    export_cpp::saveOutputs<_48_MUL_13_OUTPUT_0_NB_OUTPUTS,
                            _48_MUL_13_OUTPUT_0_OUT_HEIGHT,
                            _48_MUL_13_OUTPUT_0_OUT_WIDTH,
                            _48_MUL_13_OUTPUT_0_MEM_CONT_OFFSET,
                            _48_MUL_13_OUTPUT_0_MEM_CONT_SIZE,
                            _48_MUL_13_OUTPUT_0_MEM_WRAP_OFFSET,
                            _48_MUL_13_OUTPUT_0_MEM_WRAP_SIZE,
                            _48_MUL_13_OUTPUT_0_DEV_FMT>
                            (_48_Mul_13_output_0,
                            _48_MUL_13_OUTPUT_0_STREAM);
    fclose(_48_MUL_13_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _49_PadConvDw_0_output_0 = (float*) (mem_0 + _49_PADCONVDW_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_48_MUL_13_OUTPUT_0_IN_BATCH,
                           _49_PADCONVDW_0_NB_CHANNELS,
                           _48_MUL_13_OUTPUT_0_IN_HEIGHT,
                           _48_MUL_13_OUTPUT_0_IN_WIDTH,
                           _49_PADCONVDW_0_NB_OUTPUTS,
                           _49_PADCONVDW_0_OUTPUT_0_OUT_HEIGHT,
                           _49_PADCONVDW_0_OUTPUT_0_OUT_WIDTH,
                           _49_PADCONVDW_0_PADDING_Y,
                           _49_PADCONVDW_0_PADDING_X,
                           _49_PADCONVDW_0_STRIDE_Y,
                           _49_PADCONVDW_0_STRIDE_X,
                           _49_PADCONVDW_0_DILATION_Y,
                           _49_PADCONVDW_0_DILATION_X,
                           _49_PADCONVDW_0_KERNEL_HEIGHT,
                           _49_PADCONVDW_0_KERNEL_WIDTH,
                           _49_PADCONVDW_0_ACTIVATION,
                           _48_MUL_13_OUTPUT_0_MEM_CONT_OFFSET,
                           _48_MUL_13_OUTPUT_0_MEM_CONT_SIZE,
                           _48_MUL_13_OUTPUT_0_MEM_WRAP_OFFSET,
                           _48_MUL_13_OUTPUT_0_MEM_WRAP_SIZE,
                           _48_MUL_13_OUTPUT_0_MEM_STRIDE,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_CONT_OFFSET,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_CONT_SIZE,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_WRAP_OFFSET,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_WRAP_SIZE,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_STRIDE>
                           (_48_Mul_13_output_0, _49_PadConvDw_0_output_0, _49_PadConvDw_0_weights_output_0, _49_PadConvDw_0_biases_output_0, _49_PADCONVDW_0_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _49_PadConvDw_0\n");
    FILE* _49_PADCONVDW_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_49_PadConvDw_0_output_0.txt", "w");
    export_cpp::saveOutputs<_49_PADCONVDW_0_OUTPUT_0_NB_OUTPUTS,
                            _49_PADCONVDW_0_OUTPUT_0_OUT_HEIGHT,
                            _49_PADCONVDW_0_OUTPUT_0_OUT_WIDTH,
                            _49_PADCONVDW_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _49_PADCONVDW_0_OUTPUT_0_MEM_CONT_SIZE,
                            _49_PADCONVDW_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _49_PADCONVDW_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _49_PADCONVDW_0_OUTPUT_0_DEV_FMT>
                            (_49_PadConvDw_0_output_0,
                            _49_PADCONVDW_0_OUTPUT_0_STREAM);
    fclose(_49_PADCONVDW_0_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _50_Conv2D_5_output_0 = (float*) (mem_0 + _50_CONV2D_5_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_49_PADCONVDW_0_OUTPUT_0_IN_BATCH,
                           _50_CONV2D_5_NB_CHANNELS,
                           _49_PADCONVDW_0_OUTPUT_0_IN_HEIGHT,
                           _49_PADCONVDW_0_OUTPUT_0_IN_WIDTH,
                           _50_CONV2D_5_NB_OUTPUTS,
                           _50_CONV2D_5_OUTPUT_0_OUT_HEIGHT,
                           _50_CONV2D_5_OUTPUT_0_OUT_WIDTH,
                           _50_CONV2D_5_PADDING_Y,
                           _50_CONV2D_5_PADDING_X,
                           _50_CONV2D_5_STRIDE_Y,
                           _50_CONV2D_5_STRIDE_X,
                           _50_CONV2D_5_DILATION_Y,
                           _50_CONV2D_5_DILATION_X,
                           _50_CONV2D_5_KERNEL_HEIGHT,
                           _50_CONV2D_5_KERNEL_WIDTH,
                           _50_CONV2D_5_ACTIVATION,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_CONT_OFFSET,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_CONT_SIZE,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_WRAP_OFFSET,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_WRAP_SIZE,
                           _49_PADCONVDW_0_OUTPUT_0_MEM_STRIDE,
                           _50_CONV2D_5_OUTPUT_0_MEM_CONT_OFFSET,
                           _50_CONV2D_5_OUTPUT_0_MEM_CONT_SIZE,
                           _50_CONV2D_5_OUTPUT_0_MEM_WRAP_OFFSET,
                           _50_CONV2D_5_OUTPUT_0_MEM_WRAP_SIZE,
                           _50_CONV2D_5_OUTPUT_0_MEM_STRIDE>
                           (_49_PadConvDw_0_output_0, _50_Conv2D_5_output_0, _50_Conv2D_5_weights_output_0, _50_Conv2D_5_biases_output_0, _50_CONV2D_5_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _50_Conv2D_5\n");
    FILE* _50_CONV2D_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_50_Conv2D_5_output_0.txt", "w");
    export_cpp::saveOutputs<_50_CONV2D_5_OUTPUT_0_NB_OUTPUTS,
                            _50_CONV2D_5_OUTPUT_0_OUT_HEIGHT,
                            _50_CONV2D_5_OUTPUT_0_OUT_WIDTH,
                            _50_CONV2D_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _50_CONV2D_5_OUTPUT_0_MEM_CONT_SIZE,
                            _50_CONV2D_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _50_CONV2D_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _50_CONV2D_5_OUTPUT_0_DEV_FMT>
                            (_50_Conv2D_5_output_0,
                            _50_CONV2D_5_OUTPUT_0_STREAM);
    fclose(_50_CONV2D_5_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _51_Sigmoid_14_output_0 = (float*) (mem_0 + _51_SIGMOID_14_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_51_SIGMOID_14_NB_ELTS>
                               (_50_Conv2D_5_output_0, _51_Sigmoid_14_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _51_Sigmoid_14\n");
    FILE* _51_SIGMOID_14_OUTPUT_0_STREAM = fopen("data/export_outputs/_51_Sigmoid_14_output_0.txt", "w");
    export_cpp::saveOutputs<_51_SIGMOID_14_OUTPUT_0_NB_OUTPUTS,
                            _51_SIGMOID_14_OUTPUT_0_OUT_HEIGHT,
                            _51_SIGMOID_14_OUTPUT_0_OUT_WIDTH,
                            _51_SIGMOID_14_OUTPUT_0_MEM_CONT_OFFSET,
                            _51_SIGMOID_14_OUTPUT_0_MEM_CONT_SIZE,
                            _51_SIGMOID_14_OUTPUT_0_MEM_WRAP_OFFSET,
                            _51_SIGMOID_14_OUTPUT_0_MEM_WRAP_SIZE,
                            _51_SIGMOID_14_OUTPUT_0_DEV_FMT>
                            (_51_Sigmoid_14_output_0,
                            _51_SIGMOID_14_OUTPUT_0_STREAM);
    fclose(_51_SIGMOID_14_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _52_Mul_14_output_0 = (float*) (mem_0 + _52_MUL_14_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_52_MUL_14_NB_MAT,
                                 _52_MUL_14_ELEM_OP,
                                 _52_MUL_14_INPUT1_CONT_SIZE,
                                 _52_MUL_14_INPUT2_CONT_SIZE,
                                 _52_MUL_14_OUTPUT_CONT_SIZE,
                                 _52_MUL_14_OFFSET_IN1,
                                 _52_MUL_14_OFFSET_IN2,
                                 _52_MUL_14_ACTIVATION,
                                 _50_CONV2D_5_OUTPUT_0_MEM_CONT_OFFSET,
                                 _50_CONV2D_5_OUTPUT_0_MEM_CONT_SIZE,
                                 _50_CONV2D_5_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _50_CONV2D_5_OUTPUT_0_MEM_WRAP_SIZE,
                                 _50_CONV2D_5_OUTPUT_0_MEM_STRIDE,
                                 _51_SIGMOID_14_OUTPUT_0_MEM_CONT_OFFSET,
                                 _51_SIGMOID_14_OUTPUT_0_MEM_CONT_SIZE,
                                 _51_SIGMOID_14_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _51_SIGMOID_14_OUTPUT_0_MEM_WRAP_SIZE,
                                 _51_SIGMOID_14_OUTPUT_0_MEM_STRIDE,
                                 _52_MUL_14_OUTPUT_0_MEM_CONT_OFFSET,
                                 _52_MUL_14_OUTPUT_0_MEM_CONT_SIZE,
                                 _52_MUL_14_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _52_MUL_14_OUTPUT_0_MEM_WRAP_SIZE,
                                 _52_MUL_14_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_52_Mul_14_output_0, _52_MUL_14_RESCALING, _50_Conv2D_5_output_0, _51_Sigmoid_14_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _52_Mul_14\n");
    FILE* _52_MUL_14_OUTPUT_0_STREAM = fopen("data/export_outputs/_52_Mul_14_output_0.txt", "w");
    export_cpp::saveOutputs<_52_MUL_14_OUTPUT_0_NB_OUTPUTS,
                            _52_MUL_14_OUTPUT_0_OUT_HEIGHT,
                            _52_MUL_14_OUTPUT_0_OUT_WIDTH,
                            _52_MUL_14_OUTPUT_0_MEM_CONT_OFFSET,
                            _52_MUL_14_OUTPUT_0_MEM_CONT_SIZE,
                            _52_MUL_14_OUTPUT_0_MEM_WRAP_OFFSET,
                            _52_MUL_14_OUTPUT_0_MEM_WRAP_SIZE,
                            _52_MUL_14_OUTPUT_0_DEV_FMT>
                            (_52_Mul_14_output_0,
                            _52_MUL_14_OUTPUT_0_STREAM);
    fclose(_52_MUL_14_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _53_Split_2_output_0 = (float*) (mem_0 + _53_SPLIT_2_OUTPUT_0_MEM_OFFSET);
    float* _53_Split_2_output_1 = (float*) (mem_0 + _53_SPLIT_2_OUTPUT_1_MEM_OFFSET);



    float* __53_Split_2_outputs[_53_SPLIT_2_NB_OUTPUTS] = {
        
        _53_Split_2_output_0,
        
        _53_Split_2_output_1
        
    };

    export_cpp::split_forward<_53_SPLIT_2_NB_OUTPUTS,
                              _53_SPLIT_2_AXIS_DIM,
                              _53_SPLIT_2_PRE_AXIS_ELEMS,
                              _53_SPLIT_2_POST_AXIS_ELEMS,
                              _53_SPLIT_2_SPLITS,
                              float>
                              (_52_Mul_14_output_0,
                               __53_Split_2_outputs);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _53_Split_2\n");
    FILE* _53_SPLIT_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_53_Split_2_output_0.txt", "w");
    export_cpp::saveOutputs<_53_SPLIT_2_OUTPUT_0_NB_OUTPUTS,
                            _53_SPLIT_2_OUTPUT_0_OUT_HEIGHT,
                            _53_SPLIT_2_OUTPUT_0_OUT_WIDTH,
                            _53_SPLIT_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _53_SPLIT_2_OUTPUT_0_MEM_CONT_SIZE,
                            _53_SPLIT_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _53_SPLIT_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _53_SPLIT_2_OUTPUT_0_DEV_FMT>
                            (_53_Split_2_output_0,
                            _53_SPLIT_2_OUTPUT_0_STREAM);
    fclose(_53_SPLIT_2_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _54_PadConv_9_output_0 = (float*) (mem_0 + _54_PADCONV_9_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_53_SPLIT_2_OUTPUT_1_IN_BATCH,
                           _54_PADCONV_9_NB_CHANNELS,
                           _53_SPLIT_2_OUTPUT_1_IN_HEIGHT,
                           _53_SPLIT_2_OUTPUT_1_IN_WIDTH,
                           _54_PADCONV_9_NB_OUTPUTS,
                           _54_PADCONV_9_OUTPUT_0_OUT_HEIGHT,
                           _54_PADCONV_9_OUTPUT_0_OUT_WIDTH,
                           _54_PADCONV_9_PADDING_Y,
                           _54_PADCONV_9_PADDING_X,
                           _54_PADCONV_9_STRIDE_Y,
                           _54_PADCONV_9_STRIDE_X,
                           _54_PADCONV_9_DILATION_Y,
                           _54_PADCONV_9_DILATION_X,
                           _54_PADCONV_9_KERNEL_HEIGHT,
                           _54_PADCONV_9_KERNEL_WIDTH,
                           _54_PADCONV_9_ACTIVATION,
                           _53_SPLIT_2_OUTPUT_1_MEM_CONT_OFFSET,
                           _53_SPLIT_2_OUTPUT_1_MEM_CONT_SIZE,
                           _53_SPLIT_2_OUTPUT_1_MEM_WRAP_OFFSET,
                           _53_SPLIT_2_OUTPUT_1_MEM_WRAP_SIZE,
                           _53_SPLIT_2_OUTPUT_1_MEM_STRIDE,
                           _54_PADCONV_9_OUTPUT_0_MEM_CONT_OFFSET,
                           _54_PADCONV_9_OUTPUT_0_MEM_CONT_SIZE,
                           _54_PADCONV_9_OUTPUT_0_MEM_WRAP_OFFSET,
                           _54_PADCONV_9_OUTPUT_0_MEM_WRAP_SIZE,
                           _54_PADCONV_9_OUTPUT_0_MEM_STRIDE>
                           (_53_Split_2_output_1, _54_PadConv_9_output_0, _54_PadConv_9_weights_output_0, _54_PadConv_9_biases_output_0, _54_PADCONV_9_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _54_PadConv_9\n");
    FILE* _54_PADCONV_9_OUTPUT_0_STREAM = fopen("data/export_outputs/_54_PadConv_9_output_0.txt", "w");
    export_cpp::saveOutputs<_54_PADCONV_9_OUTPUT_0_NB_OUTPUTS,
                            _54_PADCONV_9_OUTPUT_0_OUT_HEIGHT,
                            _54_PADCONV_9_OUTPUT_0_OUT_WIDTH,
                            _54_PADCONV_9_OUTPUT_0_MEM_CONT_OFFSET,
                            _54_PADCONV_9_OUTPUT_0_MEM_CONT_SIZE,
                            _54_PADCONV_9_OUTPUT_0_MEM_WRAP_OFFSET,
                            _54_PADCONV_9_OUTPUT_0_MEM_WRAP_SIZE,
                            _54_PADCONV_9_OUTPUT_0_DEV_FMT>
                            (_54_PadConv_9_output_0,
                            _54_PADCONV_9_OUTPUT_0_STREAM);
    fclose(_54_PADCONV_9_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _55_Sigmoid_15_output_0 = (float*) (mem_0 + _55_SIGMOID_15_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_55_SIGMOID_15_NB_ELTS>
                               (_54_PadConv_9_output_0, _55_Sigmoid_15_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _55_Sigmoid_15\n");
    FILE* _55_SIGMOID_15_OUTPUT_0_STREAM = fopen("data/export_outputs/_55_Sigmoid_15_output_0.txt", "w");
    export_cpp::saveOutputs<_55_SIGMOID_15_OUTPUT_0_NB_OUTPUTS,
                            _55_SIGMOID_15_OUTPUT_0_OUT_HEIGHT,
                            _55_SIGMOID_15_OUTPUT_0_OUT_WIDTH,
                            _55_SIGMOID_15_OUTPUT_0_MEM_CONT_OFFSET,
                            _55_SIGMOID_15_OUTPUT_0_MEM_CONT_SIZE,
                            _55_SIGMOID_15_OUTPUT_0_MEM_WRAP_OFFSET,
                            _55_SIGMOID_15_OUTPUT_0_MEM_WRAP_SIZE,
                            _55_SIGMOID_15_OUTPUT_0_DEV_FMT>
                            (_55_Sigmoid_15_output_0,
                            _55_SIGMOID_15_OUTPUT_0_STREAM);
    fclose(_55_SIGMOID_15_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _56_Mul_15_output_0 = (float*) (mem_0 + _56_MUL_15_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_56_MUL_15_NB_MAT,
                                 _56_MUL_15_ELEM_OP,
                                 _56_MUL_15_INPUT1_CONT_SIZE,
                                 _56_MUL_15_INPUT2_CONT_SIZE,
                                 _56_MUL_15_OUTPUT_CONT_SIZE,
                                 _56_MUL_15_OFFSET_IN1,
                                 _56_MUL_15_OFFSET_IN2,
                                 _56_MUL_15_ACTIVATION,
                                 _54_PADCONV_9_OUTPUT_0_MEM_CONT_OFFSET,
                                 _54_PADCONV_9_OUTPUT_0_MEM_CONT_SIZE,
                                 _54_PADCONV_9_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _54_PADCONV_9_OUTPUT_0_MEM_WRAP_SIZE,
                                 _54_PADCONV_9_OUTPUT_0_MEM_STRIDE,
                                 _55_SIGMOID_15_OUTPUT_0_MEM_CONT_OFFSET,
                                 _55_SIGMOID_15_OUTPUT_0_MEM_CONT_SIZE,
                                 _55_SIGMOID_15_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _55_SIGMOID_15_OUTPUT_0_MEM_WRAP_SIZE,
                                 _55_SIGMOID_15_OUTPUT_0_MEM_STRIDE,
                                 _56_MUL_15_OUTPUT_0_MEM_CONT_OFFSET,
                                 _56_MUL_15_OUTPUT_0_MEM_CONT_SIZE,
                                 _56_MUL_15_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _56_MUL_15_OUTPUT_0_MEM_WRAP_SIZE,
                                 _56_MUL_15_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_56_Mul_15_output_0, _56_MUL_15_RESCALING, _54_PadConv_9_output_0, _55_Sigmoid_15_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _56_Mul_15\n");
    FILE* _56_MUL_15_OUTPUT_0_STREAM = fopen("data/export_outputs/_56_Mul_15_output_0.txt", "w");
    export_cpp::saveOutputs<_56_MUL_15_OUTPUT_0_NB_OUTPUTS,
                            _56_MUL_15_OUTPUT_0_OUT_HEIGHT,
                            _56_MUL_15_OUTPUT_0_OUT_WIDTH,
                            _56_MUL_15_OUTPUT_0_MEM_CONT_OFFSET,
                            _56_MUL_15_OUTPUT_0_MEM_CONT_SIZE,
                            _56_MUL_15_OUTPUT_0_MEM_WRAP_OFFSET,
                            _56_MUL_15_OUTPUT_0_MEM_WRAP_SIZE,
                            _56_MUL_15_OUTPUT_0_DEV_FMT>
                            (_56_Mul_15_output_0,
                            _56_MUL_15_OUTPUT_0_STREAM);
    fclose(_56_MUL_15_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _57_PadConv_10_output_0 = (float*) (mem_0 + _57_PADCONV_10_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_56_MUL_15_OUTPUT_0_IN_BATCH,
                           _57_PADCONV_10_NB_CHANNELS,
                           _56_MUL_15_OUTPUT_0_IN_HEIGHT,
                           _56_MUL_15_OUTPUT_0_IN_WIDTH,
                           _57_PADCONV_10_NB_OUTPUTS,
                           _57_PADCONV_10_OUTPUT_0_OUT_HEIGHT,
                           _57_PADCONV_10_OUTPUT_0_OUT_WIDTH,
                           _57_PADCONV_10_PADDING_Y,
                           _57_PADCONV_10_PADDING_X,
                           _57_PADCONV_10_STRIDE_Y,
                           _57_PADCONV_10_STRIDE_X,
                           _57_PADCONV_10_DILATION_Y,
                           _57_PADCONV_10_DILATION_X,
                           _57_PADCONV_10_KERNEL_HEIGHT,
                           _57_PADCONV_10_KERNEL_WIDTH,
                           _57_PADCONV_10_ACTIVATION,
                           _56_MUL_15_OUTPUT_0_MEM_CONT_OFFSET,
                           _56_MUL_15_OUTPUT_0_MEM_CONT_SIZE,
                           _56_MUL_15_OUTPUT_0_MEM_WRAP_OFFSET,
                           _56_MUL_15_OUTPUT_0_MEM_WRAP_SIZE,
                           _56_MUL_15_OUTPUT_0_MEM_STRIDE,
                           _57_PADCONV_10_OUTPUT_0_MEM_CONT_OFFSET,
                           _57_PADCONV_10_OUTPUT_0_MEM_CONT_SIZE,
                           _57_PADCONV_10_OUTPUT_0_MEM_WRAP_OFFSET,
                           _57_PADCONV_10_OUTPUT_0_MEM_WRAP_SIZE,
                           _57_PADCONV_10_OUTPUT_0_MEM_STRIDE>
                           (_56_Mul_15_output_0, _57_PadConv_10_output_0, _57_PadConv_10_weights_output_0, _57_PadConv_10_biases_output_0, _57_PADCONV_10_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _57_PadConv_10\n");
    FILE* _57_PADCONV_10_OUTPUT_0_STREAM = fopen("data/export_outputs/_57_PadConv_10_output_0.txt", "w");
    export_cpp::saveOutputs<_57_PADCONV_10_OUTPUT_0_NB_OUTPUTS,
                            _57_PADCONV_10_OUTPUT_0_OUT_HEIGHT,
                            _57_PADCONV_10_OUTPUT_0_OUT_WIDTH,
                            _57_PADCONV_10_OUTPUT_0_MEM_CONT_OFFSET,
                            _57_PADCONV_10_OUTPUT_0_MEM_CONT_SIZE,
                            _57_PADCONV_10_OUTPUT_0_MEM_WRAP_OFFSET,
                            _57_PADCONV_10_OUTPUT_0_MEM_WRAP_SIZE,
                            _57_PADCONV_10_OUTPUT_0_DEV_FMT>
                            (_57_PadConv_10_output_0,
                            _57_PADCONV_10_OUTPUT_0_STREAM);
    fclose(_57_PADCONV_10_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _58_Sigmoid_16_output_0 = (float*) (mem_0 + _58_SIGMOID_16_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_58_SIGMOID_16_NB_ELTS>
                               (_57_PadConv_10_output_0, _58_Sigmoid_16_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _58_Sigmoid_16\n");
    FILE* _58_SIGMOID_16_OUTPUT_0_STREAM = fopen("data/export_outputs/_58_Sigmoid_16_output_0.txt", "w");
    export_cpp::saveOutputs<_58_SIGMOID_16_OUTPUT_0_NB_OUTPUTS,
                            _58_SIGMOID_16_OUTPUT_0_OUT_HEIGHT,
                            _58_SIGMOID_16_OUTPUT_0_OUT_WIDTH,
                            _58_SIGMOID_16_OUTPUT_0_MEM_CONT_OFFSET,
                            _58_SIGMOID_16_OUTPUT_0_MEM_CONT_SIZE,
                            _58_SIGMOID_16_OUTPUT_0_MEM_WRAP_OFFSET,
                            _58_SIGMOID_16_OUTPUT_0_MEM_WRAP_SIZE,
                            _58_SIGMOID_16_OUTPUT_0_DEV_FMT>
                            (_58_Sigmoid_16_output_0,
                            _58_SIGMOID_16_OUTPUT_0_STREAM);
    fclose(_58_SIGMOID_16_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _59_Mul_16_output_0 = (float*) (mem_0 + _59_MUL_16_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_59_MUL_16_NB_MAT,
                                 _59_MUL_16_ELEM_OP,
                                 _59_MUL_16_INPUT1_CONT_SIZE,
                                 _59_MUL_16_INPUT2_CONT_SIZE,
                                 _59_MUL_16_OUTPUT_CONT_SIZE,
                                 _59_MUL_16_OFFSET_IN1,
                                 _59_MUL_16_OFFSET_IN2,
                                 _59_MUL_16_ACTIVATION,
                                 _57_PADCONV_10_OUTPUT_0_MEM_CONT_OFFSET,
                                 _57_PADCONV_10_OUTPUT_0_MEM_CONT_SIZE,
                                 _57_PADCONV_10_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _57_PADCONV_10_OUTPUT_0_MEM_WRAP_SIZE,
                                 _57_PADCONV_10_OUTPUT_0_MEM_STRIDE,
                                 _58_SIGMOID_16_OUTPUT_0_MEM_CONT_OFFSET,
                                 _58_SIGMOID_16_OUTPUT_0_MEM_CONT_SIZE,
                                 _58_SIGMOID_16_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _58_SIGMOID_16_OUTPUT_0_MEM_WRAP_SIZE,
                                 _58_SIGMOID_16_OUTPUT_0_MEM_STRIDE,
                                 _59_MUL_16_OUTPUT_0_MEM_CONT_OFFSET,
                                 _59_MUL_16_OUTPUT_0_MEM_CONT_SIZE,
                                 _59_MUL_16_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _59_MUL_16_OUTPUT_0_MEM_WRAP_SIZE,
                                 _59_MUL_16_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_59_Mul_16_output_0, _59_MUL_16_RESCALING, _57_PadConv_10_output_0, _58_Sigmoid_16_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _59_Mul_16\n");
    FILE* _59_MUL_16_OUTPUT_0_STREAM = fopen("data/export_outputs/_59_Mul_16_output_0.txt", "w");
    export_cpp::saveOutputs<_59_MUL_16_OUTPUT_0_NB_OUTPUTS,
                            _59_MUL_16_OUTPUT_0_OUT_HEIGHT,
                            _59_MUL_16_OUTPUT_0_OUT_WIDTH,
                            _59_MUL_16_OUTPUT_0_MEM_CONT_OFFSET,
                            _59_MUL_16_OUTPUT_0_MEM_CONT_SIZE,
                            _59_MUL_16_OUTPUT_0_MEM_WRAP_OFFSET,
                            _59_MUL_16_OUTPUT_0_MEM_WRAP_SIZE,
                            _59_MUL_16_OUTPUT_0_DEV_FMT>
                            (_59_Mul_16_output_0,
                            _59_MUL_16_OUTPUT_0_STREAM);
    fclose(_59_MUL_16_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _60_Add_3_output_0 = (float*) (mem_0 + _60_ADD_3_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_60_ADD_3_NB_MAT,
                                 _60_ADD_3_ELEM_OP,
                                 _60_ADD_3_INPUT1_CONT_SIZE,
                                 _60_ADD_3_INPUT2_CONT_SIZE,
                                 _60_ADD_3_OUTPUT_CONT_SIZE,
                                 _60_ADD_3_OFFSET_IN1,
                                 _60_ADD_3_OFFSET_IN2,
                                 _60_ADD_3_ACTIVATION,
                                 _53_SPLIT_2_OUTPUT_1_MEM_CONT_OFFSET,
                                 _53_SPLIT_2_OUTPUT_1_MEM_CONT_SIZE,
                                 _53_SPLIT_2_OUTPUT_1_MEM_WRAP_OFFSET,
                                 _53_SPLIT_2_OUTPUT_1_MEM_WRAP_SIZE,
                                 _53_SPLIT_2_OUTPUT_1_MEM_STRIDE,
                                 _59_MUL_16_OUTPUT_0_MEM_CONT_OFFSET,
                                 _59_MUL_16_OUTPUT_0_MEM_CONT_SIZE,
                                 _59_MUL_16_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _59_MUL_16_OUTPUT_0_MEM_WRAP_SIZE,
                                 _59_MUL_16_OUTPUT_0_MEM_STRIDE,
                                 _60_ADD_3_OUTPUT_0_MEM_CONT_OFFSET,
                                 _60_ADD_3_OUTPUT_0_MEM_CONT_SIZE,
                                 _60_ADD_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _60_ADD_3_OUTPUT_0_MEM_WRAP_SIZE,
                                 _60_ADD_3_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_60_Add_3_output_0, _60_ADD_3_RESCALING, _53_Split_2_output_1, _59_Mul_16_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _60_Add_3\n");
    FILE* _60_ADD_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_60_Add_3_output_0.txt", "w");
    export_cpp::saveOutputs<_60_ADD_3_OUTPUT_0_NB_OUTPUTS,
                            _60_ADD_3_OUTPUT_0_OUT_HEIGHT,
                            _60_ADD_3_OUTPUT_0_OUT_WIDTH,
                            _60_ADD_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _60_ADD_3_OUTPUT_0_MEM_CONT_SIZE,
                            _60_ADD_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _60_ADD_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _60_ADD_3_OUTPUT_0_DEV_FMT>
                            (_60_Add_3_output_0,
                            _60_ADD_3_OUTPUT_0_STREAM);
    fclose(_60_ADD_3_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _61_PadConv_11_output_0 = (float*) (mem_0 + _61_PADCONV_11_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_60_ADD_3_OUTPUT_0_IN_BATCH,
                           _61_PADCONV_11_NB_CHANNELS,
                           _60_ADD_3_OUTPUT_0_IN_HEIGHT,
                           _60_ADD_3_OUTPUT_0_IN_WIDTH,
                           _61_PADCONV_11_NB_OUTPUTS,
                           _61_PADCONV_11_OUTPUT_0_OUT_HEIGHT,
                           _61_PADCONV_11_OUTPUT_0_OUT_WIDTH,
                           _61_PADCONV_11_PADDING_Y,
                           _61_PADCONV_11_PADDING_X,
                           _61_PADCONV_11_STRIDE_Y,
                           _61_PADCONV_11_STRIDE_X,
                           _61_PADCONV_11_DILATION_Y,
                           _61_PADCONV_11_DILATION_X,
                           _61_PADCONV_11_KERNEL_HEIGHT,
                           _61_PADCONV_11_KERNEL_WIDTH,
                           _61_PADCONV_11_ACTIVATION,
                           _60_ADD_3_OUTPUT_0_MEM_CONT_OFFSET,
                           _60_ADD_3_OUTPUT_0_MEM_CONT_SIZE,
                           _60_ADD_3_OUTPUT_0_MEM_WRAP_OFFSET,
                           _60_ADD_3_OUTPUT_0_MEM_WRAP_SIZE,
                           _60_ADD_3_OUTPUT_0_MEM_STRIDE,
                           _61_PADCONV_11_OUTPUT_0_MEM_CONT_OFFSET,
                           _61_PADCONV_11_OUTPUT_0_MEM_CONT_SIZE,
                           _61_PADCONV_11_OUTPUT_0_MEM_WRAP_OFFSET,
                           _61_PADCONV_11_OUTPUT_0_MEM_WRAP_SIZE,
                           _61_PADCONV_11_OUTPUT_0_MEM_STRIDE>
                           (_60_Add_3_output_0, _61_PadConv_11_output_0, _61_PadConv_11_weights_output_0, _61_PadConv_11_biases_output_0, _61_PADCONV_11_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _61_PadConv_11\n");
    FILE* _61_PADCONV_11_OUTPUT_0_STREAM = fopen("data/export_outputs/_61_PadConv_11_output_0.txt", "w");
    export_cpp::saveOutputs<_61_PADCONV_11_OUTPUT_0_NB_OUTPUTS,
                            _61_PADCONV_11_OUTPUT_0_OUT_HEIGHT,
                            _61_PADCONV_11_OUTPUT_0_OUT_WIDTH,
                            _61_PADCONV_11_OUTPUT_0_MEM_CONT_OFFSET,
                            _61_PADCONV_11_OUTPUT_0_MEM_CONT_SIZE,
                            _61_PADCONV_11_OUTPUT_0_MEM_WRAP_OFFSET,
                            _61_PADCONV_11_OUTPUT_0_MEM_WRAP_SIZE,
                            _61_PADCONV_11_OUTPUT_0_DEV_FMT>
                            (_61_PadConv_11_output_0,
                            _61_PADCONV_11_OUTPUT_0_STREAM);
    fclose(_61_PADCONV_11_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _62_Sigmoid_17_output_0 = (float*) (mem_0 + _62_SIGMOID_17_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_62_SIGMOID_17_NB_ELTS>
                               (_61_PadConv_11_output_0, _62_Sigmoid_17_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _62_Sigmoid_17\n");
    FILE* _62_SIGMOID_17_OUTPUT_0_STREAM = fopen("data/export_outputs/_62_Sigmoid_17_output_0.txt", "w");
    export_cpp::saveOutputs<_62_SIGMOID_17_OUTPUT_0_NB_OUTPUTS,
                            _62_SIGMOID_17_OUTPUT_0_OUT_HEIGHT,
                            _62_SIGMOID_17_OUTPUT_0_OUT_WIDTH,
                            _62_SIGMOID_17_OUTPUT_0_MEM_CONT_OFFSET,
                            _62_SIGMOID_17_OUTPUT_0_MEM_CONT_SIZE,
                            _62_SIGMOID_17_OUTPUT_0_MEM_WRAP_OFFSET,
                            _62_SIGMOID_17_OUTPUT_0_MEM_WRAP_SIZE,
                            _62_SIGMOID_17_OUTPUT_0_DEV_FMT>
                            (_62_Sigmoid_17_output_0,
                            _62_SIGMOID_17_OUTPUT_0_STREAM);
    fclose(_62_SIGMOID_17_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _63_Mul_17_output_0 = (float*) (mem_0 + _63_MUL_17_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_63_MUL_17_NB_MAT,
                                 _63_MUL_17_ELEM_OP,
                                 _63_MUL_17_INPUT1_CONT_SIZE,
                                 _63_MUL_17_INPUT2_CONT_SIZE,
                                 _63_MUL_17_OUTPUT_CONT_SIZE,
                                 _63_MUL_17_OFFSET_IN1,
                                 _63_MUL_17_OFFSET_IN2,
                                 _63_MUL_17_ACTIVATION,
                                 _61_PADCONV_11_OUTPUT_0_MEM_CONT_OFFSET,
                                 _61_PADCONV_11_OUTPUT_0_MEM_CONT_SIZE,
                                 _61_PADCONV_11_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _61_PADCONV_11_OUTPUT_0_MEM_WRAP_SIZE,
                                 _61_PADCONV_11_OUTPUT_0_MEM_STRIDE,
                                 _62_SIGMOID_17_OUTPUT_0_MEM_CONT_OFFSET,
                                 _62_SIGMOID_17_OUTPUT_0_MEM_CONT_SIZE,
                                 _62_SIGMOID_17_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _62_SIGMOID_17_OUTPUT_0_MEM_WRAP_SIZE,
                                 _62_SIGMOID_17_OUTPUT_0_MEM_STRIDE,
                                 _63_MUL_17_OUTPUT_0_MEM_CONT_OFFSET,
                                 _63_MUL_17_OUTPUT_0_MEM_CONT_SIZE,
                                 _63_MUL_17_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _63_MUL_17_OUTPUT_0_MEM_WRAP_SIZE,
                                 _63_MUL_17_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_63_Mul_17_output_0, _63_MUL_17_RESCALING, _61_PadConv_11_output_0, _62_Sigmoid_17_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _63_Mul_17\n");
    FILE* _63_MUL_17_OUTPUT_0_STREAM = fopen("data/export_outputs/_63_Mul_17_output_0.txt", "w");
    export_cpp::saveOutputs<_63_MUL_17_OUTPUT_0_NB_OUTPUTS,
                            _63_MUL_17_OUTPUT_0_OUT_HEIGHT,
                            _63_MUL_17_OUTPUT_0_OUT_WIDTH,
                            _63_MUL_17_OUTPUT_0_MEM_CONT_OFFSET,
                            _63_MUL_17_OUTPUT_0_MEM_CONT_SIZE,
                            _63_MUL_17_OUTPUT_0_MEM_WRAP_OFFSET,
                            _63_MUL_17_OUTPUT_0_MEM_WRAP_SIZE,
                            _63_MUL_17_OUTPUT_0_DEV_FMT>
                            (_63_Mul_17_output_0,
                            _63_MUL_17_OUTPUT_0_STREAM);
    fclose(_63_MUL_17_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _64_PadConv_12_output_0 = (float*) (mem_0 + _64_PADCONV_12_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_63_MUL_17_OUTPUT_0_IN_BATCH,
                           _64_PADCONV_12_NB_CHANNELS,
                           _63_MUL_17_OUTPUT_0_IN_HEIGHT,
                           _63_MUL_17_OUTPUT_0_IN_WIDTH,
                           _64_PADCONV_12_NB_OUTPUTS,
                           _64_PADCONV_12_OUTPUT_0_OUT_HEIGHT,
                           _64_PADCONV_12_OUTPUT_0_OUT_WIDTH,
                           _64_PADCONV_12_PADDING_Y,
                           _64_PADCONV_12_PADDING_X,
                           _64_PADCONV_12_STRIDE_Y,
                           _64_PADCONV_12_STRIDE_X,
                           _64_PADCONV_12_DILATION_Y,
                           _64_PADCONV_12_DILATION_X,
                           _64_PADCONV_12_KERNEL_HEIGHT,
                           _64_PADCONV_12_KERNEL_WIDTH,
                           _64_PADCONV_12_ACTIVATION,
                           _63_MUL_17_OUTPUT_0_MEM_CONT_OFFSET,
                           _63_MUL_17_OUTPUT_0_MEM_CONT_SIZE,
                           _63_MUL_17_OUTPUT_0_MEM_WRAP_OFFSET,
                           _63_MUL_17_OUTPUT_0_MEM_WRAP_SIZE,
                           _63_MUL_17_OUTPUT_0_MEM_STRIDE,
                           _64_PADCONV_12_OUTPUT_0_MEM_CONT_OFFSET,
                           _64_PADCONV_12_OUTPUT_0_MEM_CONT_SIZE,
                           _64_PADCONV_12_OUTPUT_0_MEM_WRAP_OFFSET,
                           _64_PADCONV_12_OUTPUT_0_MEM_WRAP_SIZE,
                           _64_PADCONV_12_OUTPUT_0_MEM_STRIDE>
                           (_63_Mul_17_output_0, _64_PadConv_12_output_0, _64_PadConv_12_weights_output_0, _64_PadConv_12_biases_output_0, _64_PADCONV_12_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _64_PadConv_12\n");
    FILE* _64_PADCONV_12_OUTPUT_0_STREAM = fopen("data/export_outputs/_64_PadConv_12_output_0.txt", "w");
    export_cpp::saveOutputs<_64_PADCONV_12_OUTPUT_0_NB_OUTPUTS,
                            _64_PADCONV_12_OUTPUT_0_OUT_HEIGHT,
                            _64_PADCONV_12_OUTPUT_0_OUT_WIDTH,
                            _64_PADCONV_12_OUTPUT_0_MEM_CONT_OFFSET,
                            _64_PADCONV_12_OUTPUT_0_MEM_CONT_SIZE,
                            _64_PADCONV_12_OUTPUT_0_MEM_WRAP_OFFSET,
                            _64_PADCONV_12_OUTPUT_0_MEM_WRAP_SIZE,
                            _64_PADCONV_12_OUTPUT_0_DEV_FMT>
                            (_64_PadConv_12_output_0,
                            _64_PADCONV_12_OUTPUT_0_STREAM);
    fclose(_64_PADCONV_12_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _65_Sigmoid_18_output_0 = (float*) (mem_0 + _65_SIGMOID_18_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_65_SIGMOID_18_NB_ELTS>
                               (_64_PadConv_12_output_0, _65_Sigmoid_18_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _65_Sigmoid_18\n");
    FILE* _65_SIGMOID_18_OUTPUT_0_STREAM = fopen("data/export_outputs/_65_Sigmoid_18_output_0.txt", "w");
    export_cpp::saveOutputs<_65_SIGMOID_18_OUTPUT_0_NB_OUTPUTS,
                            _65_SIGMOID_18_OUTPUT_0_OUT_HEIGHT,
                            _65_SIGMOID_18_OUTPUT_0_OUT_WIDTH,
                            _65_SIGMOID_18_OUTPUT_0_MEM_CONT_OFFSET,
                            _65_SIGMOID_18_OUTPUT_0_MEM_CONT_SIZE,
                            _65_SIGMOID_18_OUTPUT_0_MEM_WRAP_OFFSET,
                            _65_SIGMOID_18_OUTPUT_0_MEM_WRAP_SIZE,
                            _65_SIGMOID_18_OUTPUT_0_DEV_FMT>
                            (_65_Sigmoid_18_output_0,
                            _65_SIGMOID_18_OUTPUT_0_STREAM);
    fclose(_65_SIGMOID_18_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _66_Mul_18_output_0 = (float*) (mem_0 + _66_MUL_18_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_66_MUL_18_NB_MAT,
                                 _66_MUL_18_ELEM_OP,
                                 _66_MUL_18_INPUT1_CONT_SIZE,
                                 _66_MUL_18_INPUT2_CONT_SIZE,
                                 _66_MUL_18_OUTPUT_CONT_SIZE,
                                 _66_MUL_18_OFFSET_IN1,
                                 _66_MUL_18_OFFSET_IN2,
                                 _66_MUL_18_ACTIVATION,
                                 _64_PADCONV_12_OUTPUT_0_MEM_CONT_OFFSET,
                                 _64_PADCONV_12_OUTPUT_0_MEM_CONT_SIZE,
                                 _64_PADCONV_12_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _64_PADCONV_12_OUTPUT_0_MEM_WRAP_SIZE,
                                 _64_PADCONV_12_OUTPUT_0_MEM_STRIDE,
                                 _65_SIGMOID_18_OUTPUT_0_MEM_CONT_OFFSET,
                                 _65_SIGMOID_18_OUTPUT_0_MEM_CONT_SIZE,
                                 _65_SIGMOID_18_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _65_SIGMOID_18_OUTPUT_0_MEM_WRAP_SIZE,
                                 _65_SIGMOID_18_OUTPUT_0_MEM_STRIDE,
                                 _66_MUL_18_OUTPUT_0_MEM_CONT_OFFSET,
                                 _66_MUL_18_OUTPUT_0_MEM_CONT_SIZE,
                                 _66_MUL_18_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _66_MUL_18_OUTPUT_0_MEM_WRAP_SIZE,
                                 _66_MUL_18_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_66_Mul_18_output_0, _66_MUL_18_RESCALING, _64_PadConv_12_output_0, _65_Sigmoid_18_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _66_Mul_18\n");
    FILE* _66_MUL_18_OUTPUT_0_STREAM = fopen("data/export_outputs/_66_Mul_18_output_0.txt", "w");
    export_cpp::saveOutputs<_66_MUL_18_OUTPUT_0_NB_OUTPUTS,
                            _66_MUL_18_OUTPUT_0_OUT_HEIGHT,
                            _66_MUL_18_OUTPUT_0_OUT_WIDTH,
                            _66_MUL_18_OUTPUT_0_MEM_CONT_OFFSET,
                            _66_MUL_18_OUTPUT_0_MEM_CONT_SIZE,
                            _66_MUL_18_OUTPUT_0_MEM_WRAP_OFFSET,
                            _66_MUL_18_OUTPUT_0_MEM_WRAP_SIZE,
                            _66_MUL_18_OUTPUT_0_DEV_FMT>
                            (_66_Mul_18_output_0,
                            _66_MUL_18_OUTPUT_0_STREAM);
    fclose(_66_MUL_18_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _67_Add_4_output_0 = (float*) (mem_0 + _67_ADD_4_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_67_ADD_4_NB_MAT,
                                 _67_ADD_4_ELEM_OP,
                                 _67_ADD_4_INPUT1_CONT_SIZE,
                                 _67_ADD_4_INPUT2_CONT_SIZE,
                                 _67_ADD_4_OUTPUT_CONT_SIZE,
                                 _67_ADD_4_OFFSET_IN1,
                                 _67_ADD_4_OFFSET_IN2,
                                 _67_ADD_4_ACTIVATION,
                                 _60_ADD_3_OUTPUT_0_MEM_CONT_OFFSET,
                                 _60_ADD_3_OUTPUT_0_MEM_CONT_SIZE,
                                 _60_ADD_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _60_ADD_3_OUTPUT_0_MEM_WRAP_SIZE,
                                 _60_ADD_3_OUTPUT_0_MEM_STRIDE,
                                 _66_MUL_18_OUTPUT_0_MEM_CONT_OFFSET,
                                 _66_MUL_18_OUTPUT_0_MEM_CONT_SIZE,
                                 _66_MUL_18_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _66_MUL_18_OUTPUT_0_MEM_WRAP_SIZE,
                                 _66_MUL_18_OUTPUT_0_MEM_STRIDE,
                                 _67_ADD_4_OUTPUT_0_MEM_CONT_OFFSET,
                                 _67_ADD_4_OUTPUT_0_MEM_CONT_SIZE,
                                 _67_ADD_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _67_ADD_4_OUTPUT_0_MEM_WRAP_SIZE,
                                 _67_ADD_4_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_67_Add_4_output_0, _67_ADD_4_RESCALING, _60_Add_3_output_0, _66_Mul_18_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _67_Add_4\n");
    FILE* _67_ADD_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_67_Add_4_output_0.txt", "w");
    export_cpp::saveOutputs<_67_ADD_4_OUTPUT_0_NB_OUTPUTS,
                            _67_ADD_4_OUTPUT_0_OUT_HEIGHT,
                            _67_ADD_4_OUTPUT_0_OUT_WIDTH,
                            _67_ADD_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _67_ADD_4_OUTPUT_0_MEM_CONT_SIZE,
                            _67_ADD_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _67_ADD_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _67_ADD_4_OUTPUT_0_DEV_FMT>
                            (_67_Add_4_output_0,
                            _67_ADD_4_OUTPUT_0_STREAM);
    fclose(_67_ADD_4_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _68_Concat_2_output_0 = (float*) (mem_0 + _68_CONCAT_2_OUTPUT_0_MEM_OFFSET);

    const float* _68_Concat_2_inputs[] = { _53_Split_2_output_0, _53_Split_2_output_1, _60_Add_3_output_0, _67_Add_4_output_0 };
    export_cpp::concat_forward<_68_CONCAT_2_AXIS_SIZE_POST,
                               _68_CONCAT_2_AXIS_SIZE_PRE,
                               _68_CONCAT_2_AXIS_SIZE,
                               _68_CONCAT_2_TOTAL_AXIS_SIZE,
                               4> 
                              (_68_Concat_2_inputs,
                               _68_Concat_2_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _68_Concat_2\n");
    FILE* _68_CONCAT_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_68_Concat_2_output_0.txt", "w");
    export_cpp::saveOutputs<_68_CONCAT_2_OUTPUT_0_NB_OUTPUTS,
                            _68_CONCAT_2_OUTPUT_0_OUT_HEIGHT,
                            _68_CONCAT_2_OUTPUT_0_OUT_WIDTH,
                            _68_CONCAT_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _68_CONCAT_2_OUTPUT_0_MEM_CONT_SIZE,
                            _68_CONCAT_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _68_CONCAT_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _68_CONCAT_2_OUTPUT_0_DEV_FMT>
                            (_68_Concat_2_output_0,
                            _68_CONCAT_2_OUTPUT_0_STREAM);
    fclose(_68_CONCAT_2_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _69_Conv2D_6_output_0 = (float*) (mem_0 + _69_CONV2D_6_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_68_CONCAT_2_OUTPUT_0_IN_BATCH,
                           _69_CONV2D_6_NB_CHANNELS,
                           _68_CONCAT_2_OUTPUT_0_IN_HEIGHT,
                           _68_CONCAT_2_OUTPUT_0_IN_WIDTH,
                           _69_CONV2D_6_NB_OUTPUTS,
                           _69_CONV2D_6_OUTPUT_0_OUT_HEIGHT,
                           _69_CONV2D_6_OUTPUT_0_OUT_WIDTH,
                           _69_CONV2D_6_PADDING_Y,
                           _69_CONV2D_6_PADDING_X,
                           _69_CONV2D_6_STRIDE_Y,
                           _69_CONV2D_6_STRIDE_X,
                           _69_CONV2D_6_DILATION_Y,
                           _69_CONV2D_6_DILATION_X,
                           _69_CONV2D_6_KERNEL_HEIGHT,
                           _69_CONV2D_6_KERNEL_WIDTH,
                           _69_CONV2D_6_ACTIVATION,
                           _68_CONCAT_2_OUTPUT_0_MEM_CONT_OFFSET,
                           _68_CONCAT_2_OUTPUT_0_MEM_CONT_SIZE,
                           _68_CONCAT_2_OUTPUT_0_MEM_WRAP_OFFSET,
                           _68_CONCAT_2_OUTPUT_0_MEM_WRAP_SIZE,
                           _68_CONCAT_2_OUTPUT_0_MEM_STRIDE,
                           _69_CONV2D_6_OUTPUT_0_MEM_CONT_OFFSET,
                           _69_CONV2D_6_OUTPUT_0_MEM_CONT_SIZE,
                           _69_CONV2D_6_OUTPUT_0_MEM_WRAP_OFFSET,
                           _69_CONV2D_6_OUTPUT_0_MEM_WRAP_SIZE,
                           _69_CONV2D_6_OUTPUT_0_MEM_STRIDE>
                           (_68_Concat_2_output_0, _69_Conv2D_6_output_0, _69_Conv2D_6_weights_output_0, _69_Conv2D_6_biases_output_0, _69_CONV2D_6_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _69_Conv2D_6\n");
    FILE* _69_CONV2D_6_OUTPUT_0_STREAM = fopen("data/export_outputs/_69_Conv2D_6_output_0.txt", "w");
    export_cpp::saveOutputs<_69_CONV2D_6_OUTPUT_0_NB_OUTPUTS,
                            _69_CONV2D_6_OUTPUT_0_OUT_HEIGHT,
                            _69_CONV2D_6_OUTPUT_0_OUT_WIDTH,
                            _69_CONV2D_6_OUTPUT_0_MEM_CONT_OFFSET,
                            _69_CONV2D_6_OUTPUT_0_MEM_CONT_SIZE,
                            _69_CONV2D_6_OUTPUT_0_MEM_WRAP_OFFSET,
                            _69_CONV2D_6_OUTPUT_0_MEM_WRAP_SIZE,
                            _69_CONV2D_6_OUTPUT_0_DEV_FMT>
                            (_69_Conv2D_6_output_0,
                            _69_CONV2D_6_OUTPUT_0_STREAM);
    fclose(_69_CONV2D_6_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _70_Sigmoid_19_output_0 = (float*) (mem_0 + _70_SIGMOID_19_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_70_SIGMOID_19_NB_ELTS>
                               (_69_Conv2D_6_output_0, _70_Sigmoid_19_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _70_Sigmoid_19\n");
    FILE* _70_SIGMOID_19_OUTPUT_0_STREAM = fopen("data/export_outputs/_70_Sigmoid_19_output_0.txt", "w");
    export_cpp::saveOutputs<_70_SIGMOID_19_OUTPUT_0_NB_OUTPUTS,
                            _70_SIGMOID_19_OUTPUT_0_OUT_HEIGHT,
                            _70_SIGMOID_19_OUTPUT_0_OUT_WIDTH,
                            _70_SIGMOID_19_OUTPUT_0_MEM_CONT_OFFSET,
                            _70_SIGMOID_19_OUTPUT_0_MEM_CONT_SIZE,
                            _70_SIGMOID_19_OUTPUT_0_MEM_WRAP_OFFSET,
                            _70_SIGMOID_19_OUTPUT_0_MEM_WRAP_SIZE,
                            _70_SIGMOID_19_OUTPUT_0_DEV_FMT>
                            (_70_Sigmoid_19_output_0,
                            _70_SIGMOID_19_OUTPUT_0_STREAM);
    fclose(_70_SIGMOID_19_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _71_Mul_19_output_0 = (float*) (mem_0 + _71_MUL_19_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_71_MUL_19_NB_MAT,
                                 _71_MUL_19_ELEM_OP,
                                 _71_MUL_19_INPUT1_CONT_SIZE,
                                 _71_MUL_19_INPUT2_CONT_SIZE,
                                 _71_MUL_19_OUTPUT_CONT_SIZE,
                                 _71_MUL_19_OFFSET_IN1,
                                 _71_MUL_19_OFFSET_IN2,
                                 _71_MUL_19_ACTIVATION,
                                 _69_CONV2D_6_OUTPUT_0_MEM_CONT_OFFSET,
                                 _69_CONV2D_6_OUTPUT_0_MEM_CONT_SIZE,
                                 _69_CONV2D_6_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _69_CONV2D_6_OUTPUT_0_MEM_WRAP_SIZE,
                                 _69_CONV2D_6_OUTPUT_0_MEM_STRIDE,
                                 _70_SIGMOID_19_OUTPUT_0_MEM_CONT_OFFSET,
                                 _70_SIGMOID_19_OUTPUT_0_MEM_CONT_SIZE,
                                 _70_SIGMOID_19_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _70_SIGMOID_19_OUTPUT_0_MEM_WRAP_SIZE,
                                 _70_SIGMOID_19_OUTPUT_0_MEM_STRIDE,
                                 _71_MUL_19_OUTPUT_0_MEM_CONT_OFFSET,
                                 _71_MUL_19_OUTPUT_0_MEM_CONT_SIZE,
                                 _71_MUL_19_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _71_MUL_19_OUTPUT_0_MEM_WRAP_SIZE,
                                 _71_MUL_19_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_71_Mul_19_output_0, _71_MUL_19_RESCALING, _69_Conv2D_6_output_0, _70_Sigmoid_19_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _71_Mul_19\n");
    FILE* _71_MUL_19_OUTPUT_0_STREAM = fopen("data/export_outputs/_71_Mul_19_output_0.txt", "w");
    export_cpp::saveOutputs<_71_MUL_19_OUTPUT_0_NB_OUTPUTS,
                            _71_MUL_19_OUTPUT_0_OUT_HEIGHT,
                            _71_MUL_19_OUTPUT_0_OUT_WIDTH,
                            _71_MUL_19_OUTPUT_0_MEM_CONT_OFFSET,
                            _71_MUL_19_OUTPUT_0_MEM_CONT_SIZE,
                            _71_MUL_19_OUTPUT_0_MEM_WRAP_OFFSET,
                            _71_MUL_19_OUTPUT_0_MEM_WRAP_SIZE,
                            _71_MUL_19_OUTPUT_0_DEV_FMT>
                            (_71_Mul_19_output_0,
                            _71_MUL_19_OUTPUT_0_STREAM);
    fclose(_71_MUL_19_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _72_Conv2D_7_output_0 = (float*) (mem_0 + _72_CONV2D_7_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_71_MUL_19_OUTPUT_0_IN_BATCH,
                           _72_CONV2D_7_NB_CHANNELS,
                           _71_MUL_19_OUTPUT_0_IN_HEIGHT,
                           _71_MUL_19_OUTPUT_0_IN_WIDTH,
                           _72_CONV2D_7_NB_OUTPUTS,
                           _72_CONV2D_7_OUTPUT_0_OUT_HEIGHT,
                           _72_CONV2D_7_OUTPUT_0_OUT_WIDTH,
                           _72_CONV2D_7_PADDING_Y,
                           _72_CONV2D_7_PADDING_X,
                           _72_CONV2D_7_STRIDE_Y,
                           _72_CONV2D_7_STRIDE_X,
                           _72_CONV2D_7_DILATION_Y,
                           _72_CONV2D_7_DILATION_X,
                           _72_CONV2D_7_KERNEL_HEIGHT,
                           _72_CONV2D_7_KERNEL_WIDTH,
                           _72_CONV2D_7_ACTIVATION,
                           _71_MUL_19_OUTPUT_0_MEM_CONT_OFFSET,
                           _71_MUL_19_OUTPUT_0_MEM_CONT_SIZE,
                           _71_MUL_19_OUTPUT_0_MEM_WRAP_OFFSET,
                           _71_MUL_19_OUTPUT_0_MEM_WRAP_SIZE,
                           _71_MUL_19_OUTPUT_0_MEM_STRIDE,
                           _72_CONV2D_7_OUTPUT_0_MEM_CONT_OFFSET,
                           _72_CONV2D_7_OUTPUT_0_MEM_CONT_SIZE,
                           _72_CONV2D_7_OUTPUT_0_MEM_WRAP_OFFSET,
                           _72_CONV2D_7_OUTPUT_0_MEM_WRAP_SIZE,
                           _72_CONV2D_7_OUTPUT_0_MEM_STRIDE>
                           (_71_Mul_19_output_0, _72_Conv2D_7_output_0, _72_Conv2D_7_weights_output_0, _72_Conv2D_7_biases_output_0, _72_CONV2D_7_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _72_Conv2D_7\n");
    FILE* _72_CONV2D_7_OUTPUT_0_STREAM = fopen("data/export_outputs/_72_Conv2D_7_output_0.txt", "w");
    export_cpp::saveOutputs<_72_CONV2D_7_OUTPUT_0_NB_OUTPUTS,
                            _72_CONV2D_7_OUTPUT_0_OUT_HEIGHT,
                            _72_CONV2D_7_OUTPUT_0_OUT_WIDTH,
                            _72_CONV2D_7_OUTPUT_0_MEM_CONT_OFFSET,
                            _72_CONV2D_7_OUTPUT_0_MEM_CONT_SIZE,
                            _72_CONV2D_7_OUTPUT_0_MEM_WRAP_OFFSET,
                            _72_CONV2D_7_OUTPUT_0_MEM_WRAP_SIZE,
                            _72_CONV2D_7_OUTPUT_0_DEV_FMT>
                            (_72_Conv2D_7_output_0,
                            _72_CONV2D_7_OUTPUT_0_STREAM);
    fclose(_72_CONV2D_7_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _73_Sigmoid_20_output_0 = (float*) (mem_0 + _73_SIGMOID_20_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_73_SIGMOID_20_NB_ELTS>
                               (_72_Conv2D_7_output_0, _73_Sigmoid_20_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _73_Sigmoid_20\n");
    FILE* _73_SIGMOID_20_OUTPUT_0_STREAM = fopen("data/export_outputs/_73_Sigmoid_20_output_0.txt", "w");
    export_cpp::saveOutputs<_73_SIGMOID_20_OUTPUT_0_NB_OUTPUTS,
                            _73_SIGMOID_20_OUTPUT_0_OUT_HEIGHT,
                            _73_SIGMOID_20_OUTPUT_0_OUT_WIDTH,
                            _73_SIGMOID_20_OUTPUT_0_MEM_CONT_OFFSET,
                            _73_SIGMOID_20_OUTPUT_0_MEM_CONT_SIZE,
                            _73_SIGMOID_20_OUTPUT_0_MEM_WRAP_OFFSET,
                            _73_SIGMOID_20_OUTPUT_0_MEM_WRAP_SIZE,
                            _73_SIGMOID_20_OUTPUT_0_DEV_FMT>
                            (_73_Sigmoid_20_output_0,
                            _73_SIGMOID_20_OUTPUT_0_STREAM);
    fclose(_73_SIGMOID_20_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _74_Mul_20_output_0 = (float*) (mem_0 + _74_MUL_20_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_74_MUL_20_NB_MAT,
                                 _74_MUL_20_ELEM_OP,
                                 _74_MUL_20_INPUT1_CONT_SIZE,
                                 _74_MUL_20_INPUT2_CONT_SIZE,
                                 _74_MUL_20_OUTPUT_CONT_SIZE,
                                 _74_MUL_20_OFFSET_IN1,
                                 _74_MUL_20_OFFSET_IN2,
                                 _74_MUL_20_ACTIVATION,
                                 _72_CONV2D_7_OUTPUT_0_MEM_CONT_OFFSET,
                                 _72_CONV2D_7_OUTPUT_0_MEM_CONT_SIZE,
                                 _72_CONV2D_7_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _72_CONV2D_7_OUTPUT_0_MEM_WRAP_SIZE,
                                 _72_CONV2D_7_OUTPUT_0_MEM_STRIDE,
                                 _73_SIGMOID_20_OUTPUT_0_MEM_CONT_OFFSET,
                                 _73_SIGMOID_20_OUTPUT_0_MEM_CONT_SIZE,
                                 _73_SIGMOID_20_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _73_SIGMOID_20_OUTPUT_0_MEM_WRAP_SIZE,
                                 _73_SIGMOID_20_OUTPUT_0_MEM_STRIDE,
                                 _74_MUL_20_OUTPUT_0_MEM_CONT_OFFSET,
                                 _74_MUL_20_OUTPUT_0_MEM_CONT_SIZE,
                                 _74_MUL_20_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _74_MUL_20_OUTPUT_0_MEM_WRAP_SIZE,
                                 _74_MUL_20_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_74_Mul_20_output_0, _74_MUL_20_RESCALING, _72_Conv2D_7_output_0, _73_Sigmoid_20_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _74_Mul_20\n");
    FILE* _74_MUL_20_OUTPUT_0_STREAM = fopen("data/export_outputs/_74_Mul_20_output_0.txt", "w");
    export_cpp::saveOutputs<_74_MUL_20_OUTPUT_0_NB_OUTPUTS,
                            _74_MUL_20_OUTPUT_0_OUT_HEIGHT,
                            _74_MUL_20_OUTPUT_0_OUT_WIDTH,
                            _74_MUL_20_OUTPUT_0_MEM_CONT_OFFSET,
                            _74_MUL_20_OUTPUT_0_MEM_CONT_SIZE,
                            _74_MUL_20_OUTPUT_0_MEM_WRAP_OFFSET,
                            _74_MUL_20_OUTPUT_0_MEM_WRAP_SIZE,
                            _74_MUL_20_OUTPUT_0_DEV_FMT>
                            (_74_Mul_20_output_0,
                            _74_MUL_20_OUTPUT_0_STREAM);
    fclose(_74_MUL_20_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _75_PadConvDw_1_output_0 = (float*) (mem_0 + _75_PADCONVDW_1_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_74_MUL_20_OUTPUT_0_IN_BATCH,
                           _75_PADCONVDW_1_NB_CHANNELS,
                           _74_MUL_20_OUTPUT_0_IN_HEIGHT,
                           _74_MUL_20_OUTPUT_0_IN_WIDTH,
                           _75_PADCONVDW_1_NB_OUTPUTS,
                           _75_PADCONVDW_1_OUTPUT_0_OUT_HEIGHT,
                           _75_PADCONVDW_1_OUTPUT_0_OUT_WIDTH,
                           _75_PADCONVDW_1_PADDING_Y,
                           _75_PADCONVDW_1_PADDING_X,
                           _75_PADCONVDW_1_STRIDE_Y,
                           _75_PADCONVDW_1_STRIDE_X,
                           _75_PADCONVDW_1_DILATION_Y,
                           _75_PADCONVDW_1_DILATION_X,
                           _75_PADCONVDW_1_KERNEL_HEIGHT,
                           _75_PADCONVDW_1_KERNEL_WIDTH,
                           _75_PADCONVDW_1_ACTIVATION,
                           _74_MUL_20_OUTPUT_0_MEM_CONT_OFFSET,
                           _74_MUL_20_OUTPUT_0_MEM_CONT_SIZE,
                           _74_MUL_20_OUTPUT_0_MEM_WRAP_OFFSET,
                           _74_MUL_20_OUTPUT_0_MEM_WRAP_SIZE,
                           _74_MUL_20_OUTPUT_0_MEM_STRIDE,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_CONT_OFFSET,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_CONT_SIZE,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_WRAP_OFFSET,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_WRAP_SIZE,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_STRIDE>
                           (_74_Mul_20_output_0, _75_PadConvDw_1_output_0, _75_PadConvDw_1_weights_output_0, _75_PadConvDw_1_biases_output_0, _75_PADCONVDW_1_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _75_PadConvDw_1\n");
    FILE* _75_PADCONVDW_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_75_PadConvDw_1_output_0.txt", "w");
    export_cpp::saveOutputs<_75_PADCONVDW_1_OUTPUT_0_NB_OUTPUTS,
                            _75_PADCONVDW_1_OUTPUT_0_OUT_HEIGHT,
                            _75_PADCONVDW_1_OUTPUT_0_OUT_WIDTH,
                            _75_PADCONVDW_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _75_PADCONVDW_1_OUTPUT_0_MEM_CONT_SIZE,
                            _75_PADCONVDW_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _75_PADCONVDW_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _75_PADCONVDW_1_OUTPUT_0_DEV_FMT>
                            (_75_PadConvDw_1_output_0,
                            _75_PADCONVDW_1_OUTPUT_0_STREAM);
    fclose(_75_PADCONVDW_1_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _76_Conv2D_8_output_0 = (float*) (mem_0 + _76_CONV2D_8_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_75_PADCONVDW_1_OUTPUT_0_IN_BATCH,
                           _76_CONV2D_8_NB_CHANNELS,
                           _75_PADCONVDW_1_OUTPUT_0_IN_HEIGHT,
                           _75_PADCONVDW_1_OUTPUT_0_IN_WIDTH,
                           _76_CONV2D_8_NB_OUTPUTS,
                           _76_CONV2D_8_OUTPUT_0_OUT_HEIGHT,
                           _76_CONV2D_8_OUTPUT_0_OUT_WIDTH,
                           _76_CONV2D_8_PADDING_Y,
                           _76_CONV2D_8_PADDING_X,
                           _76_CONV2D_8_STRIDE_Y,
                           _76_CONV2D_8_STRIDE_X,
                           _76_CONV2D_8_DILATION_Y,
                           _76_CONV2D_8_DILATION_X,
                           _76_CONV2D_8_KERNEL_HEIGHT,
                           _76_CONV2D_8_KERNEL_WIDTH,
                           _76_CONV2D_8_ACTIVATION,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_CONT_OFFSET,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_CONT_SIZE,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_WRAP_OFFSET,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_WRAP_SIZE,
                           _75_PADCONVDW_1_OUTPUT_0_MEM_STRIDE,
                           _76_CONV2D_8_OUTPUT_0_MEM_CONT_OFFSET,
                           _76_CONV2D_8_OUTPUT_0_MEM_CONT_SIZE,
                           _76_CONV2D_8_OUTPUT_0_MEM_WRAP_OFFSET,
                           _76_CONV2D_8_OUTPUT_0_MEM_WRAP_SIZE,
                           _76_CONV2D_8_OUTPUT_0_MEM_STRIDE>
                           (_75_PadConvDw_1_output_0, _76_Conv2D_8_output_0, _76_Conv2D_8_weights_output_0, _76_Conv2D_8_biases_output_0, _76_CONV2D_8_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _76_Conv2D_8\n");
    FILE* _76_CONV2D_8_OUTPUT_0_STREAM = fopen("data/export_outputs/_76_Conv2D_8_output_0.txt", "w");
    export_cpp::saveOutputs<_76_CONV2D_8_OUTPUT_0_NB_OUTPUTS,
                            _76_CONV2D_8_OUTPUT_0_OUT_HEIGHT,
                            _76_CONV2D_8_OUTPUT_0_OUT_WIDTH,
                            _76_CONV2D_8_OUTPUT_0_MEM_CONT_OFFSET,
                            _76_CONV2D_8_OUTPUT_0_MEM_CONT_SIZE,
                            _76_CONV2D_8_OUTPUT_0_MEM_WRAP_OFFSET,
                            _76_CONV2D_8_OUTPUT_0_MEM_WRAP_SIZE,
                            _76_CONV2D_8_OUTPUT_0_DEV_FMT>
                            (_76_Conv2D_8_output_0,
                            _76_CONV2D_8_OUTPUT_0_STREAM);
    fclose(_76_CONV2D_8_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _77_Sigmoid_21_output_0 = (float*) (mem_0 + _77_SIGMOID_21_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_77_SIGMOID_21_NB_ELTS>
                               (_76_Conv2D_8_output_0, _77_Sigmoid_21_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _77_Sigmoid_21\n");
    FILE* _77_SIGMOID_21_OUTPUT_0_STREAM = fopen("data/export_outputs/_77_Sigmoid_21_output_0.txt", "w");
    export_cpp::saveOutputs<_77_SIGMOID_21_OUTPUT_0_NB_OUTPUTS,
                            _77_SIGMOID_21_OUTPUT_0_OUT_HEIGHT,
                            _77_SIGMOID_21_OUTPUT_0_OUT_WIDTH,
                            _77_SIGMOID_21_OUTPUT_0_MEM_CONT_OFFSET,
                            _77_SIGMOID_21_OUTPUT_0_MEM_CONT_SIZE,
                            _77_SIGMOID_21_OUTPUT_0_MEM_WRAP_OFFSET,
                            _77_SIGMOID_21_OUTPUT_0_MEM_WRAP_SIZE,
                            _77_SIGMOID_21_OUTPUT_0_DEV_FMT>
                            (_77_Sigmoid_21_output_0,
                            _77_SIGMOID_21_OUTPUT_0_STREAM);
    fclose(_77_SIGMOID_21_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _78_Mul_21_output_0 = (float*) (mem_0 + _78_MUL_21_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_78_MUL_21_NB_MAT,
                                 _78_MUL_21_ELEM_OP,
                                 _78_MUL_21_INPUT1_CONT_SIZE,
                                 _78_MUL_21_INPUT2_CONT_SIZE,
                                 _78_MUL_21_OUTPUT_CONT_SIZE,
                                 _78_MUL_21_OFFSET_IN1,
                                 _78_MUL_21_OFFSET_IN2,
                                 _78_MUL_21_ACTIVATION,
                                 _76_CONV2D_8_OUTPUT_0_MEM_CONT_OFFSET,
                                 _76_CONV2D_8_OUTPUT_0_MEM_CONT_SIZE,
                                 _76_CONV2D_8_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _76_CONV2D_8_OUTPUT_0_MEM_WRAP_SIZE,
                                 _76_CONV2D_8_OUTPUT_0_MEM_STRIDE,
                                 _77_SIGMOID_21_OUTPUT_0_MEM_CONT_OFFSET,
                                 _77_SIGMOID_21_OUTPUT_0_MEM_CONT_SIZE,
                                 _77_SIGMOID_21_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _77_SIGMOID_21_OUTPUT_0_MEM_WRAP_SIZE,
                                 _77_SIGMOID_21_OUTPUT_0_MEM_STRIDE,
                                 _78_MUL_21_OUTPUT_0_MEM_CONT_OFFSET,
                                 _78_MUL_21_OUTPUT_0_MEM_CONT_SIZE,
                                 _78_MUL_21_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _78_MUL_21_OUTPUT_0_MEM_WRAP_SIZE,
                                 _78_MUL_21_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_78_Mul_21_output_0, _78_MUL_21_RESCALING, _76_Conv2D_8_output_0, _77_Sigmoid_21_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _78_Mul_21\n");
    FILE* _78_MUL_21_OUTPUT_0_STREAM = fopen("data/export_outputs/_78_Mul_21_output_0.txt", "w");
    export_cpp::saveOutputs<_78_MUL_21_OUTPUT_0_NB_OUTPUTS,
                            _78_MUL_21_OUTPUT_0_OUT_HEIGHT,
                            _78_MUL_21_OUTPUT_0_OUT_WIDTH,
                            _78_MUL_21_OUTPUT_0_MEM_CONT_OFFSET,
                            _78_MUL_21_OUTPUT_0_MEM_CONT_SIZE,
                            _78_MUL_21_OUTPUT_0_MEM_WRAP_OFFSET,
                            _78_MUL_21_OUTPUT_0_MEM_WRAP_SIZE,
                            _78_MUL_21_OUTPUT_0_DEV_FMT>
                            (_78_Mul_21_output_0,
                            _78_MUL_21_OUTPUT_0_STREAM);
    fclose(_78_MUL_21_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _79_Split_3_output_0 = (float*) (mem_0 + _79_SPLIT_3_OUTPUT_0_MEM_OFFSET);
    float* _79_Split_3_output_1 = (float*) (mem_0 + _79_SPLIT_3_OUTPUT_1_MEM_OFFSET);



    float* __79_Split_3_outputs[_79_SPLIT_3_NB_OUTPUTS] = {
        
        _79_Split_3_output_0,
        
        _79_Split_3_output_1
        
    };

    export_cpp::split_forward<_79_SPLIT_3_NB_OUTPUTS,
                              _79_SPLIT_3_AXIS_DIM,
                              _79_SPLIT_3_PRE_AXIS_ELEMS,
                              _79_SPLIT_3_POST_AXIS_ELEMS,
                              _79_SPLIT_3_SPLITS,
                              float>
                              (_78_Mul_21_output_0,
                               __79_Split_3_outputs);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _79_Split_3\n");
    FILE* _79_SPLIT_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_79_Split_3_output_0.txt", "w");
    export_cpp::saveOutputs<_79_SPLIT_3_OUTPUT_0_NB_OUTPUTS,
                            _79_SPLIT_3_OUTPUT_0_OUT_HEIGHT,
                            _79_SPLIT_3_OUTPUT_0_OUT_WIDTH,
                            _79_SPLIT_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _79_SPLIT_3_OUTPUT_0_MEM_CONT_SIZE,
                            _79_SPLIT_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _79_SPLIT_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _79_SPLIT_3_OUTPUT_0_DEV_FMT>
                            (_79_Split_3_output_0,
                            _79_SPLIT_3_OUTPUT_0_STREAM);
    fclose(_79_SPLIT_3_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _80_PadConv_13_output_0 = (float*) (mem_0 + _80_PADCONV_13_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_79_SPLIT_3_OUTPUT_1_IN_BATCH,
                           _80_PADCONV_13_NB_CHANNELS,
                           _79_SPLIT_3_OUTPUT_1_IN_HEIGHT,
                           _79_SPLIT_3_OUTPUT_1_IN_WIDTH,
                           _80_PADCONV_13_NB_OUTPUTS,
                           _80_PADCONV_13_OUTPUT_0_OUT_HEIGHT,
                           _80_PADCONV_13_OUTPUT_0_OUT_WIDTH,
                           _80_PADCONV_13_PADDING_Y,
                           _80_PADCONV_13_PADDING_X,
                           _80_PADCONV_13_STRIDE_Y,
                           _80_PADCONV_13_STRIDE_X,
                           _80_PADCONV_13_DILATION_Y,
                           _80_PADCONV_13_DILATION_X,
                           _80_PADCONV_13_KERNEL_HEIGHT,
                           _80_PADCONV_13_KERNEL_WIDTH,
                           _80_PADCONV_13_ACTIVATION,
                           _79_SPLIT_3_OUTPUT_1_MEM_CONT_OFFSET,
                           _79_SPLIT_3_OUTPUT_1_MEM_CONT_SIZE,
                           _79_SPLIT_3_OUTPUT_1_MEM_WRAP_OFFSET,
                           _79_SPLIT_3_OUTPUT_1_MEM_WRAP_SIZE,
                           _79_SPLIT_3_OUTPUT_1_MEM_STRIDE,
                           _80_PADCONV_13_OUTPUT_0_MEM_CONT_OFFSET,
                           _80_PADCONV_13_OUTPUT_0_MEM_CONT_SIZE,
                           _80_PADCONV_13_OUTPUT_0_MEM_WRAP_OFFSET,
                           _80_PADCONV_13_OUTPUT_0_MEM_WRAP_SIZE,
                           _80_PADCONV_13_OUTPUT_0_MEM_STRIDE>
                           (_79_Split_3_output_1, _80_PadConv_13_output_0, _80_PadConv_13_weights_output_0, _80_PadConv_13_biases_output_0, _80_PADCONV_13_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _80_PadConv_13\n");
    FILE* _80_PADCONV_13_OUTPUT_0_STREAM = fopen("data/export_outputs/_80_PadConv_13_output_0.txt", "w");
    export_cpp::saveOutputs<_80_PADCONV_13_OUTPUT_0_NB_OUTPUTS,
                            _80_PADCONV_13_OUTPUT_0_OUT_HEIGHT,
                            _80_PADCONV_13_OUTPUT_0_OUT_WIDTH,
                            _80_PADCONV_13_OUTPUT_0_MEM_CONT_OFFSET,
                            _80_PADCONV_13_OUTPUT_0_MEM_CONT_SIZE,
                            _80_PADCONV_13_OUTPUT_0_MEM_WRAP_OFFSET,
                            _80_PADCONV_13_OUTPUT_0_MEM_WRAP_SIZE,
                            _80_PADCONV_13_OUTPUT_0_DEV_FMT>
                            (_80_PadConv_13_output_0,
                            _80_PADCONV_13_OUTPUT_0_STREAM);
    fclose(_80_PADCONV_13_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _81_Sigmoid_22_output_0 = (float*) (mem_0 + _81_SIGMOID_22_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_81_SIGMOID_22_NB_ELTS>
                               (_80_PadConv_13_output_0, _81_Sigmoid_22_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _81_Sigmoid_22\n");
    FILE* _81_SIGMOID_22_OUTPUT_0_STREAM = fopen("data/export_outputs/_81_Sigmoid_22_output_0.txt", "w");
    export_cpp::saveOutputs<_81_SIGMOID_22_OUTPUT_0_NB_OUTPUTS,
                            _81_SIGMOID_22_OUTPUT_0_OUT_HEIGHT,
                            _81_SIGMOID_22_OUTPUT_0_OUT_WIDTH,
                            _81_SIGMOID_22_OUTPUT_0_MEM_CONT_OFFSET,
                            _81_SIGMOID_22_OUTPUT_0_MEM_CONT_SIZE,
                            _81_SIGMOID_22_OUTPUT_0_MEM_WRAP_OFFSET,
                            _81_SIGMOID_22_OUTPUT_0_MEM_WRAP_SIZE,
                            _81_SIGMOID_22_OUTPUT_0_DEV_FMT>
                            (_81_Sigmoid_22_output_0,
                            _81_SIGMOID_22_OUTPUT_0_STREAM);
    fclose(_81_SIGMOID_22_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _82_Mul_22_output_0 = (float*) (mem_0 + _82_MUL_22_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_82_MUL_22_NB_MAT,
                                 _82_MUL_22_ELEM_OP,
                                 _82_MUL_22_INPUT1_CONT_SIZE,
                                 _82_MUL_22_INPUT2_CONT_SIZE,
                                 _82_MUL_22_OUTPUT_CONT_SIZE,
                                 _82_MUL_22_OFFSET_IN1,
                                 _82_MUL_22_OFFSET_IN2,
                                 _82_MUL_22_ACTIVATION,
                                 _80_PADCONV_13_OUTPUT_0_MEM_CONT_OFFSET,
                                 _80_PADCONV_13_OUTPUT_0_MEM_CONT_SIZE,
                                 _80_PADCONV_13_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _80_PADCONV_13_OUTPUT_0_MEM_WRAP_SIZE,
                                 _80_PADCONV_13_OUTPUT_0_MEM_STRIDE,
                                 _81_SIGMOID_22_OUTPUT_0_MEM_CONT_OFFSET,
                                 _81_SIGMOID_22_OUTPUT_0_MEM_CONT_SIZE,
                                 _81_SIGMOID_22_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _81_SIGMOID_22_OUTPUT_0_MEM_WRAP_SIZE,
                                 _81_SIGMOID_22_OUTPUT_0_MEM_STRIDE,
                                 _82_MUL_22_OUTPUT_0_MEM_CONT_OFFSET,
                                 _82_MUL_22_OUTPUT_0_MEM_CONT_SIZE,
                                 _82_MUL_22_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _82_MUL_22_OUTPUT_0_MEM_WRAP_SIZE,
                                 _82_MUL_22_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_82_Mul_22_output_0, _82_MUL_22_RESCALING, _80_PadConv_13_output_0, _81_Sigmoid_22_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _82_Mul_22\n");
    FILE* _82_MUL_22_OUTPUT_0_STREAM = fopen("data/export_outputs/_82_Mul_22_output_0.txt", "w");
    export_cpp::saveOutputs<_82_MUL_22_OUTPUT_0_NB_OUTPUTS,
                            _82_MUL_22_OUTPUT_0_OUT_HEIGHT,
                            _82_MUL_22_OUTPUT_0_OUT_WIDTH,
                            _82_MUL_22_OUTPUT_0_MEM_CONT_OFFSET,
                            _82_MUL_22_OUTPUT_0_MEM_CONT_SIZE,
                            _82_MUL_22_OUTPUT_0_MEM_WRAP_OFFSET,
                            _82_MUL_22_OUTPUT_0_MEM_WRAP_SIZE,
                            _82_MUL_22_OUTPUT_0_DEV_FMT>
                            (_82_Mul_22_output_0,
                            _82_MUL_22_OUTPUT_0_STREAM);
    fclose(_82_MUL_22_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _83_PadConv_14_output_0 = (float*) (mem_0 + _83_PADCONV_14_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_82_MUL_22_OUTPUT_0_IN_BATCH,
                           _83_PADCONV_14_NB_CHANNELS,
                           _82_MUL_22_OUTPUT_0_IN_HEIGHT,
                           _82_MUL_22_OUTPUT_0_IN_WIDTH,
                           _83_PADCONV_14_NB_OUTPUTS,
                           _83_PADCONV_14_OUTPUT_0_OUT_HEIGHT,
                           _83_PADCONV_14_OUTPUT_0_OUT_WIDTH,
                           _83_PADCONV_14_PADDING_Y,
                           _83_PADCONV_14_PADDING_X,
                           _83_PADCONV_14_STRIDE_Y,
                           _83_PADCONV_14_STRIDE_X,
                           _83_PADCONV_14_DILATION_Y,
                           _83_PADCONV_14_DILATION_X,
                           _83_PADCONV_14_KERNEL_HEIGHT,
                           _83_PADCONV_14_KERNEL_WIDTH,
                           _83_PADCONV_14_ACTIVATION,
                           _82_MUL_22_OUTPUT_0_MEM_CONT_OFFSET,
                           _82_MUL_22_OUTPUT_0_MEM_CONT_SIZE,
                           _82_MUL_22_OUTPUT_0_MEM_WRAP_OFFSET,
                           _82_MUL_22_OUTPUT_0_MEM_WRAP_SIZE,
                           _82_MUL_22_OUTPUT_0_MEM_STRIDE,
                           _83_PADCONV_14_OUTPUT_0_MEM_CONT_OFFSET,
                           _83_PADCONV_14_OUTPUT_0_MEM_CONT_SIZE,
                           _83_PADCONV_14_OUTPUT_0_MEM_WRAP_OFFSET,
                           _83_PADCONV_14_OUTPUT_0_MEM_WRAP_SIZE,
                           _83_PADCONV_14_OUTPUT_0_MEM_STRIDE>
                           (_82_Mul_22_output_0, _83_PadConv_14_output_0, _83_PadConv_14_weights_output_0, _83_PadConv_14_biases_output_0, _83_PADCONV_14_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _83_PadConv_14\n");
    FILE* _83_PADCONV_14_OUTPUT_0_STREAM = fopen("data/export_outputs/_83_PadConv_14_output_0.txt", "w");
    export_cpp::saveOutputs<_83_PADCONV_14_OUTPUT_0_NB_OUTPUTS,
                            _83_PADCONV_14_OUTPUT_0_OUT_HEIGHT,
                            _83_PADCONV_14_OUTPUT_0_OUT_WIDTH,
                            _83_PADCONV_14_OUTPUT_0_MEM_CONT_OFFSET,
                            _83_PADCONV_14_OUTPUT_0_MEM_CONT_SIZE,
                            _83_PADCONV_14_OUTPUT_0_MEM_WRAP_OFFSET,
                            _83_PADCONV_14_OUTPUT_0_MEM_WRAP_SIZE,
                            _83_PADCONV_14_OUTPUT_0_DEV_FMT>
                            (_83_PadConv_14_output_0,
                            _83_PADCONV_14_OUTPUT_0_STREAM);
    fclose(_83_PADCONV_14_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _84_Sigmoid_23_output_0 = (float*) (mem_0 + _84_SIGMOID_23_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_84_SIGMOID_23_NB_ELTS>
                               (_83_PadConv_14_output_0, _84_Sigmoid_23_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _84_Sigmoid_23\n");
    FILE* _84_SIGMOID_23_OUTPUT_0_STREAM = fopen("data/export_outputs/_84_Sigmoid_23_output_0.txt", "w");
    export_cpp::saveOutputs<_84_SIGMOID_23_OUTPUT_0_NB_OUTPUTS,
                            _84_SIGMOID_23_OUTPUT_0_OUT_HEIGHT,
                            _84_SIGMOID_23_OUTPUT_0_OUT_WIDTH,
                            _84_SIGMOID_23_OUTPUT_0_MEM_CONT_OFFSET,
                            _84_SIGMOID_23_OUTPUT_0_MEM_CONT_SIZE,
                            _84_SIGMOID_23_OUTPUT_0_MEM_WRAP_OFFSET,
                            _84_SIGMOID_23_OUTPUT_0_MEM_WRAP_SIZE,
                            _84_SIGMOID_23_OUTPUT_0_DEV_FMT>
                            (_84_Sigmoid_23_output_0,
                            _84_SIGMOID_23_OUTPUT_0_STREAM);
    fclose(_84_SIGMOID_23_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _85_Mul_23_output_0 = (float*) (mem_0 + _85_MUL_23_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_85_MUL_23_NB_MAT,
                                 _85_MUL_23_ELEM_OP,
                                 _85_MUL_23_INPUT1_CONT_SIZE,
                                 _85_MUL_23_INPUT2_CONT_SIZE,
                                 _85_MUL_23_OUTPUT_CONT_SIZE,
                                 _85_MUL_23_OFFSET_IN1,
                                 _85_MUL_23_OFFSET_IN2,
                                 _85_MUL_23_ACTIVATION,
                                 _83_PADCONV_14_OUTPUT_0_MEM_CONT_OFFSET,
                                 _83_PADCONV_14_OUTPUT_0_MEM_CONT_SIZE,
                                 _83_PADCONV_14_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _83_PADCONV_14_OUTPUT_0_MEM_WRAP_SIZE,
                                 _83_PADCONV_14_OUTPUT_0_MEM_STRIDE,
                                 _84_SIGMOID_23_OUTPUT_0_MEM_CONT_OFFSET,
                                 _84_SIGMOID_23_OUTPUT_0_MEM_CONT_SIZE,
                                 _84_SIGMOID_23_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _84_SIGMOID_23_OUTPUT_0_MEM_WRAP_SIZE,
                                 _84_SIGMOID_23_OUTPUT_0_MEM_STRIDE,
                                 _85_MUL_23_OUTPUT_0_MEM_CONT_OFFSET,
                                 _85_MUL_23_OUTPUT_0_MEM_CONT_SIZE,
                                 _85_MUL_23_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _85_MUL_23_OUTPUT_0_MEM_WRAP_SIZE,
                                 _85_MUL_23_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_85_Mul_23_output_0, _85_MUL_23_RESCALING, _83_PadConv_14_output_0, _84_Sigmoid_23_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _85_Mul_23\n");
    FILE* _85_MUL_23_OUTPUT_0_STREAM = fopen("data/export_outputs/_85_Mul_23_output_0.txt", "w");
    export_cpp::saveOutputs<_85_MUL_23_OUTPUT_0_NB_OUTPUTS,
                            _85_MUL_23_OUTPUT_0_OUT_HEIGHT,
                            _85_MUL_23_OUTPUT_0_OUT_WIDTH,
                            _85_MUL_23_OUTPUT_0_MEM_CONT_OFFSET,
                            _85_MUL_23_OUTPUT_0_MEM_CONT_SIZE,
                            _85_MUL_23_OUTPUT_0_MEM_WRAP_OFFSET,
                            _85_MUL_23_OUTPUT_0_MEM_WRAP_SIZE,
                            _85_MUL_23_OUTPUT_0_DEV_FMT>
                            (_85_Mul_23_output_0,
                            _85_MUL_23_OUTPUT_0_STREAM);
    fclose(_85_MUL_23_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _86_Add_5_output_0 = (float*) (mem_0 + _86_ADD_5_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_86_ADD_5_NB_MAT,
                                 _86_ADD_5_ELEM_OP,
                                 _86_ADD_5_INPUT1_CONT_SIZE,
                                 _86_ADD_5_INPUT2_CONT_SIZE,
                                 _86_ADD_5_OUTPUT_CONT_SIZE,
                                 _86_ADD_5_OFFSET_IN1,
                                 _86_ADD_5_OFFSET_IN2,
                                 _86_ADD_5_ACTIVATION,
                                 _79_SPLIT_3_OUTPUT_1_MEM_CONT_OFFSET,
                                 _79_SPLIT_3_OUTPUT_1_MEM_CONT_SIZE,
                                 _79_SPLIT_3_OUTPUT_1_MEM_WRAP_OFFSET,
                                 _79_SPLIT_3_OUTPUT_1_MEM_WRAP_SIZE,
                                 _79_SPLIT_3_OUTPUT_1_MEM_STRIDE,
                                 _85_MUL_23_OUTPUT_0_MEM_CONT_OFFSET,
                                 _85_MUL_23_OUTPUT_0_MEM_CONT_SIZE,
                                 _85_MUL_23_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _85_MUL_23_OUTPUT_0_MEM_WRAP_SIZE,
                                 _85_MUL_23_OUTPUT_0_MEM_STRIDE,
                                 _86_ADD_5_OUTPUT_0_MEM_CONT_OFFSET,
                                 _86_ADD_5_OUTPUT_0_MEM_CONT_SIZE,
                                 _86_ADD_5_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _86_ADD_5_OUTPUT_0_MEM_WRAP_SIZE,
                                 _86_ADD_5_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_86_Add_5_output_0, _86_ADD_5_RESCALING, _79_Split_3_output_1, _85_Mul_23_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _86_Add_5\n");
    FILE* _86_ADD_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_86_Add_5_output_0.txt", "w");
    export_cpp::saveOutputs<_86_ADD_5_OUTPUT_0_NB_OUTPUTS,
                            _86_ADD_5_OUTPUT_0_OUT_HEIGHT,
                            _86_ADD_5_OUTPUT_0_OUT_WIDTH,
                            _86_ADD_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _86_ADD_5_OUTPUT_0_MEM_CONT_SIZE,
                            _86_ADD_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _86_ADD_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _86_ADD_5_OUTPUT_0_DEV_FMT>
                            (_86_Add_5_output_0,
                            _86_ADD_5_OUTPUT_0_STREAM);
    fclose(_86_ADD_5_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _87_Concat_3_output_0 = (float*) (mem_0 + _87_CONCAT_3_OUTPUT_0_MEM_OFFSET);

    const float* _87_Concat_3_inputs[] = { _79_Split_3_output_0, _79_Split_3_output_1, _86_Add_5_output_0 };
    export_cpp::concat_forward<_87_CONCAT_3_AXIS_SIZE_POST,
                               _87_CONCAT_3_AXIS_SIZE_PRE,
                               _87_CONCAT_3_AXIS_SIZE,
                               _87_CONCAT_3_TOTAL_AXIS_SIZE,
                               3> 
                              (_87_Concat_3_inputs,
                               _87_Concat_3_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _87_Concat_3\n");
    FILE* _87_CONCAT_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_87_Concat_3_output_0.txt", "w");
    export_cpp::saveOutputs<_87_CONCAT_3_OUTPUT_0_NB_OUTPUTS,
                            _87_CONCAT_3_OUTPUT_0_OUT_HEIGHT,
                            _87_CONCAT_3_OUTPUT_0_OUT_WIDTH,
                            _87_CONCAT_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _87_CONCAT_3_OUTPUT_0_MEM_CONT_SIZE,
                            _87_CONCAT_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _87_CONCAT_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _87_CONCAT_3_OUTPUT_0_DEV_FMT>
                            (_87_Concat_3_output_0,
                            _87_CONCAT_3_OUTPUT_0_STREAM);
    fclose(_87_CONCAT_3_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _88_Conv2D_9_output_0 = (float*) (mem_0 + _88_CONV2D_9_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_87_CONCAT_3_OUTPUT_0_IN_BATCH,
                           _88_CONV2D_9_NB_CHANNELS,
                           _87_CONCAT_3_OUTPUT_0_IN_HEIGHT,
                           _87_CONCAT_3_OUTPUT_0_IN_WIDTH,
                           _88_CONV2D_9_NB_OUTPUTS,
                           _88_CONV2D_9_OUTPUT_0_OUT_HEIGHT,
                           _88_CONV2D_9_OUTPUT_0_OUT_WIDTH,
                           _88_CONV2D_9_PADDING_Y,
                           _88_CONV2D_9_PADDING_X,
                           _88_CONV2D_9_STRIDE_Y,
                           _88_CONV2D_9_STRIDE_X,
                           _88_CONV2D_9_DILATION_Y,
                           _88_CONV2D_9_DILATION_X,
                           _88_CONV2D_9_KERNEL_HEIGHT,
                           _88_CONV2D_9_KERNEL_WIDTH,
                           _88_CONV2D_9_ACTIVATION,
                           _87_CONCAT_3_OUTPUT_0_MEM_CONT_OFFSET,
                           _87_CONCAT_3_OUTPUT_0_MEM_CONT_SIZE,
                           _87_CONCAT_3_OUTPUT_0_MEM_WRAP_OFFSET,
                           _87_CONCAT_3_OUTPUT_0_MEM_WRAP_SIZE,
                           _87_CONCAT_3_OUTPUT_0_MEM_STRIDE,
                           _88_CONV2D_9_OUTPUT_0_MEM_CONT_OFFSET,
                           _88_CONV2D_9_OUTPUT_0_MEM_CONT_SIZE,
                           _88_CONV2D_9_OUTPUT_0_MEM_WRAP_OFFSET,
                           _88_CONV2D_9_OUTPUT_0_MEM_WRAP_SIZE,
                           _88_CONV2D_9_OUTPUT_0_MEM_STRIDE>
                           (_87_Concat_3_output_0, _88_Conv2D_9_output_0, _88_Conv2D_9_weights_output_0, _88_Conv2D_9_biases_output_0, _88_CONV2D_9_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _88_Conv2D_9\n");
    FILE* _88_CONV2D_9_OUTPUT_0_STREAM = fopen("data/export_outputs/_88_Conv2D_9_output_0.txt", "w");
    export_cpp::saveOutputs<_88_CONV2D_9_OUTPUT_0_NB_OUTPUTS,
                            _88_CONV2D_9_OUTPUT_0_OUT_HEIGHT,
                            _88_CONV2D_9_OUTPUT_0_OUT_WIDTH,
                            _88_CONV2D_9_OUTPUT_0_MEM_CONT_OFFSET,
                            _88_CONV2D_9_OUTPUT_0_MEM_CONT_SIZE,
                            _88_CONV2D_9_OUTPUT_0_MEM_WRAP_OFFSET,
                            _88_CONV2D_9_OUTPUT_0_MEM_WRAP_SIZE,
                            _88_CONV2D_9_OUTPUT_0_DEV_FMT>
                            (_88_Conv2D_9_output_0,
                            _88_CONV2D_9_OUTPUT_0_STREAM);
    fclose(_88_CONV2D_9_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _89_Sigmoid_24_output_0 = (float*) (mem_0 + _89_SIGMOID_24_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_89_SIGMOID_24_NB_ELTS>
                               (_88_Conv2D_9_output_0, _89_Sigmoid_24_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _89_Sigmoid_24\n");
    FILE* _89_SIGMOID_24_OUTPUT_0_STREAM = fopen("data/export_outputs/_89_Sigmoid_24_output_0.txt", "w");
    export_cpp::saveOutputs<_89_SIGMOID_24_OUTPUT_0_NB_OUTPUTS,
                            _89_SIGMOID_24_OUTPUT_0_OUT_HEIGHT,
                            _89_SIGMOID_24_OUTPUT_0_OUT_WIDTH,
                            _89_SIGMOID_24_OUTPUT_0_MEM_CONT_OFFSET,
                            _89_SIGMOID_24_OUTPUT_0_MEM_CONT_SIZE,
                            _89_SIGMOID_24_OUTPUT_0_MEM_WRAP_OFFSET,
                            _89_SIGMOID_24_OUTPUT_0_MEM_WRAP_SIZE,
                            _89_SIGMOID_24_OUTPUT_0_DEV_FMT>
                            (_89_Sigmoid_24_output_0,
                            _89_SIGMOID_24_OUTPUT_0_STREAM);
    fclose(_89_SIGMOID_24_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _90_Mul_24_output_0 = (float*) (mem_0 + _90_MUL_24_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_90_MUL_24_NB_MAT,
                                 _90_MUL_24_ELEM_OP,
                                 _90_MUL_24_INPUT1_CONT_SIZE,
                                 _90_MUL_24_INPUT2_CONT_SIZE,
                                 _90_MUL_24_OUTPUT_CONT_SIZE,
                                 _90_MUL_24_OFFSET_IN1,
                                 _90_MUL_24_OFFSET_IN2,
                                 _90_MUL_24_ACTIVATION,
                                 _88_CONV2D_9_OUTPUT_0_MEM_CONT_OFFSET,
                                 _88_CONV2D_9_OUTPUT_0_MEM_CONT_SIZE,
                                 _88_CONV2D_9_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _88_CONV2D_9_OUTPUT_0_MEM_WRAP_SIZE,
                                 _88_CONV2D_9_OUTPUT_0_MEM_STRIDE,
                                 _89_SIGMOID_24_OUTPUT_0_MEM_CONT_OFFSET,
                                 _89_SIGMOID_24_OUTPUT_0_MEM_CONT_SIZE,
                                 _89_SIGMOID_24_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _89_SIGMOID_24_OUTPUT_0_MEM_WRAP_SIZE,
                                 _89_SIGMOID_24_OUTPUT_0_MEM_STRIDE,
                                 _90_MUL_24_OUTPUT_0_MEM_CONT_OFFSET,
                                 _90_MUL_24_OUTPUT_0_MEM_CONT_SIZE,
                                 _90_MUL_24_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _90_MUL_24_OUTPUT_0_MEM_WRAP_SIZE,
                                 _90_MUL_24_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_90_Mul_24_output_0, _90_MUL_24_RESCALING, _88_Conv2D_9_output_0, _89_Sigmoid_24_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _90_Mul_24\n");
    FILE* _90_MUL_24_OUTPUT_0_STREAM = fopen("data/export_outputs/_90_Mul_24_output_0.txt", "w");
    export_cpp::saveOutputs<_90_MUL_24_OUTPUT_0_NB_OUTPUTS,
                            _90_MUL_24_OUTPUT_0_OUT_HEIGHT,
                            _90_MUL_24_OUTPUT_0_OUT_WIDTH,
                            _90_MUL_24_OUTPUT_0_MEM_CONT_OFFSET,
                            _90_MUL_24_OUTPUT_0_MEM_CONT_SIZE,
                            _90_MUL_24_OUTPUT_0_MEM_WRAP_OFFSET,
                            _90_MUL_24_OUTPUT_0_MEM_WRAP_SIZE,
                            _90_MUL_24_OUTPUT_0_DEV_FMT>
                            (_90_Mul_24_output_0,
                            _90_MUL_24_OUTPUT_0_STREAM);
    fclose(_90_MUL_24_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _91_Conv2D_10_output_0 = (float*) (mem_0 + _91_CONV2D_10_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_90_MUL_24_OUTPUT_0_IN_BATCH,
                           _91_CONV2D_10_NB_CHANNELS,
                           _90_MUL_24_OUTPUT_0_IN_HEIGHT,
                           _90_MUL_24_OUTPUT_0_IN_WIDTH,
                           _91_CONV2D_10_NB_OUTPUTS,
                           _91_CONV2D_10_OUTPUT_0_OUT_HEIGHT,
                           _91_CONV2D_10_OUTPUT_0_OUT_WIDTH,
                           _91_CONV2D_10_PADDING_Y,
                           _91_CONV2D_10_PADDING_X,
                           _91_CONV2D_10_STRIDE_Y,
                           _91_CONV2D_10_STRIDE_X,
                           _91_CONV2D_10_DILATION_Y,
                           _91_CONV2D_10_DILATION_X,
                           _91_CONV2D_10_KERNEL_HEIGHT,
                           _91_CONV2D_10_KERNEL_WIDTH,
                           _91_CONV2D_10_ACTIVATION,
                           _90_MUL_24_OUTPUT_0_MEM_CONT_OFFSET,
                           _90_MUL_24_OUTPUT_0_MEM_CONT_SIZE,
                           _90_MUL_24_OUTPUT_0_MEM_WRAP_OFFSET,
                           _90_MUL_24_OUTPUT_0_MEM_WRAP_SIZE,
                           _90_MUL_24_OUTPUT_0_MEM_STRIDE,
                           _91_CONV2D_10_OUTPUT_0_MEM_CONT_OFFSET,
                           _91_CONV2D_10_OUTPUT_0_MEM_CONT_SIZE,
                           _91_CONV2D_10_OUTPUT_0_MEM_WRAP_OFFSET,
                           _91_CONV2D_10_OUTPUT_0_MEM_WRAP_SIZE,
                           _91_CONV2D_10_OUTPUT_0_MEM_STRIDE>
                           (_90_Mul_24_output_0, _91_Conv2D_10_output_0, _91_Conv2D_10_weights_output_0, _91_Conv2D_10_biases_output_0, _91_CONV2D_10_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _91_Conv2D_10\n");
    FILE* _91_CONV2D_10_OUTPUT_0_STREAM = fopen("data/export_outputs/_91_Conv2D_10_output_0.txt", "w");
    export_cpp::saveOutputs<_91_CONV2D_10_OUTPUT_0_NB_OUTPUTS,
                            _91_CONV2D_10_OUTPUT_0_OUT_HEIGHT,
                            _91_CONV2D_10_OUTPUT_0_OUT_WIDTH,
                            _91_CONV2D_10_OUTPUT_0_MEM_CONT_OFFSET,
                            _91_CONV2D_10_OUTPUT_0_MEM_CONT_SIZE,
                            _91_CONV2D_10_OUTPUT_0_MEM_WRAP_OFFSET,
                            _91_CONV2D_10_OUTPUT_0_MEM_WRAP_SIZE,
                            _91_CONV2D_10_OUTPUT_0_DEV_FMT>
                            (_91_Conv2D_10_output_0,
                            _91_CONV2D_10_OUTPUT_0_STREAM);
    fclose(_91_CONV2D_10_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _92_Sigmoid_25_output_0 = (float*) (mem_0 + _92_SIGMOID_25_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_92_SIGMOID_25_NB_ELTS>
                               (_91_Conv2D_10_output_0, _92_Sigmoid_25_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _92_Sigmoid_25\n");
    FILE* _92_SIGMOID_25_OUTPUT_0_STREAM = fopen("data/export_outputs/_92_Sigmoid_25_output_0.txt", "w");
    export_cpp::saveOutputs<_92_SIGMOID_25_OUTPUT_0_NB_OUTPUTS,
                            _92_SIGMOID_25_OUTPUT_0_OUT_HEIGHT,
                            _92_SIGMOID_25_OUTPUT_0_OUT_WIDTH,
                            _92_SIGMOID_25_OUTPUT_0_MEM_CONT_OFFSET,
                            _92_SIGMOID_25_OUTPUT_0_MEM_CONT_SIZE,
                            _92_SIGMOID_25_OUTPUT_0_MEM_WRAP_OFFSET,
                            _92_SIGMOID_25_OUTPUT_0_MEM_WRAP_SIZE,
                            _92_SIGMOID_25_OUTPUT_0_DEV_FMT>
                            (_92_Sigmoid_25_output_0,
                            _92_SIGMOID_25_OUTPUT_0_STREAM);
    fclose(_92_SIGMOID_25_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _93_Mul_25_output_0 = (float*) (mem_0 + _93_MUL_25_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_93_MUL_25_NB_MAT,
                                 _93_MUL_25_ELEM_OP,
                                 _93_MUL_25_INPUT1_CONT_SIZE,
                                 _93_MUL_25_INPUT2_CONT_SIZE,
                                 _93_MUL_25_OUTPUT_CONT_SIZE,
                                 _93_MUL_25_OFFSET_IN1,
                                 _93_MUL_25_OFFSET_IN2,
                                 _93_MUL_25_ACTIVATION,
                                 _91_CONV2D_10_OUTPUT_0_MEM_CONT_OFFSET,
                                 _91_CONV2D_10_OUTPUT_0_MEM_CONT_SIZE,
                                 _91_CONV2D_10_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _91_CONV2D_10_OUTPUT_0_MEM_WRAP_SIZE,
                                 _91_CONV2D_10_OUTPUT_0_MEM_STRIDE,
                                 _92_SIGMOID_25_OUTPUT_0_MEM_CONT_OFFSET,
                                 _92_SIGMOID_25_OUTPUT_0_MEM_CONT_SIZE,
                                 _92_SIGMOID_25_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _92_SIGMOID_25_OUTPUT_0_MEM_WRAP_SIZE,
                                 _92_SIGMOID_25_OUTPUT_0_MEM_STRIDE,
                                 _93_MUL_25_OUTPUT_0_MEM_CONT_OFFSET,
                                 _93_MUL_25_OUTPUT_0_MEM_CONT_SIZE,
                                 _93_MUL_25_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _93_MUL_25_OUTPUT_0_MEM_WRAP_SIZE,
                                 _93_MUL_25_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_93_Mul_25_output_0, _93_MUL_25_RESCALING, _91_Conv2D_10_output_0, _92_Sigmoid_25_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _93_Mul_25\n");
    FILE* _93_MUL_25_OUTPUT_0_STREAM = fopen("data/export_outputs/_93_Mul_25_output_0.txt", "w");
    export_cpp::saveOutputs<_93_MUL_25_OUTPUT_0_NB_OUTPUTS,
                            _93_MUL_25_OUTPUT_0_OUT_HEIGHT,
                            _93_MUL_25_OUTPUT_0_OUT_WIDTH,
                            _93_MUL_25_OUTPUT_0_MEM_CONT_OFFSET,
                            _93_MUL_25_OUTPUT_0_MEM_CONT_SIZE,
                            _93_MUL_25_OUTPUT_0_MEM_WRAP_OFFSET,
                            _93_MUL_25_OUTPUT_0_MEM_WRAP_SIZE,
                            _93_MUL_25_OUTPUT_0_DEV_FMT>
                            (_93_Mul_25_output_0,
                            _93_MUL_25_OUTPUT_0_STREAM);
    fclose(_93_MUL_25_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _94_PadMaxPool_0_output_0 = (float*) (mem_0 + _94_PADMAXPOOL_0_OUTPUT_0_MEM_OFFSET);
    int64_t* _94_PadMaxPool_0_output_1 = (int64_t*) (mem_0 + _94_PADMAXPOOL_0_OUTPUT_1_MEM_OFFSET);

    export_cpp::pooling_forward<_93_MUL_25_OUTPUT_0_IN_BATCH,
                                _93_MUL_25_OUTPUT_0_NB_CHANNELS,
                                _93_MUL_25_OUTPUT_0_IN_HEIGHT,
                                _93_MUL_25_OUTPUT_0_IN_WIDTH,
                                _94_PADMAXPOOL_0_OUTPUT_0_NB_OUTPUTS,
                                _94_PADMAXPOOL_0_OUTPUT_0_OUT_HEIGHT,
                                _94_PADMAXPOOL_0_OUTPUT_0_OUT_WIDTH,
                                _94_PADMAXPOOL_0_PADDING_Y,
                                _94_PADMAXPOOL_0_PADDING_X,
                                _94_PADMAXPOOL_0_STRIDE_Y,
                                _94_PADMAXPOOL_0_STRIDE_X,
                                _94_PADMAXPOOL_0_KERNEL_HEIGHT,
                                _94_PADMAXPOOL_0_KERNEL_WIDTH,
                                _94_PADMAXPOOL_0_POOLING_TYPE,
                                _94_PADMAXPOOL_0_ACTIVATION,
                                _93_MUL_25_OUTPUT_0_MEM_CONT_OFFSET,
                                _93_MUL_25_OUTPUT_0_MEM_CONT_SIZE,
                                _93_MUL_25_OUTPUT_0_MEM_WRAP_OFFSET,
                                _93_MUL_25_OUTPUT_0_MEM_WRAP_SIZE,
                                _93_MUL_25_OUTPUT_0_MEM_STRIDE,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_CONT_OFFSET,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_CONT_SIZE,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_WRAP_SIZE,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_STRIDE>
                                (_93_Mul_25_output_0, _94_PadMaxPool_0_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _94_PadMaxPool_0\n");
    FILE* _94_PADMAXPOOL_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_94_PadMaxPool_0_output_0.txt", "w");
    export_cpp::saveOutputs<_94_PADMAXPOOL_0_OUTPUT_0_NB_OUTPUTS,
                            _94_PADMAXPOOL_0_OUTPUT_0_OUT_HEIGHT,
                            _94_PADMAXPOOL_0_OUTPUT_0_OUT_WIDTH,
                            _94_PADMAXPOOL_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _94_PADMAXPOOL_0_OUTPUT_0_MEM_CONT_SIZE,
                            _94_PADMAXPOOL_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _94_PADMAXPOOL_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _94_PADMAXPOOL_0_OUTPUT_0_DEV_FMT>
                            (_94_PadMaxPool_0_output_0,
                            _94_PADMAXPOOL_0_OUTPUT_0_STREAM);
    fclose(_94_PADMAXPOOL_0_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _95_PadMaxPool_1_output_0 = (float*) (mem_0 + _95_PADMAXPOOL_1_OUTPUT_0_MEM_OFFSET);
    int64_t* _95_PadMaxPool_1_output_1 = (int64_t*) (mem_0 + _95_PADMAXPOOL_1_OUTPUT_1_MEM_OFFSET);

    export_cpp::pooling_forward<_94_PADMAXPOOL_0_OUTPUT_0_IN_BATCH,
                                _94_PADMAXPOOL_0_OUTPUT_0_NB_CHANNELS,
                                _94_PADMAXPOOL_0_OUTPUT_0_IN_HEIGHT,
                                _94_PADMAXPOOL_0_OUTPUT_0_IN_WIDTH,
                                _95_PADMAXPOOL_1_OUTPUT_0_NB_OUTPUTS,
                                _95_PADMAXPOOL_1_OUTPUT_0_OUT_HEIGHT,
                                _95_PADMAXPOOL_1_OUTPUT_0_OUT_WIDTH,
                                _95_PADMAXPOOL_1_PADDING_Y,
                                _95_PADMAXPOOL_1_PADDING_X,
                                _95_PADMAXPOOL_1_STRIDE_Y,
                                _95_PADMAXPOOL_1_STRIDE_X,
                                _95_PADMAXPOOL_1_KERNEL_HEIGHT,
                                _95_PADMAXPOOL_1_KERNEL_WIDTH,
                                _95_PADMAXPOOL_1_POOLING_TYPE,
                                _95_PADMAXPOOL_1_ACTIVATION,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_CONT_OFFSET,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_CONT_SIZE,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_WRAP_SIZE,
                                _94_PADMAXPOOL_0_OUTPUT_0_MEM_STRIDE,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_CONT_OFFSET,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_CONT_SIZE,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_WRAP_SIZE,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_STRIDE>
                                (_94_PadMaxPool_0_output_0, _95_PadMaxPool_1_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _95_PadMaxPool_1\n");
    FILE* _95_PADMAXPOOL_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_95_PadMaxPool_1_output_0.txt", "w");
    export_cpp::saveOutputs<_95_PADMAXPOOL_1_OUTPUT_0_NB_OUTPUTS,
                            _95_PADMAXPOOL_1_OUTPUT_0_OUT_HEIGHT,
                            _95_PADMAXPOOL_1_OUTPUT_0_OUT_WIDTH,
                            _95_PADMAXPOOL_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _95_PADMAXPOOL_1_OUTPUT_0_MEM_CONT_SIZE,
                            _95_PADMAXPOOL_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _95_PADMAXPOOL_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _95_PADMAXPOOL_1_OUTPUT_0_DEV_FMT>
                            (_95_PadMaxPool_1_output_0,
                            _95_PADMAXPOOL_1_OUTPUT_0_STREAM);
    fclose(_95_PADMAXPOOL_1_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _96_PadMaxPool_2_output_0 = (float*) (mem_0 + _96_PADMAXPOOL_2_OUTPUT_0_MEM_OFFSET);
    int64_t* _96_PadMaxPool_2_output_1 = (int64_t*) (mem_0 + _96_PADMAXPOOL_2_OUTPUT_1_MEM_OFFSET);

    export_cpp::pooling_forward<_95_PADMAXPOOL_1_OUTPUT_0_IN_BATCH,
                                _95_PADMAXPOOL_1_OUTPUT_0_NB_CHANNELS,
                                _95_PADMAXPOOL_1_OUTPUT_0_IN_HEIGHT,
                                _95_PADMAXPOOL_1_OUTPUT_0_IN_WIDTH,
                                _96_PADMAXPOOL_2_OUTPUT_0_NB_OUTPUTS,
                                _96_PADMAXPOOL_2_OUTPUT_0_OUT_HEIGHT,
                                _96_PADMAXPOOL_2_OUTPUT_0_OUT_WIDTH,
                                _96_PADMAXPOOL_2_PADDING_Y,
                                _96_PADMAXPOOL_2_PADDING_X,
                                _96_PADMAXPOOL_2_STRIDE_Y,
                                _96_PADMAXPOOL_2_STRIDE_X,
                                _96_PADMAXPOOL_2_KERNEL_HEIGHT,
                                _96_PADMAXPOOL_2_KERNEL_WIDTH,
                                _96_PADMAXPOOL_2_POOLING_TYPE,
                                _96_PADMAXPOOL_2_ACTIVATION,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_CONT_OFFSET,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_CONT_SIZE,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_WRAP_SIZE,
                                _95_PADMAXPOOL_1_OUTPUT_0_MEM_STRIDE,
                                _96_PADMAXPOOL_2_OUTPUT_0_MEM_CONT_OFFSET,
                                _96_PADMAXPOOL_2_OUTPUT_0_MEM_CONT_SIZE,
                                _96_PADMAXPOOL_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                _96_PADMAXPOOL_2_OUTPUT_0_MEM_WRAP_SIZE,
                                _96_PADMAXPOOL_2_OUTPUT_0_MEM_STRIDE>
                                (_95_PadMaxPool_1_output_0, _96_PadMaxPool_2_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _96_PadMaxPool_2\n");
    FILE* _96_PADMAXPOOL_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_96_PadMaxPool_2_output_0.txt", "w");
    export_cpp::saveOutputs<_96_PADMAXPOOL_2_OUTPUT_0_NB_OUTPUTS,
                            _96_PADMAXPOOL_2_OUTPUT_0_OUT_HEIGHT,
                            _96_PADMAXPOOL_2_OUTPUT_0_OUT_WIDTH,
                            _96_PADMAXPOOL_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _96_PADMAXPOOL_2_OUTPUT_0_MEM_CONT_SIZE,
                            _96_PADMAXPOOL_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _96_PADMAXPOOL_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _96_PADMAXPOOL_2_OUTPUT_0_DEV_FMT>
                            (_96_PadMaxPool_2_output_0,
                            _96_PADMAXPOOL_2_OUTPUT_0_STREAM);
    fclose(_96_PADMAXPOOL_2_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _97_Concat_4_output_0 = (float*) (mem_0 + _97_CONCAT_4_OUTPUT_0_MEM_OFFSET);

    const float* _97_Concat_4_inputs[] = { _93_Mul_25_output_0, _94_PadMaxPool_0_output_0, _95_PadMaxPool_1_output_0, _96_PadMaxPool_2_output_0 };
    export_cpp::concat_forward<_97_CONCAT_4_AXIS_SIZE_POST,
                               _97_CONCAT_4_AXIS_SIZE_PRE,
                               _97_CONCAT_4_AXIS_SIZE,
                               _97_CONCAT_4_TOTAL_AXIS_SIZE,
                               4> 
                              (_97_Concat_4_inputs,
                               _97_Concat_4_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _97_Concat_4\n");
    FILE* _97_CONCAT_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_97_Concat_4_output_0.txt", "w");
    export_cpp::saveOutputs<_97_CONCAT_4_OUTPUT_0_NB_OUTPUTS,
                            _97_CONCAT_4_OUTPUT_0_OUT_HEIGHT,
                            _97_CONCAT_4_OUTPUT_0_OUT_WIDTH,
                            _97_CONCAT_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _97_CONCAT_4_OUTPUT_0_MEM_CONT_SIZE,
                            _97_CONCAT_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _97_CONCAT_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _97_CONCAT_4_OUTPUT_0_DEV_FMT>
                            (_97_Concat_4_output_0,
                            _97_CONCAT_4_OUTPUT_0_STREAM);
    fclose(_97_CONCAT_4_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _98_Conv2D_11_output_0 = (float*) (mem_0 + _98_CONV2D_11_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_97_CONCAT_4_OUTPUT_0_IN_BATCH,
                           _98_CONV2D_11_NB_CHANNELS,
                           _97_CONCAT_4_OUTPUT_0_IN_HEIGHT,
                           _97_CONCAT_4_OUTPUT_0_IN_WIDTH,
                           _98_CONV2D_11_NB_OUTPUTS,
                           _98_CONV2D_11_OUTPUT_0_OUT_HEIGHT,
                           _98_CONV2D_11_OUTPUT_0_OUT_WIDTH,
                           _98_CONV2D_11_PADDING_Y,
                           _98_CONV2D_11_PADDING_X,
                           _98_CONV2D_11_STRIDE_Y,
                           _98_CONV2D_11_STRIDE_X,
                           _98_CONV2D_11_DILATION_Y,
                           _98_CONV2D_11_DILATION_X,
                           _98_CONV2D_11_KERNEL_HEIGHT,
                           _98_CONV2D_11_KERNEL_WIDTH,
                           _98_CONV2D_11_ACTIVATION,
                           _97_CONCAT_4_OUTPUT_0_MEM_CONT_OFFSET,
                           _97_CONCAT_4_OUTPUT_0_MEM_CONT_SIZE,
                           _97_CONCAT_4_OUTPUT_0_MEM_WRAP_OFFSET,
                           _97_CONCAT_4_OUTPUT_0_MEM_WRAP_SIZE,
                           _97_CONCAT_4_OUTPUT_0_MEM_STRIDE,
                           _98_CONV2D_11_OUTPUT_0_MEM_CONT_OFFSET,
                           _98_CONV2D_11_OUTPUT_0_MEM_CONT_SIZE,
                           _98_CONV2D_11_OUTPUT_0_MEM_WRAP_OFFSET,
                           _98_CONV2D_11_OUTPUT_0_MEM_WRAP_SIZE,
                           _98_CONV2D_11_OUTPUT_0_MEM_STRIDE>
                           (_97_Concat_4_output_0, _98_Conv2D_11_output_0, _98_Conv2D_11_weights_output_0, _98_Conv2D_11_biases_output_0, _98_CONV2D_11_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _98_Conv2D_11\n");
    FILE* _98_CONV2D_11_OUTPUT_0_STREAM = fopen("data/export_outputs/_98_Conv2D_11_output_0.txt", "w");
    export_cpp::saveOutputs<_98_CONV2D_11_OUTPUT_0_NB_OUTPUTS,
                            _98_CONV2D_11_OUTPUT_0_OUT_HEIGHT,
                            _98_CONV2D_11_OUTPUT_0_OUT_WIDTH,
                            _98_CONV2D_11_OUTPUT_0_MEM_CONT_OFFSET,
                            _98_CONV2D_11_OUTPUT_0_MEM_CONT_SIZE,
                            _98_CONV2D_11_OUTPUT_0_MEM_WRAP_OFFSET,
                            _98_CONV2D_11_OUTPUT_0_MEM_WRAP_SIZE,
                            _98_CONV2D_11_OUTPUT_0_DEV_FMT>
                            (_98_Conv2D_11_output_0,
                            _98_CONV2D_11_OUTPUT_0_STREAM);
    fclose(_98_CONV2D_11_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _99_Sigmoid_26_output_0 = (float*) (mem_0 + _99_SIGMOID_26_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_99_SIGMOID_26_NB_ELTS>
                               (_98_Conv2D_11_output_0, _99_Sigmoid_26_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _99_Sigmoid_26\n");
    FILE* _99_SIGMOID_26_OUTPUT_0_STREAM = fopen("data/export_outputs/_99_Sigmoid_26_output_0.txt", "w");
    export_cpp::saveOutputs<_99_SIGMOID_26_OUTPUT_0_NB_OUTPUTS,
                            _99_SIGMOID_26_OUTPUT_0_OUT_HEIGHT,
                            _99_SIGMOID_26_OUTPUT_0_OUT_WIDTH,
                            _99_SIGMOID_26_OUTPUT_0_MEM_CONT_OFFSET,
                            _99_SIGMOID_26_OUTPUT_0_MEM_CONT_SIZE,
                            _99_SIGMOID_26_OUTPUT_0_MEM_WRAP_OFFSET,
                            _99_SIGMOID_26_OUTPUT_0_MEM_WRAP_SIZE,
                            _99_SIGMOID_26_OUTPUT_0_DEV_FMT>
                            (_99_Sigmoid_26_output_0,
                            _99_SIGMOID_26_OUTPUT_0_STREAM);
    fclose(_99_SIGMOID_26_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _100_Mul_26_output_0 = (float*) (mem_0 + _100_MUL_26_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_100_MUL_26_NB_MAT,
                                 _100_MUL_26_ELEM_OP,
                                 _100_MUL_26_INPUT1_CONT_SIZE,
                                 _100_MUL_26_INPUT2_CONT_SIZE,
                                 _100_MUL_26_OUTPUT_CONT_SIZE,
                                 _100_MUL_26_OFFSET_IN1,
                                 _100_MUL_26_OFFSET_IN2,
                                 _100_MUL_26_ACTIVATION,
                                 _98_CONV2D_11_OUTPUT_0_MEM_CONT_OFFSET,
                                 _98_CONV2D_11_OUTPUT_0_MEM_CONT_SIZE,
                                 _98_CONV2D_11_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _98_CONV2D_11_OUTPUT_0_MEM_WRAP_SIZE,
                                 _98_CONV2D_11_OUTPUT_0_MEM_STRIDE,
                                 _99_SIGMOID_26_OUTPUT_0_MEM_CONT_OFFSET,
                                 _99_SIGMOID_26_OUTPUT_0_MEM_CONT_SIZE,
                                 _99_SIGMOID_26_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _99_SIGMOID_26_OUTPUT_0_MEM_WRAP_SIZE,
                                 _99_SIGMOID_26_OUTPUT_0_MEM_STRIDE,
                                 _100_MUL_26_OUTPUT_0_MEM_CONT_OFFSET,
                                 _100_MUL_26_OUTPUT_0_MEM_CONT_SIZE,
                                 _100_MUL_26_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _100_MUL_26_OUTPUT_0_MEM_WRAP_SIZE,
                                 _100_MUL_26_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_100_Mul_26_output_0, _100_MUL_26_RESCALING, _98_Conv2D_11_output_0, _99_Sigmoid_26_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _100_Mul_26\n");
    FILE* _100_MUL_26_OUTPUT_0_STREAM = fopen("data/export_outputs/_100_Mul_26_output_0.txt", "w");
    export_cpp::saveOutputs<_100_MUL_26_OUTPUT_0_NB_OUTPUTS,
                            _100_MUL_26_OUTPUT_0_OUT_HEIGHT,
                            _100_MUL_26_OUTPUT_0_OUT_WIDTH,
                            _100_MUL_26_OUTPUT_0_MEM_CONT_OFFSET,
                            _100_MUL_26_OUTPUT_0_MEM_CONT_SIZE,
                            _100_MUL_26_OUTPUT_0_MEM_WRAP_OFFSET,
                            _100_MUL_26_OUTPUT_0_MEM_WRAP_SIZE,
                            _100_MUL_26_OUTPUT_0_DEV_FMT>
                            (_100_Mul_26_output_0,
                            _100_MUL_26_OUTPUT_0_STREAM);
    fclose(_100_MUL_26_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _101_Conv2D_12_output_0 = (float*) (mem_0 + _101_CONV2D_12_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_100_MUL_26_OUTPUT_0_IN_BATCH,
                           _101_CONV2D_12_NB_CHANNELS,
                           _100_MUL_26_OUTPUT_0_IN_HEIGHT,
                           _100_MUL_26_OUTPUT_0_IN_WIDTH,
                           _101_CONV2D_12_NB_OUTPUTS,
                           _101_CONV2D_12_OUTPUT_0_OUT_HEIGHT,
                           _101_CONV2D_12_OUTPUT_0_OUT_WIDTH,
                           _101_CONV2D_12_PADDING_Y,
                           _101_CONV2D_12_PADDING_X,
                           _101_CONV2D_12_STRIDE_Y,
                           _101_CONV2D_12_STRIDE_X,
                           _101_CONV2D_12_DILATION_Y,
                           _101_CONV2D_12_DILATION_X,
                           _101_CONV2D_12_KERNEL_HEIGHT,
                           _101_CONV2D_12_KERNEL_WIDTH,
                           _101_CONV2D_12_ACTIVATION,
                           _100_MUL_26_OUTPUT_0_MEM_CONT_OFFSET,
                           _100_MUL_26_OUTPUT_0_MEM_CONT_SIZE,
                           _100_MUL_26_OUTPUT_0_MEM_WRAP_OFFSET,
                           _100_MUL_26_OUTPUT_0_MEM_WRAP_SIZE,
                           _100_MUL_26_OUTPUT_0_MEM_STRIDE,
                           _101_CONV2D_12_OUTPUT_0_MEM_CONT_OFFSET,
                           _101_CONV2D_12_OUTPUT_0_MEM_CONT_SIZE,
                           _101_CONV2D_12_OUTPUT_0_MEM_WRAP_OFFSET,
                           _101_CONV2D_12_OUTPUT_0_MEM_WRAP_SIZE,
                           _101_CONV2D_12_OUTPUT_0_MEM_STRIDE>
                           (_100_Mul_26_output_0, _101_Conv2D_12_output_0, _101_Conv2D_12_weights_output_0, _101_Conv2D_12_biases_output_0, _101_CONV2D_12_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _101_Conv2D_12\n");
    FILE* _101_CONV2D_12_OUTPUT_0_STREAM = fopen("data/export_outputs/_101_Conv2D_12_output_0.txt", "w");
    export_cpp::saveOutputs<_101_CONV2D_12_OUTPUT_0_NB_OUTPUTS,
                            _101_CONV2D_12_OUTPUT_0_OUT_HEIGHT,
                            _101_CONV2D_12_OUTPUT_0_OUT_WIDTH,
                            _101_CONV2D_12_OUTPUT_0_MEM_CONT_OFFSET,
                            _101_CONV2D_12_OUTPUT_0_MEM_CONT_SIZE,
                            _101_CONV2D_12_OUTPUT_0_MEM_WRAP_OFFSET,
                            _101_CONV2D_12_OUTPUT_0_MEM_WRAP_SIZE,
                            _101_CONV2D_12_OUTPUT_0_DEV_FMT>
                            (_101_Conv2D_12_output_0,
                            _101_CONV2D_12_OUTPUT_0_STREAM);
    fclose(_101_CONV2D_12_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _102_Sigmoid_27_output_0 = (float*) (mem_0 + _102_SIGMOID_27_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_102_SIGMOID_27_NB_ELTS>
                               (_101_Conv2D_12_output_0, _102_Sigmoid_27_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _102_Sigmoid_27\n");
    FILE* _102_SIGMOID_27_OUTPUT_0_STREAM = fopen("data/export_outputs/_102_Sigmoid_27_output_0.txt", "w");
    export_cpp::saveOutputs<_102_SIGMOID_27_OUTPUT_0_NB_OUTPUTS,
                            _102_SIGMOID_27_OUTPUT_0_OUT_HEIGHT,
                            _102_SIGMOID_27_OUTPUT_0_OUT_WIDTH,
                            _102_SIGMOID_27_OUTPUT_0_MEM_CONT_OFFSET,
                            _102_SIGMOID_27_OUTPUT_0_MEM_CONT_SIZE,
                            _102_SIGMOID_27_OUTPUT_0_MEM_WRAP_OFFSET,
                            _102_SIGMOID_27_OUTPUT_0_MEM_WRAP_SIZE,
                            _102_SIGMOID_27_OUTPUT_0_DEV_FMT>
                            (_102_Sigmoid_27_output_0,
                            _102_SIGMOID_27_OUTPUT_0_STREAM);
    fclose(_102_SIGMOID_27_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _103_Mul_27_output_0 = (float*) (mem_0 + _103_MUL_27_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_103_MUL_27_NB_MAT,
                                 _103_MUL_27_ELEM_OP,
                                 _103_MUL_27_INPUT1_CONT_SIZE,
                                 _103_MUL_27_INPUT2_CONT_SIZE,
                                 _103_MUL_27_OUTPUT_CONT_SIZE,
                                 _103_MUL_27_OFFSET_IN1,
                                 _103_MUL_27_OFFSET_IN2,
                                 _103_MUL_27_ACTIVATION,
                                 _101_CONV2D_12_OUTPUT_0_MEM_CONT_OFFSET,
                                 _101_CONV2D_12_OUTPUT_0_MEM_CONT_SIZE,
                                 _101_CONV2D_12_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _101_CONV2D_12_OUTPUT_0_MEM_WRAP_SIZE,
                                 _101_CONV2D_12_OUTPUT_0_MEM_STRIDE,
                                 _102_SIGMOID_27_OUTPUT_0_MEM_CONT_OFFSET,
                                 _102_SIGMOID_27_OUTPUT_0_MEM_CONT_SIZE,
                                 _102_SIGMOID_27_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _102_SIGMOID_27_OUTPUT_0_MEM_WRAP_SIZE,
                                 _102_SIGMOID_27_OUTPUT_0_MEM_STRIDE,
                                 _103_MUL_27_OUTPUT_0_MEM_CONT_OFFSET,
                                 _103_MUL_27_OUTPUT_0_MEM_CONT_SIZE,
                                 _103_MUL_27_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _103_MUL_27_OUTPUT_0_MEM_WRAP_SIZE,
                                 _103_MUL_27_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_103_Mul_27_output_0, _103_MUL_27_RESCALING, _101_Conv2D_12_output_0, _102_Sigmoid_27_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _103_Mul_27\n");
    FILE* _103_MUL_27_OUTPUT_0_STREAM = fopen("data/export_outputs/_103_Mul_27_output_0.txt", "w");
    export_cpp::saveOutputs<_103_MUL_27_OUTPUT_0_NB_OUTPUTS,
                            _103_MUL_27_OUTPUT_0_OUT_HEIGHT,
                            _103_MUL_27_OUTPUT_0_OUT_WIDTH,
                            _103_MUL_27_OUTPUT_0_MEM_CONT_OFFSET,
                            _103_MUL_27_OUTPUT_0_MEM_CONT_SIZE,
                            _103_MUL_27_OUTPUT_0_MEM_WRAP_OFFSET,
                            _103_MUL_27_OUTPUT_0_MEM_WRAP_SIZE,
                            _103_MUL_27_OUTPUT_0_DEV_FMT>
                            (_103_Mul_27_output_0,
                            _103_MUL_27_OUTPUT_0_STREAM);
    fclose(_103_MUL_27_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _104_Split_4_output_0 = (float*) (mem_0 + _104_SPLIT_4_OUTPUT_0_MEM_OFFSET);
    float* _104_Split_4_output_1 = (float*) (mem_0 + _104_SPLIT_4_OUTPUT_1_MEM_OFFSET);



    float* __104_Split_4_outputs[_104_SPLIT_4_NB_OUTPUTS] = {
        
        _104_Split_4_output_0,
        
        _104_Split_4_output_1
        
    };

    export_cpp::split_forward<_104_SPLIT_4_NB_OUTPUTS,
                              _104_SPLIT_4_AXIS_DIM,
                              _104_SPLIT_4_PRE_AXIS_ELEMS,
                              _104_SPLIT_4_POST_AXIS_ELEMS,
                              _104_SPLIT_4_SPLITS,
                              float>
                              (_103_Mul_27_output_0,
                               __104_Split_4_outputs);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _104_Split_4\n");
    FILE* _104_SPLIT_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_104_Split_4_output_0.txt", "w");
    export_cpp::saveOutputs<_104_SPLIT_4_OUTPUT_0_NB_OUTPUTS,
                            _104_SPLIT_4_OUTPUT_0_OUT_HEIGHT,
                            _104_SPLIT_4_OUTPUT_0_OUT_WIDTH,
                            _104_SPLIT_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _104_SPLIT_4_OUTPUT_0_MEM_CONT_SIZE,
                            _104_SPLIT_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _104_SPLIT_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _104_SPLIT_4_OUTPUT_0_DEV_FMT>
                            (_104_Split_4_output_0,
                            _104_SPLIT_4_OUTPUT_0_STREAM);
    fclose(_104_SPLIT_4_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _105_Add_6_output_0 = (float*) (mem_0 + _105_ADD_6_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_105_ADD_6_NB_MAT,
                                 _105_ADD_6_ELEM_OP,
                                 _105_ADD_6_INPUT1_CONT_SIZE,
                                 _105_ADD_6_INPUT2_CONT_SIZE,
                                 _105_ADD_6_OUTPUT_CONT_SIZE,
                                 _105_ADD_6_OFFSET_IN1,
                                 _105_ADD_6_OFFSET_IN2,
                                 _105_ADD_6_ACTIVATION,
                                 _104_SPLIT_4_OUTPUT_1_MEM_CONT_OFFSET,
                                 _104_SPLIT_4_OUTPUT_1_MEM_CONT_SIZE,
                                 _104_SPLIT_4_OUTPUT_1_MEM_WRAP_OFFSET,
                                 _104_SPLIT_4_OUTPUT_1_MEM_WRAP_SIZE,
                                 _104_SPLIT_4_OUTPUT_1_MEM_STRIDE,
                                 _104_SPLIT_4_OUTPUT_1_MEM_CONT_OFFSET,
                                 _104_SPLIT_4_OUTPUT_1_MEM_CONT_SIZE,
                                 _104_SPLIT_4_OUTPUT_1_MEM_WRAP_OFFSET,
                                 _104_SPLIT_4_OUTPUT_1_MEM_WRAP_SIZE,
                                 _104_SPLIT_4_OUTPUT_1_MEM_STRIDE,
                                 _105_ADD_6_OUTPUT_0_MEM_CONT_OFFSET,
                                 _105_ADD_6_OUTPUT_0_MEM_CONT_SIZE,
                                 _105_ADD_6_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _105_ADD_6_OUTPUT_0_MEM_WRAP_SIZE,
                                 _105_ADD_6_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_105_Add_6_output_0, _105_ADD_6_RESCALING, _104_Split_4_output_1, _104_Split_4_output_1);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _105_Add_6\n");
    FILE* _105_ADD_6_OUTPUT_0_STREAM = fopen("data/export_outputs/_105_Add_6_output_0.txt", "w");
    export_cpp::saveOutputs<_105_ADD_6_OUTPUT_0_NB_OUTPUTS,
                            _105_ADD_6_OUTPUT_0_OUT_HEIGHT,
                            _105_ADD_6_OUTPUT_0_OUT_WIDTH,
                            _105_ADD_6_OUTPUT_0_MEM_CONT_OFFSET,
                            _105_ADD_6_OUTPUT_0_MEM_CONT_SIZE,
                            _105_ADD_6_OUTPUT_0_MEM_WRAP_OFFSET,
                            _105_ADD_6_OUTPUT_0_MEM_WRAP_SIZE,
                            _105_ADD_6_OUTPUT_0_DEV_FMT>
                            (_105_Add_6_output_0,
                            _105_ADD_6_OUTPUT_0_STREAM);
    fclose(_105_ADD_6_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _106_Conv2D_13_output_0 = (float*) (mem_0 + _106_CONV2D_13_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_105_ADD_6_OUTPUT_0_IN_BATCH,
                           _106_CONV2D_13_NB_CHANNELS,
                           _105_ADD_6_OUTPUT_0_IN_HEIGHT,
                           _105_ADD_6_OUTPUT_0_IN_WIDTH,
                           _106_CONV2D_13_NB_OUTPUTS,
                           _106_CONV2D_13_OUTPUT_0_OUT_HEIGHT,
                           _106_CONV2D_13_OUTPUT_0_OUT_WIDTH,
                           _106_CONV2D_13_PADDING_Y,
                           _106_CONV2D_13_PADDING_X,
                           _106_CONV2D_13_STRIDE_Y,
                           _106_CONV2D_13_STRIDE_X,
                           _106_CONV2D_13_DILATION_Y,
                           _106_CONV2D_13_DILATION_X,
                           _106_CONV2D_13_KERNEL_HEIGHT,
                           _106_CONV2D_13_KERNEL_WIDTH,
                           _106_CONV2D_13_ACTIVATION,
                           _105_ADD_6_OUTPUT_0_MEM_CONT_OFFSET,
                           _105_ADD_6_OUTPUT_0_MEM_CONT_SIZE,
                           _105_ADD_6_OUTPUT_0_MEM_WRAP_OFFSET,
                           _105_ADD_6_OUTPUT_0_MEM_WRAP_SIZE,
                           _105_ADD_6_OUTPUT_0_MEM_STRIDE,
                           _106_CONV2D_13_OUTPUT_0_MEM_CONT_OFFSET,
                           _106_CONV2D_13_OUTPUT_0_MEM_CONT_SIZE,
                           _106_CONV2D_13_OUTPUT_0_MEM_WRAP_OFFSET,
                           _106_CONV2D_13_OUTPUT_0_MEM_WRAP_SIZE,
                           _106_CONV2D_13_OUTPUT_0_MEM_STRIDE>
                           (_105_Add_6_output_0, _106_Conv2D_13_output_0, _106_Conv2D_13_weights_output_0, _106_Conv2D_13_biases_output_0, _106_CONV2D_13_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _106_Conv2D_13\n");
    FILE* _106_CONV2D_13_OUTPUT_0_STREAM = fopen("data/export_outputs/_106_Conv2D_13_output_0.txt", "w");
    export_cpp::saveOutputs<_106_CONV2D_13_OUTPUT_0_NB_OUTPUTS,
                            _106_CONV2D_13_OUTPUT_0_OUT_HEIGHT,
                            _106_CONV2D_13_OUTPUT_0_OUT_WIDTH,
                            _106_CONV2D_13_OUTPUT_0_MEM_CONT_OFFSET,
                            _106_CONV2D_13_OUTPUT_0_MEM_CONT_SIZE,
                            _106_CONV2D_13_OUTPUT_0_MEM_WRAP_OFFSET,
                            _106_CONV2D_13_OUTPUT_0_MEM_WRAP_SIZE,
                            _106_CONV2D_13_OUTPUT_0_DEV_FMT>
                            (_106_Conv2D_13_output_0,
                            _106_CONV2D_13_OUTPUT_0_STREAM);
    fclose(_106_CONV2D_13_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _107_Sigmoid_28_output_0 = (float*) (mem_0 + _107_SIGMOID_28_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_107_SIGMOID_28_NB_ELTS>
                               (_106_Conv2D_13_output_0, _107_Sigmoid_28_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _107_Sigmoid_28\n");
    FILE* _107_SIGMOID_28_OUTPUT_0_STREAM = fopen("data/export_outputs/_107_Sigmoid_28_output_0.txt", "w");
    export_cpp::saveOutputs<_107_SIGMOID_28_OUTPUT_0_NB_OUTPUTS,
                            _107_SIGMOID_28_OUTPUT_0_OUT_HEIGHT,
                            _107_SIGMOID_28_OUTPUT_0_OUT_WIDTH,
                            _107_SIGMOID_28_OUTPUT_0_MEM_CONT_OFFSET,
                            _107_SIGMOID_28_OUTPUT_0_MEM_CONT_SIZE,
                            _107_SIGMOID_28_OUTPUT_0_MEM_WRAP_OFFSET,
                            _107_SIGMOID_28_OUTPUT_0_MEM_WRAP_SIZE,
                            _107_SIGMOID_28_OUTPUT_0_DEV_FMT>
                            (_107_Sigmoid_28_output_0,
                            _107_SIGMOID_28_OUTPUT_0_STREAM);
    fclose(_107_SIGMOID_28_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _108_Mul_28_output_0 = (float*) (mem_0 + _108_MUL_28_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_108_MUL_28_NB_MAT,
                                 _108_MUL_28_ELEM_OP,
                                 _108_MUL_28_INPUT1_CONT_SIZE,
                                 _108_MUL_28_INPUT2_CONT_SIZE,
                                 _108_MUL_28_OUTPUT_CONT_SIZE,
                                 _108_MUL_28_OFFSET_IN1,
                                 _108_MUL_28_OFFSET_IN2,
                                 _108_MUL_28_ACTIVATION,
                                 _106_CONV2D_13_OUTPUT_0_MEM_CONT_OFFSET,
                                 _106_CONV2D_13_OUTPUT_0_MEM_CONT_SIZE,
                                 _106_CONV2D_13_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _106_CONV2D_13_OUTPUT_0_MEM_WRAP_SIZE,
                                 _106_CONV2D_13_OUTPUT_0_MEM_STRIDE,
                                 _107_SIGMOID_28_OUTPUT_0_MEM_CONT_OFFSET,
                                 _107_SIGMOID_28_OUTPUT_0_MEM_CONT_SIZE,
                                 _107_SIGMOID_28_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _107_SIGMOID_28_OUTPUT_0_MEM_WRAP_SIZE,
                                 _107_SIGMOID_28_OUTPUT_0_MEM_STRIDE,
                                 _108_MUL_28_OUTPUT_0_MEM_CONT_OFFSET,
                                 _108_MUL_28_OUTPUT_0_MEM_CONT_SIZE,
                                 _108_MUL_28_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _108_MUL_28_OUTPUT_0_MEM_WRAP_SIZE,
                                 _108_MUL_28_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_108_Mul_28_output_0, _108_MUL_28_RESCALING, _106_Conv2D_13_output_0, _107_Sigmoid_28_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _108_Mul_28\n");
    FILE* _108_MUL_28_OUTPUT_0_STREAM = fopen("data/export_outputs/_108_Mul_28_output_0.txt", "w");
    export_cpp::saveOutputs<_108_MUL_28_OUTPUT_0_NB_OUTPUTS,
                            _108_MUL_28_OUTPUT_0_OUT_HEIGHT,
                            _108_MUL_28_OUTPUT_0_OUT_WIDTH,
                            _108_MUL_28_OUTPUT_0_MEM_CONT_OFFSET,
                            _108_MUL_28_OUTPUT_0_MEM_CONT_SIZE,
                            _108_MUL_28_OUTPUT_0_MEM_WRAP_OFFSET,
                            _108_MUL_28_OUTPUT_0_MEM_WRAP_SIZE,
                            _108_MUL_28_OUTPUT_0_DEV_FMT>
                            (_108_Mul_28_output_0,
                            _108_MUL_28_OUTPUT_0_STREAM);
    fclose(_108_MUL_28_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _109_Conv2D_14_output_0 = (float*) (mem_0 + _109_CONV2D_14_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_108_MUL_28_OUTPUT_0_IN_BATCH,
                           _109_CONV2D_14_NB_CHANNELS,
                           _108_MUL_28_OUTPUT_0_IN_HEIGHT,
                           _108_MUL_28_OUTPUT_0_IN_WIDTH,
                           _109_CONV2D_14_NB_OUTPUTS,
                           _109_CONV2D_14_OUTPUT_0_OUT_HEIGHT,
                           _109_CONV2D_14_OUTPUT_0_OUT_WIDTH,
                           _109_CONV2D_14_PADDING_Y,
                           _109_CONV2D_14_PADDING_X,
                           _109_CONV2D_14_STRIDE_Y,
                           _109_CONV2D_14_STRIDE_X,
                           _109_CONV2D_14_DILATION_Y,
                           _109_CONV2D_14_DILATION_X,
                           _109_CONV2D_14_KERNEL_HEIGHT,
                           _109_CONV2D_14_KERNEL_WIDTH,
                           _109_CONV2D_14_ACTIVATION,
                           _108_MUL_28_OUTPUT_0_MEM_CONT_OFFSET,
                           _108_MUL_28_OUTPUT_0_MEM_CONT_SIZE,
                           _108_MUL_28_OUTPUT_0_MEM_WRAP_OFFSET,
                           _108_MUL_28_OUTPUT_0_MEM_WRAP_SIZE,
                           _108_MUL_28_OUTPUT_0_MEM_STRIDE,
                           _109_CONV2D_14_OUTPUT_0_MEM_CONT_OFFSET,
                           _109_CONV2D_14_OUTPUT_0_MEM_CONT_SIZE,
                           _109_CONV2D_14_OUTPUT_0_MEM_WRAP_OFFSET,
                           _109_CONV2D_14_OUTPUT_0_MEM_WRAP_SIZE,
                           _109_CONV2D_14_OUTPUT_0_MEM_STRIDE>
                           (_108_Mul_28_output_0, _109_Conv2D_14_output_0, _109_Conv2D_14_weights_output_0, _109_Conv2D_14_biases_output_0, _109_CONV2D_14_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _109_Conv2D_14\n");
    FILE* _109_CONV2D_14_OUTPUT_0_STREAM = fopen("data/export_outputs/_109_Conv2D_14_output_0.txt", "w");
    export_cpp::saveOutputs<_109_CONV2D_14_OUTPUT_0_NB_OUTPUTS,
                            _109_CONV2D_14_OUTPUT_0_OUT_HEIGHT,
                            _109_CONV2D_14_OUTPUT_0_OUT_WIDTH,
                            _109_CONV2D_14_OUTPUT_0_MEM_CONT_OFFSET,
                            _109_CONV2D_14_OUTPUT_0_MEM_CONT_SIZE,
                            _109_CONV2D_14_OUTPUT_0_MEM_WRAP_OFFSET,
                            _109_CONV2D_14_OUTPUT_0_MEM_WRAP_SIZE,
                            _109_CONV2D_14_OUTPUT_0_DEV_FMT>
                            (_109_Conv2D_14_output_0,
                            _109_CONV2D_14_OUTPUT_0_STREAM);
    fclose(_109_CONV2D_14_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _110_Add_7_output_0 = (float*) (mem_0 + _110_ADD_7_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_110_ADD_7_NB_MAT,
                                 _110_ADD_7_ELEM_OP,
                                 _110_ADD_7_INPUT1_CONT_SIZE,
                                 _110_ADD_7_INPUT2_CONT_SIZE,
                                 _110_ADD_7_OUTPUT_CONT_SIZE,
                                 _110_ADD_7_OFFSET_IN1,
                                 _110_ADD_7_OFFSET_IN2,
                                 _110_ADD_7_ACTIVATION,
                                 _105_ADD_6_OUTPUT_0_MEM_CONT_OFFSET,
                                 _105_ADD_6_OUTPUT_0_MEM_CONT_SIZE,
                                 _105_ADD_6_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _105_ADD_6_OUTPUT_0_MEM_WRAP_SIZE,
                                 _105_ADD_6_OUTPUT_0_MEM_STRIDE,
                                 _109_CONV2D_14_OUTPUT_0_MEM_CONT_OFFSET,
                                 _109_CONV2D_14_OUTPUT_0_MEM_CONT_SIZE,
                                 _109_CONV2D_14_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _109_CONV2D_14_OUTPUT_0_MEM_WRAP_SIZE,
                                 _109_CONV2D_14_OUTPUT_0_MEM_STRIDE,
                                 _110_ADD_7_OUTPUT_0_MEM_CONT_OFFSET,
                                 _110_ADD_7_OUTPUT_0_MEM_CONT_SIZE,
                                 _110_ADD_7_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _110_ADD_7_OUTPUT_0_MEM_WRAP_SIZE,
                                 _110_ADD_7_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_110_Add_7_output_0, _110_ADD_7_RESCALING, _105_Add_6_output_0, _109_Conv2D_14_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _110_Add_7\n");
    FILE* _110_ADD_7_OUTPUT_0_STREAM = fopen("data/export_outputs/_110_Add_7_output_0.txt", "w");
    export_cpp::saveOutputs<_110_ADD_7_OUTPUT_0_NB_OUTPUTS,
                            _110_ADD_7_OUTPUT_0_OUT_HEIGHT,
                            _110_ADD_7_OUTPUT_0_OUT_WIDTH,
                            _110_ADD_7_OUTPUT_0_MEM_CONT_OFFSET,
                            _110_ADD_7_OUTPUT_0_MEM_CONT_SIZE,
                            _110_ADD_7_OUTPUT_0_MEM_WRAP_OFFSET,
                            _110_ADD_7_OUTPUT_0_MEM_WRAP_SIZE,
                            _110_ADD_7_OUTPUT_0_DEV_FMT>
                            (_110_Add_7_output_0,
                            _110_ADD_7_OUTPUT_0_STREAM);
    fclose(_110_ADD_7_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _111_Concat_5_output_0 = (float*) (mem_0 + _111_CONCAT_5_OUTPUT_0_MEM_OFFSET);

    const float* _111_Concat_5_inputs[] = { _104_Split_4_output_0, _110_Add_7_output_0 };
    export_cpp::concat_forward<_111_CONCAT_5_AXIS_SIZE_POST,
                               _111_CONCAT_5_AXIS_SIZE_PRE,
                               _111_CONCAT_5_AXIS_SIZE,
                               _111_CONCAT_5_TOTAL_AXIS_SIZE,
                               2> 
                              (_111_Concat_5_inputs,
                               _111_Concat_5_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _111_Concat_5\n");
    FILE* _111_CONCAT_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_111_Concat_5_output_0.txt", "w");
    export_cpp::saveOutputs<_111_CONCAT_5_OUTPUT_0_NB_OUTPUTS,
                            _111_CONCAT_5_OUTPUT_0_OUT_HEIGHT,
                            _111_CONCAT_5_OUTPUT_0_OUT_WIDTH,
                            _111_CONCAT_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _111_CONCAT_5_OUTPUT_0_MEM_CONT_SIZE,
                            _111_CONCAT_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _111_CONCAT_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _111_CONCAT_5_OUTPUT_0_DEV_FMT>
                            (_111_Concat_5_output_0,
                            _111_CONCAT_5_OUTPUT_0_STREAM);
    fclose(_111_CONCAT_5_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _112_Conv2D_15_output_0 = (float*) (mem_0 + _112_CONV2D_15_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_111_CONCAT_5_OUTPUT_0_IN_BATCH,
                           _112_CONV2D_15_NB_CHANNELS,
                           _111_CONCAT_5_OUTPUT_0_IN_HEIGHT,
                           _111_CONCAT_5_OUTPUT_0_IN_WIDTH,
                           _112_CONV2D_15_NB_OUTPUTS,
                           _112_CONV2D_15_OUTPUT_0_OUT_HEIGHT,
                           _112_CONV2D_15_OUTPUT_0_OUT_WIDTH,
                           _112_CONV2D_15_PADDING_Y,
                           _112_CONV2D_15_PADDING_X,
                           _112_CONV2D_15_STRIDE_Y,
                           _112_CONV2D_15_STRIDE_X,
                           _112_CONV2D_15_DILATION_Y,
                           _112_CONV2D_15_DILATION_X,
                           _112_CONV2D_15_KERNEL_HEIGHT,
                           _112_CONV2D_15_KERNEL_WIDTH,
                           _112_CONV2D_15_ACTIVATION,
                           _111_CONCAT_5_OUTPUT_0_MEM_CONT_OFFSET,
                           _111_CONCAT_5_OUTPUT_0_MEM_CONT_SIZE,
                           _111_CONCAT_5_OUTPUT_0_MEM_WRAP_OFFSET,
                           _111_CONCAT_5_OUTPUT_0_MEM_WRAP_SIZE,
                           _111_CONCAT_5_OUTPUT_0_MEM_STRIDE,
                           _112_CONV2D_15_OUTPUT_0_MEM_CONT_OFFSET,
                           _112_CONV2D_15_OUTPUT_0_MEM_CONT_SIZE,
                           _112_CONV2D_15_OUTPUT_0_MEM_WRAP_OFFSET,
                           _112_CONV2D_15_OUTPUT_0_MEM_WRAP_SIZE,
                           _112_CONV2D_15_OUTPUT_0_MEM_STRIDE>
                           (_111_Concat_5_output_0, _112_Conv2D_15_output_0, _112_Conv2D_15_weights_output_0, _112_Conv2D_15_biases_output_0, _112_CONV2D_15_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _112_Conv2D_15\n");
    FILE* _112_CONV2D_15_OUTPUT_0_STREAM = fopen("data/export_outputs/_112_Conv2D_15_output_0.txt", "w");
    export_cpp::saveOutputs<_112_CONV2D_15_OUTPUT_0_NB_OUTPUTS,
                            _112_CONV2D_15_OUTPUT_0_OUT_HEIGHT,
                            _112_CONV2D_15_OUTPUT_0_OUT_WIDTH,
                            _112_CONV2D_15_OUTPUT_0_MEM_CONT_OFFSET,
                            _112_CONV2D_15_OUTPUT_0_MEM_CONT_SIZE,
                            _112_CONV2D_15_OUTPUT_0_MEM_WRAP_OFFSET,
                            _112_CONV2D_15_OUTPUT_0_MEM_WRAP_SIZE,
                            _112_CONV2D_15_OUTPUT_0_DEV_FMT>
                            (_112_Conv2D_15_output_0,
                            _112_CONV2D_15_OUTPUT_0_STREAM);
    fclose(_112_CONV2D_15_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _113_Sigmoid_29_output_0 = (float*) (mem_0 + _113_SIGMOID_29_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_113_SIGMOID_29_NB_ELTS>
                               (_112_Conv2D_15_output_0, _113_Sigmoid_29_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _113_Sigmoid_29\n");
    FILE* _113_SIGMOID_29_OUTPUT_0_STREAM = fopen("data/export_outputs/_113_Sigmoid_29_output_0.txt", "w");
    export_cpp::saveOutputs<_113_SIGMOID_29_OUTPUT_0_NB_OUTPUTS,
                            _113_SIGMOID_29_OUTPUT_0_OUT_HEIGHT,
                            _113_SIGMOID_29_OUTPUT_0_OUT_WIDTH,
                            _113_SIGMOID_29_OUTPUT_0_MEM_CONT_OFFSET,
                            _113_SIGMOID_29_OUTPUT_0_MEM_CONT_SIZE,
                            _113_SIGMOID_29_OUTPUT_0_MEM_WRAP_OFFSET,
                            _113_SIGMOID_29_OUTPUT_0_MEM_WRAP_SIZE,
                            _113_SIGMOID_29_OUTPUT_0_DEV_FMT>
                            (_113_Sigmoid_29_output_0,
                            _113_SIGMOID_29_OUTPUT_0_STREAM);
    fclose(_113_SIGMOID_29_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _114_Mul_29_output_0 = (float*) (mem_0 + _114_MUL_29_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_114_MUL_29_NB_MAT,
                                 _114_MUL_29_ELEM_OP,
                                 _114_MUL_29_INPUT1_CONT_SIZE,
                                 _114_MUL_29_INPUT2_CONT_SIZE,
                                 _114_MUL_29_OUTPUT_CONT_SIZE,
                                 _114_MUL_29_OFFSET_IN1,
                                 _114_MUL_29_OFFSET_IN2,
                                 _114_MUL_29_ACTIVATION,
                                 _112_CONV2D_15_OUTPUT_0_MEM_CONT_OFFSET,
                                 _112_CONV2D_15_OUTPUT_0_MEM_CONT_SIZE,
                                 _112_CONV2D_15_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _112_CONV2D_15_OUTPUT_0_MEM_WRAP_SIZE,
                                 _112_CONV2D_15_OUTPUT_0_MEM_STRIDE,
                                 _113_SIGMOID_29_OUTPUT_0_MEM_CONT_OFFSET,
                                 _113_SIGMOID_29_OUTPUT_0_MEM_CONT_SIZE,
                                 _113_SIGMOID_29_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _113_SIGMOID_29_OUTPUT_0_MEM_WRAP_SIZE,
                                 _113_SIGMOID_29_OUTPUT_0_MEM_STRIDE,
                                 _114_MUL_29_OUTPUT_0_MEM_CONT_OFFSET,
                                 _114_MUL_29_OUTPUT_0_MEM_CONT_SIZE,
                                 _114_MUL_29_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _114_MUL_29_OUTPUT_0_MEM_WRAP_SIZE,
                                 _114_MUL_29_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_114_Mul_29_output_0, _114_MUL_29_RESCALING, _112_Conv2D_15_output_0, _113_Sigmoid_29_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _114_Mul_29\n");
    FILE* _114_MUL_29_OUTPUT_0_STREAM = fopen("data/export_outputs/_114_Mul_29_output_0.txt", "w");
    export_cpp::saveOutputs<_114_MUL_29_OUTPUT_0_NB_OUTPUTS,
                            _114_MUL_29_OUTPUT_0_OUT_HEIGHT,
                            _114_MUL_29_OUTPUT_0_OUT_WIDTH,
                            _114_MUL_29_OUTPUT_0_MEM_CONT_OFFSET,
                            _114_MUL_29_OUTPUT_0_MEM_CONT_SIZE,
                            _114_MUL_29_OUTPUT_0_MEM_WRAP_OFFSET,
                            _114_MUL_29_OUTPUT_0_MEM_WRAP_SIZE,
                            _114_MUL_29_OUTPUT_0_DEV_FMT>
                            (_114_Mul_29_output_0,
                            _114_MUL_29_OUTPUT_0_STREAM);
    fclose(_114_MUL_29_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _115_Resize_0_output_0 = (float*) (mem_0 + _115_RESIZE_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::resize_forward<_114_MUL_29_OUTPUT_0_IN_BATCH,
                               _114_MUL_29_OUTPUT_0_NB_CHANNELS,
                               _114_MUL_29_OUTPUT_0_IN_HEIGHT, 
                               _114_MUL_29_OUTPUT_0_IN_WIDTH,
                               _115_RESIZE_0_OUTPUT_0_NB_OUTPUTS,
                               _115_RESIZE_0_OUTPUT_0_OUT_HEIGHT,
                               _115_RESIZE_0_OUTPUT_0_OUT_WIDTH,
                               _114_MUL_29_OUTPUT_0_MEM_CONT_OFFSET,
                               _114_MUL_29_OUTPUT_0_MEM_CONT_SIZE,
                               _114_MUL_29_OUTPUT_0_MEM_WRAP_OFFSET,
                               _114_MUL_29_OUTPUT_0_MEM_WRAP_SIZE,
                               _114_MUL_29_OUTPUT_0_MEM_STRIDE,
                               _115_RESIZE_0_OUTPUT_0_MEM_CONT_OFFSET,
                               _115_RESIZE_0_OUTPUT_0_MEM_CONT_SIZE,
                               _115_RESIZE_0_OUTPUT_0_MEM_WRAP_OFFSET,
                               _115_RESIZE_0_OUTPUT_0_MEM_WRAP_SIZE,
                               _115_RESIZE_0_OUTPUT_0_MEM_STRIDE,
                               _115_RESIZE_0_ANTIALIAS,
                               _115_RESIZE_0_EXCLUDE_OUTSIDE,
                               float,
                               float,
                               _115_RESIZE_0_INTERPOLATION_MODE,
                               _115_RESIZE_0_COORDTRANS>
                               (_114_Mul_29_output_0, _115_Resize_0_output_0,
                                _115_RESIZE_0_CUBICA);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _115_Resize_0\n");
    FILE* _115_RESIZE_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_115_Resize_0_output_0.txt", "w");
    export_cpp::saveOutputs<_115_RESIZE_0_OUTPUT_0_NB_OUTPUTS,
                            _115_RESIZE_0_OUTPUT_0_OUT_HEIGHT,
                            _115_RESIZE_0_OUTPUT_0_OUT_WIDTH,
                            _115_RESIZE_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _115_RESIZE_0_OUTPUT_0_MEM_CONT_SIZE,
                            _115_RESIZE_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _115_RESIZE_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _115_RESIZE_0_OUTPUT_0_DEV_FMT>
                            (_115_Resize_0_output_0,
                            _115_RESIZE_0_OUTPUT_0_STREAM);
    fclose(_115_RESIZE_0_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _116_Concat_6_output_0 = (float*) (mem_0 + _116_CONCAT_6_OUTPUT_0_MEM_OFFSET);

    const float* _116_Concat_6_inputs[] = { _115_Resize_0_output_0, _71_Mul_19_output_0 };
    export_cpp::concat_forward<_116_CONCAT_6_AXIS_SIZE_POST,
                               _116_CONCAT_6_AXIS_SIZE_PRE,
                               _116_CONCAT_6_AXIS_SIZE,
                               _116_CONCAT_6_TOTAL_AXIS_SIZE,
                               2> 
                              (_116_Concat_6_inputs,
                               _116_Concat_6_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _116_Concat_6\n");
    FILE* _116_CONCAT_6_OUTPUT_0_STREAM = fopen("data/export_outputs/_116_Concat_6_output_0.txt", "w");
    export_cpp::saveOutputs<_116_CONCAT_6_OUTPUT_0_NB_OUTPUTS,
                            _116_CONCAT_6_OUTPUT_0_OUT_HEIGHT,
                            _116_CONCAT_6_OUTPUT_0_OUT_WIDTH,
                            _116_CONCAT_6_OUTPUT_0_MEM_CONT_OFFSET,
                            _116_CONCAT_6_OUTPUT_0_MEM_CONT_SIZE,
                            _116_CONCAT_6_OUTPUT_0_MEM_WRAP_OFFSET,
                            _116_CONCAT_6_OUTPUT_0_MEM_WRAP_SIZE,
                            _116_CONCAT_6_OUTPUT_0_DEV_FMT>
                            (_116_Concat_6_output_0,
                            _116_CONCAT_6_OUTPUT_0_STREAM);
    fclose(_116_CONCAT_6_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _117_Conv2D_16_output_0 = (float*) (mem_0 + _117_CONV2D_16_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_116_CONCAT_6_OUTPUT_0_IN_BATCH,
                           _117_CONV2D_16_NB_CHANNELS,
                           _116_CONCAT_6_OUTPUT_0_IN_HEIGHT,
                           _116_CONCAT_6_OUTPUT_0_IN_WIDTH,
                           _117_CONV2D_16_NB_OUTPUTS,
                           _117_CONV2D_16_OUTPUT_0_OUT_HEIGHT,
                           _117_CONV2D_16_OUTPUT_0_OUT_WIDTH,
                           _117_CONV2D_16_PADDING_Y,
                           _117_CONV2D_16_PADDING_X,
                           _117_CONV2D_16_STRIDE_Y,
                           _117_CONV2D_16_STRIDE_X,
                           _117_CONV2D_16_DILATION_Y,
                           _117_CONV2D_16_DILATION_X,
                           _117_CONV2D_16_KERNEL_HEIGHT,
                           _117_CONV2D_16_KERNEL_WIDTH,
                           _117_CONV2D_16_ACTIVATION,
                           _116_CONCAT_6_OUTPUT_0_MEM_CONT_OFFSET,
                           _116_CONCAT_6_OUTPUT_0_MEM_CONT_SIZE,
                           _116_CONCAT_6_OUTPUT_0_MEM_WRAP_OFFSET,
                           _116_CONCAT_6_OUTPUT_0_MEM_WRAP_SIZE,
                           _116_CONCAT_6_OUTPUT_0_MEM_STRIDE,
                           _117_CONV2D_16_OUTPUT_0_MEM_CONT_OFFSET,
                           _117_CONV2D_16_OUTPUT_0_MEM_CONT_SIZE,
                           _117_CONV2D_16_OUTPUT_0_MEM_WRAP_OFFSET,
                           _117_CONV2D_16_OUTPUT_0_MEM_WRAP_SIZE,
                           _117_CONV2D_16_OUTPUT_0_MEM_STRIDE>
                           (_116_Concat_6_output_0, _117_Conv2D_16_output_0, _117_Conv2D_16_weights_output_0, _117_Conv2D_16_biases_output_0, _117_CONV2D_16_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _117_Conv2D_16\n");
    FILE* _117_CONV2D_16_OUTPUT_0_STREAM = fopen("data/export_outputs/_117_Conv2D_16_output_0.txt", "w");
    export_cpp::saveOutputs<_117_CONV2D_16_OUTPUT_0_NB_OUTPUTS,
                            _117_CONV2D_16_OUTPUT_0_OUT_HEIGHT,
                            _117_CONV2D_16_OUTPUT_0_OUT_WIDTH,
                            _117_CONV2D_16_OUTPUT_0_MEM_CONT_OFFSET,
                            _117_CONV2D_16_OUTPUT_0_MEM_CONT_SIZE,
                            _117_CONV2D_16_OUTPUT_0_MEM_WRAP_OFFSET,
                            _117_CONV2D_16_OUTPUT_0_MEM_WRAP_SIZE,
                            _117_CONV2D_16_OUTPUT_0_DEV_FMT>
                            (_117_Conv2D_16_output_0,
                            _117_CONV2D_16_OUTPUT_0_STREAM);
    fclose(_117_CONV2D_16_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _118_Sigmoid_30_output_0 = (float*) (mem_0 + _118_SIGMOID_30_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_118_SIGMOID_30_NB_ELTS>
                               (_117_Conv2D_16_output_0, _118_Sigmoid_30_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _118_Sigmoid_30\n");
    FILE* _118_SIGMOID_30_OUTPUT_0_STREAM = fopen("data/export_outputs/_118_Sigmoid_30_output_0.txt", "w");
    export_cpp::saveOutputs<_118_SIGMOID_30_OUTPUT_0_NB_OUTPUTS,
                            _118_SIGMOID_30_OUTPUT_0_OUT_HEIGHT,
                            _118_SIGMOID_30_OUTPUT_0_OUT_WIDTH,
                            _118_SIGMOID_30_OUTPUT_0_MEM_CONT_OFFSET,
                            _118_SIGMOID_30_OUTPUT_0_MEM_CONT_SIZE,
                            _118_SIGMOID_30_OUTPUT_0_MEM_WRAP_OFFSET,
                            _118_SIGMOID_30_OUTPUT_0_MEM_WRAP_SIZE,
                            _118_SIGMOID_30_OUTPUT_0_DEV_FMT>
                            (_118_Sigmoid_30_output_0,
                            _118_SIGMOID_30_OUTPUT_0_STREAM);
    fclose(_118_SIGMOID_30_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _119_Mul_30_output_0 = (float*) (mem_0 + _119_MUL_30_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_119_MUL_30_NB_MAT,
                                 _119_MUL_30_ELEM_OP,
                                 _119_MUL_30_INPUT1_CONT_SIZE,
                                 _119_MUL_30_INPUT2_CONT_SIZE,
                                 _119_MUL_30_OUTPUT_CONT_SIZE,
                                 _119_MUL_30_OFFSET_IN1,
                                 _119_MUL_30_OFFSET_IN2,
                                 _119_MUL_30_ACTIVATION,
                                 _117_CONV2D_16_OUTPUT_0_MEM_CONT_OFFSET,
                                 _117_CONV2D_16_OUTPUT_0_MEM_CONT_SIZE,
                                 _117_CONV2D_16_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _117_CONV2D_16_OUTPUT_0_MEM_WRAP_SIZE,
                                 _117_CONV2D_16_OUTPUT_0_MEM_STRIDE,
                                 _118_SIGMOID_30_OUTPUT_0_MEM_CONT_OFFSET,
                                 _118_SIGMOID_30_OUTPUT_0_MEM_CONT_SIZE,
                                 _118_SIGMOID_30_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _118_SIGMOID_30_OUTPUT_0_MEM_WRAP_SIZE,
                                 _118_SIGMOID_30_OUTPUT_0_MEM_STRIDE,
                                 _119_MUL_30_OUTPUT_0_MEM_CONT_OFFSET,
                                 _119_MUL_30_OUTPUT_0_MEM_CONT_SIZE,
                                 _119_MUL_30_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _119_MUL_30_OUTPUT_0_MEM_WRAP_SIZE,
                                 _119_MUL_30_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_119_Mul_30_output_0, _119_MUL_30_RESCALING, _117_Conv2D_16_output_0, _118_Sigmoid_30_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _119_Mul_30\n");
    FILE* _119_MUL_30_OUTPUT_0_STREAM = fopen("data/export_outputs/_119_Mul_30_output_0.txt", "w");
    export_cpp::saveOutputs<_119_MUL_30_OUTPUT_0_NB_OUTPUTS,
                            _119_MUL_30_OUTPUT_0_OUT_HEIGHT,
                            _119_MUL_30_OUTPUT_0_OUT_WIDTH,
                            _119_MUL_30_OUTPUT_0_MEM_CONT_OFFSET,
                            _119_MUL_30_OUTPUT_0_MEM_CONT_SIZE,
                            _119_MUL_30_OUTPUT_0_MEM_WRAP_OFFSET,
                            _119_MUL_30_OUTPUT_0_MEM_WRAP_SIZE,
                            _119_MUL_30_OUTPUT_0_DEV_FMT>
                            (_119_Mul_30_output_0,
                            _119_MUL_30_OUTPUT_0_STREAM);
    fclose(_119_MUL_30_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _120_Split_5_output_0 = (float*) (mem_0 + _120_SPLIT_5_OUTPUT_0_MEM_OFFSET);
    float* _120_Split_5_output_1 = (float*) (mem_0 + _120_SPLIT_5_OUTPUT_1_MEM_OFFSET);



    float* __120_Split_5_outputs[_120_SPLIT_5_NB_OUTPUTS] = {
        
        _120_Split_5_output_0,
        
        _120_Split_5_output_1
        
    };

    export_cpp::split_forward<_120_SPLIT_5_NB_OUTPUTS,
                              _120_SPLIT_5_AXIS_DIM,
                              _120_SPLIT_5_PRE_AXIS_ELEMS,
                              _120_SPLIT_5_POST_AXIS_ELEMS,
                              _120_SPLIT_5_SPLITS,
                              float>
                              (_119_Mul_30_output_0,
                               __120_Split_5_outputs);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _120_Split_5\n");
    FILE* _120_SPLIT_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_120_Split_5_output_0.txt", "w");
    export_cpp::saveOutputs<_120_SPLIT_5_OUTPUT_0_NB_OUTPUTS,
                            _120_SPLIT_5_OUTPUT_0_OUT_HEIGHT,
                            _120_SPLIT_5_OUTPUT_0_OUT_WIDTH,
                            _120_SPLIT_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _120_SPLIT_5_OUTPUT_0_MEM_CONT_SIZE,
                            _120_SPLIT_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _120_SPLIT_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _120_SPLIT_5_OUTPUT_0_DEV_FMT>
                            (_120_Split_5_output_0,
                            _120_SPLIT_5_OUTPUT_0_STREAM);
    fclose(_120_SPLIT_5_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _121_PadConv_15_output_0 = (float*) (mem_0 + _121_PADCONV_15_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_120_SPLIT_5_OUTPUT_1_IN_BATCH,
                           _121_PADCONV_15_NB_CHANNELS,
                           _120_SPLIT_5_OUTPUT_1_IN_HEIGHT,
                           _120_SPLIT_5_OUTPUT_1_IN_WIDTH,
                           _121_PADCONV_15_NB_OUTPUTS,
                           _121_PADCONV_15_OUTPUT_0_OUT_HEIGHT,
                           _121_PADCONV_15_OUTPUT_0_OUT_WIDTH,
                           _121_PADCONV_15_PADDING_Y,
                           _121_PADCONV_15_PADDING_X,
                           _121_PADCONV_15_STRIDE_Y,
                           _121_PADCONV_15_STRIDE_X,
                           _121_PADCONV_15_DILATION_Y,
                           _121_PADCONV_15_DILATION_X,
                           _121_PADCONV_15_KERNEL_HEIGHT,
                           _121_PADCONV_15_KERNEL_WIDTH,
                           _121_PADCONV_15_ACTIVATION,
                           _120_SPLIT_5_OUTPUT_1_MEM_CONT_OFFSET,
                           _120_SPLIT_5_OUTPUT_1_MEM_CONT_SIZE,
                           _120_SPLIT_5_OUTPUT_1_MEM_WRAP_OFFSET,
                           _120_SPLIT_5_OUTPUT_1_MEM_WRAP_SIZE,
                           _120_SPLIT_5_OUTPUT_1_MEM_STRIDE,
                           _121_PADCONV_15_OUTPUT_0_MEM_CONT_OFFSET,
                           _121_PADCONV_15_OUTPUT_0_MEM_CONT_SIZE,
                           _121_PADCONV_15_OUTPUT_0_MEM_WRAP_OFFSET,
                           _121_PADCONV_15_OUTPUT_0_MEM_WRAP_SIZE,
                           _121_PADCONV_15_OUTPUT_0_MEM_STRIDE>
                           (_120_Split_5_output_1, _121_PadConv_15_output_0, _121_PadConv_15_weights_output_0, _121_PadConv_15_biases_output_0, _121_PADCONV_15_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _121_PadConv_15\n");
    FILE* _121_PADCONV_15_OUTPUT_0_STREAM = fopen("data/export_outputs/_121_PadConv_15_output_0.txt", "w");
    export_cpp::saveOutputs<_121_PADCONV_15_OUTPUT_0_NB_OUTPUTS,
                            _121_PADCONV_15_OUTPUT_0_OUT_HEIGHT,
                            _121_PADCONV_15_OUTPUT_0_OUT_WIDTH,
                            _121_PADCONV_15_OUTPUT_0_MEM_CONT_OFFSET,
                            _121_PADCONV_15_OUTPUT_0_MEM_CONT_SIZE,
                            _121_PADCONV_15_OUTPUT_0_MEM_WRAP_OFFSET,
                            _121_PADCONV_15_OUTPUT_0_MEM_WRAP_SIZE,
                            _121_PADCONV_15_OUTPUT_0_DEV_FMT>
                            (_121_PadConv_15_output_0,
                            _121_PADCONV_15_OUTPUT_0_STREAM);
    fclose(_121_PADCONV_15_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _122_Sigmoid_31_output_0 = (float*) (mem_0 + _122_SIGMOID_31_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_122_SIGMOID_31_NB_ELTS>
                               (_121_PadConv_15_output_0, _122_Sigmoid_31_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _122_Sigmoid_31\n");
    FILE* _122_SIGMOID_31_OUTPUT_0_STREAM = fopen("data/export_outputs/_122_Sigmoid_31_output_0.txt", "w");
    export_cpp::saveOutputs<_122_SIGMOID_31_OUTPUT_0_NB_OUTPUTS,
                            _122_SIGMOID_31_OUTPUT_0_OUT_HEIGHT,
                            _122_SIGMOID_31_OUTPUT_0_OUT_WIDTH,
                            _122_SIGMOID_31_OUTPUT_0_MEM_CONT_OFFSET,
                            _122_SIGMOID_31_OUTPUT_0_MEM_CONT_SIZE,
                            _122_SIGMOID_31_OUTPUT_0_MEM_WRAP_OFFSET,
                            _122_SIGMOID_31_OUTPUT_0_MEM_WRAP_SIZE,
                            _122_SIGMOID_31_OUTPUT_0_DEV_FMT>
                            (_122_Sigmoid_31_output_0,
                            _122_SIGMOID_31_OUTPUT_0_STREAM);
    fclose(_122_SIGMOID_31_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _123_Mul_31_output_0 = (float*) (mem_0 + _123_MUL_31_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_123_MUL_31_NB_MAT,
                                 _123_MUL_31_ELEM_OP,
                                 _123_MUL_31_INPUT1_CONT_SIZE,
                                 _123_MUL_31_INPUT2_CONT_SIZE,
                                 _123_MUL_31_OUTPUT_CONT_SIZE,
                                 _123_MUL_31_OFFSET_IN1,
                                 _123_MUL_31_OFFSET_IN2,
                                 _123_MUL_31_ACTIVATION,
                                 _121_PADCONV_15_OUTPUT_0_MEM_CONT_OFFSET,
                                 _121_PADCONV_15_OUTPUT_0_MEM_CONT_SIZE,
                                 _121_PADCONV_15_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _121_PADCONV_15_OUTPUT_0_MEM_WRAP_SIZE,
                                 _121_PADCONV_15_OUTPUT_0_MEM_STRIDE,
                                 _122_SIGMOID_31_OUTPUT_0_MEM_CONT_OFFSET,
                                 _122_SIGMOID_31_OUTPUT_0_MEM_CONT_SIZE,
                                 _122_SIGMOID_31_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _122_SIGMOID_31_OUTPUT_0_MEM_WRAP_SIZE,
                                 _122_SIGMOID_31_OUTPUT_0_MEM_STRIDE,
                                 _123_MUL_31_OUTPUT_0_MEM_CONT_OFFSET,
                                 _123_MUL_31_OUTPUT_0_MEM_CONT_SIZE,
                                 _123_MUL_31_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _123_MUL_31_OUTPUT_0_MEM_WRAP_SIZE,
                                 _123_MUL_31_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_123_Mul_31_output_0, _123_MUL_31_RESCALING, _121_PadConv_15_output_0, _122_Sigmoid_31_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _123_Mul_31\n");
    FILE* _123_MUL_31_OUTPUT_0_STREAM = fopen("data/export_outputs/_123_Mul_31_output_0.txt", "w");
    export_cpp::saveOutputs<_123_MUL_31_OUTPUT_0_NB_OUTPUTS,
                            _123_MUL_31_OUTPUT_0_OUT_HEIGHT,
                            _123_MUL_31_OUTPUT_0_OUT_WIDTH,
                            _123_MUL_31_OUTPUT_0_MEM_CONT_OFFSET,
                            _123_MUL_31_OUTPUT_0_MEM_CONT_SIZE,
                            _123_MUL_31_OUTPUT_0_MEM_WRAP_OFFSET,
                            _123_MUL_31_OUTPUT_0_MEM_WRAP_SIZE,
                            _123_MUL_31_OUTPUT_0_DEV_FMT>
                            (_123_Mul_31_output_0,
                            _123_MUL_31_OUTPUT_0_STREAM);
    fclose(_123_MUL_31_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _124_PadConv_16_output_0 = (float*) (mem_0 + _124_PADCONV_16_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_123_MUL_31_OUTPUT_0_IN_BATCH,
                           _124_PADCONV_16_NB_CHANNELS,
                           _123_MUL_31_OUTPUT_0_IN_HEIGHT,
                           _123_MUL_31_OUTPUT_0_IN_WIDTH,
                           _124_PADCONV_16_NB_OUTPUTS,
                           _124_PADCONV_16_OUTPUT_0_OUT_HEIGHT,
                           _124_PADCONV_16_OUTPUT_0_OUT_WIDTH,
                           _124_PADCONV_16_PADDING_Y,
                           _124_PADCONV_16_PADDING_X,
                           _124_PADCONV_16_STRIDE_Y,
                           _124_PADCONV_16_STRIDE_X,
                           _124_PADCONV_16_DILATION_Y,
                           _124_PADCONV_16_DILATION_X,
                           _124_PADCONV_16_KERNEL_HEIGHT,
                           _124_PADCONV_16_KERNEL_WIDTH,
                           _124_PADCONV_16_ACTIVATION,
                           _123_MUL_31_OUTPUT_0_MEM_CONT_OFFSET,
                           _123_MUL_31_OUTPUT_0_MEM_CONT_SIZE,
                           _123_MUL_31_OUTPUT_0_MEM_WRAP_OFFSET,
                           _123_MUL_31_OUTPUT_0_MEM_WRAP_SIZE,
                           _123_MUL_31_OUTPUT_0_MEM_STRIDE,
                           _124_PADCONV_16_OUTPUT_0_MEM_CONT_OFFSET,
                           _124_PADCONV_16_OUTPUT_0_MEM_CONT_SIZE,
                           _124_PADCONV_16_OUTPUT_0_MEM_WRAP_OFFSET,
                           _124_PADCONV_16_OUTPUT_0_MEM_WRAP_SIZE,
                           _124_PADCONV_16_OUTPUT_0_MEM_STRIDE>
                           (_123_Mul_31_output_0, _124_PadConv_16_output_0, _124_PadConv_16_weights_output_0, _124_PadConv_16_biases_output_0, _124_PADCONV_16_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _124_PadConv_16\n");
    FILE* _124_PADCONV_16_OUTPUT_0_STREAM = fopen("data/export_outputs/_124_PadConv_16_output_0.txt", "w");
    export_cpp::saveOutputs<_124_PADCONV_16_OUTPUT_0_NB_OUTPUTS,
                            _124_PADCONV_16_OUTPUT_0_OUT_HEIGHT,
                            _124_PADCONV_16_OUTPUT_0_OUT_WIDTH,
                            _124_PADCONV_16_OUTPUT_0_MEM_CONT_OFFSET,
                            _124_PADCONV_16_OUTPUT_0_MEM_CONT_SIZE,
                            _124_PADCONV_16_OUTPUT_0_MEM_WRAP_OFFSET,
                            _124_PADCONV_16_OUTPUT_0_MEM_WRAP_SIZE,
                            _124_PADCONV_16_OUTPUT_0_DEV_FMT>
                            (_124_PadConv_16_output_0,
                            _124_PADCONV_16_OUTPUT_0_STREAM);
    fclose(_124_PADCONV_16_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _125_Sigmoid_32_output_0 = (float*) (mem_0 + _125_SIGMOID_32_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_125_SIGMOID_32_NB_ELTS>
                               (_124_PadConv_16_output_0, _125_Sigmoid_32_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _125_Sigmoid_32\n");
    FILE* _125_SIGMOID_32_OUTPUT_0_STREAM = fopen("data/export_outputs/_125_Sigmoid_32_output_0.txt", "w");
    export_cpp::saveOutputs<_125_SIGMOID_32_OUTPUT_0_NB_OUTPUTS,
                            _125_SIGMOID_32_OUTPUT_0_OUT_HEIGHT,
                            _125_SIGMOID_32_OUTPUT_0_OUT_WIDTH,
                            _125_SIGMOID_32_OUTPUT_0_MEM_CONT_OFFSET,
                            _125_SIGMOID_32_OUTPUT_0_MEM_CONT_SIZE,
                            _125_SIGMOID_32_OUTPUT_0_MEM_WRAP_OFFSET,
                            _125_SIGMOID_32_OUTPUT_0_MEM_WRAP_SIZE,
                            _125_SIGMOID_32_OUTPUT_0_DEV_FMT>
                            (_125_Sigmoid_32_output_0,
                            _125_SIGMOID_32_OUTPUT_0_STREAM);
    fclose(_125_SIGMOID_32_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _126_Mul_32_output_0 = (float*) (mem_0 + _126_MUL_32_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_126_MUL_32_NB_MAT,
                                 _126_MUL_32_ELEM_OP,
                                 _126_MUL_32_INPUT1_CONT_SIZE,
                                 _126_MUL_32_INPUT2_CONT_SIZE,
                                 _126_MUL_32_OUTPUT_CONT_SIZE,
                                 _126_MUL_32_OFFSET_IN1,
                                 _126_MUL_32_OFFSET_IN2,
                                 _126_MUL_32_ACTIVATION,
                                 _124_PADCONV_16_OUTPUT_0_MEM_CONT_OFFSET,
                                 _124_PADCONV_16_OUTPUT_0_MEM_CONT_SIZE,
                                 _124_PADCONV_16_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _124_PADCONV_16_OUTPUT_0_MEM_WRAP_SIZE,
                                 _124_PADCONV_16_OUTPUT_0_MEM_STRIDE,
                                 _125_SIGMOID_32_OUTPUT_0_MEM_CONT_OFFSET,
                                 _125_SIGMOID_32_OUTPUT_0_MEM_CONT_SIZE,
                                 _125_SIGMOID_32_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _125_SIGMOID_32_OUTPUT_0_MEM_WRAP_SIZE,
                                 _125_SIGMOID_32_OUTPUT_0_MEM_STRIDE,
                                 _126_MUL_32_OUTPUT_0_MEM_CONT_OFFSET,
                                 _126_MUL_32_OUTPUT_0_MEM_CONT_SIZE,
                                 _126_MUL_32_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _126_MUL_32_OUTPUT_0_MEM_WRAP_SIZE,
                                 _126_MUL_32_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_126_Mul_32_output_0, _126_MUL_32_RESCALING, _124_PadConv_16_output_0, _125_Sigmoid_32_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _126_Mul_32\n");
    FILE* _126_MUL_32_OUTPUT_0_STREAM = fopen("data/export_outputs/_126_Mul_32_output_0.txt", "w");
    export_cpp::saveOutputs<_126_MUL_32_OUTPUT_0_NB_OUTPUTS,
                            _126_MUL_32_OUTPUT_0_OUT_HEIGHT,
                            _126_MUL_32_OUTPUT_0_OUT_WIDTH,
                            _126_MUL_32_OUTPUT_0_MEM_CONT_OFFSET,
                            _126_MUL_32_OUTPUT_0_MEM_CONT_SIZE,
                            _126_MUL_32_OUTPUT_0_MEM_WRAP_OFFSET,
                            _126_MUL_32_OUTPUT_0_MEM_WRAP_SIZE,
                            _126_MUL_32_OUTPUT_0_DEV_FMT>
                            (_126_Mul_32_output_0,
                            _126_MUL_32_OUTPUT_0_STREAM);
    fclose(_126_MUL_32_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _127_Concat_7_output_0 = (float*) (mem_0 + _127_CONCAT_7_OUTPUT_0_MEM_OFFSET);

    const float* _127_Concat_7_inputs[] = { _120_Split_5_output_0, _120_Split_5_output_1, _126_Mul_32_output_0 };
    export_cpp::concat_forward<_127_CONCAT_7_AXIS_SIZE_POST,
                               _127_CONCAT_7_AXIS_SIZE_PRE,
                               _127_CONCAT_7_AXIS_SIZE,
                               _127_CONCAT_7_TOTAL_AXIS_SIZE,
                               3> 
                              (_127_Concat_7_inputs,
                               _127_Concat_7_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _127_Concat_7\n");
    FILE* _127_CONCAT_7_OUTPUT_0_STREAM = fopen("data/export_outputs/_127_Concat_7_output_0.txt", "w");
    export_cpp::saveOutputs<_127_CONCAT_7_OUTPUT_0_NB_OUTPUTS,
                            _127_CONCAT_7_OUTPUT_0_OUT_HEIGHT,
                            _127_CONCAT_7_OUTPUT_0_OUT_WIDTH,
                            _127_CONCAT_7_OUTPUT_0_MEM_CONT_OFFSET,
                            _127_CONCAT_7_OUTPUT_0_MEM_CONT_SIZE,
                            _127_CONCAT_7_OUTPUT_0_MEM_WRAP_OFFSET,
                            _127_CONCAT_7_OUTPUT_0_MEM_WRAP_SIZE,
                            _127_CONCAT_7_OUTPUT_0_DEV_FMT>
                            (_127_Concat_7_output_0,
                            _127_CONCAT_7_OUTPUT_0_STREAM);
    fclose(_127_CONCAT_7_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _128_Conv2D_17_output_0 = (float*) (mem_0 + _128_CONV2D_17_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_127_CONCAT_7_OUTPUT_0_IN_BATCH,
                           _128_CONV2D_17_NB_CHANNELS,
                           _127_CONCAT_7_OUTPUT_0_IN_HEIGHT,
                           _127_CONCAT_7_OUTPUT_0_IN_WIDTH,
                           _128_CONV2D_17_NB_OUTPUTS,
                           _128_CONV2D_17_OUTPUT_0_OUT_HEIGHT,
                           _128_CONV2D_17_OUTPUT_0_OUT_WIDTH,
                           _128_CONV2D_17_PADDING_Y,
                           _128_CONV2D_17_PADDING_X,
                           _128_CONV2D_17_STRIDE_Y,
                           _128_CONV2D_17_STRIDE_X,
                           _128_CONV2D_17_DILATION_Y,
                           _128_CONV2D_17_DILATION_X,
                           _128_CONV2D_17_KERNEL_HEIGHT,
                           _128_CONV2D_17_KERNEL_WIDTH,
                           _128_CONV2D_17_ACTIVATION,
                           _127_CONCAT_7_OUTPUT_0_MEM_CONT_OFFSET,
                           _127_CONCAT_7_OUTPUT_0_MEM_CONT_SIZE,
                           _127_CONCAT_7_OUTPUT_0_MEM_WRAP_OFFSET,
                           _127_CONCAT_7_OUTPUT_0_MEM_WRAP_SIZE,
                           _127_CONCAT_7_OUTPUT_0_MEM_STRIDE,
                           _128_CONV2D_17_OUTPUT_0_MEM_CONT_OFFSET,
                           _128_CONV2D_17_OUTPUT_0_MEM_CONT_SIZE,
                           _128_CONV2D_17_OUTPUT_0_MEM_WRAP_OFFSET,
                           _128_CONV2D_17_OUTPUT_0_MEM_WRAP_SIZE,
                           _128_CONV2D_17_OUTPUT_0_MEM_STRIDE>
                           (_127_Concat_7_output_0, _128_Conv2D_17_output_0, _128_Conv2D_17_weights_output_0, _128_Conv2D_17_biases_output_0, _128_CONV2D_17_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _128_Conv2D_17\n");
    FILE* _128_CONV2D_17_OUTPUT_0_STREAM = fopen("data/export_outputs/_128_Conv2D_17_output_0.txt", "w");
    export_cpp::saveOutputs<_128_CONV2D_17_OUTPUT_0_NB_OUTPUTS,
                            _128_CONV2D_17_OUTPUT_0_OUT_HEIGHT,
                            _128_CONV2D_17_OUTPUT_0_OUT_WIDTH,
                            _128_CONV2D_17_OUTPUT_0_MEM_CONT_OFFSET,
                            _128_CONV2D_17_OUTPUT_0_MEM_CONT_SIZE,
                            _128_CONV2D_17_OUTPUT_0_MEM_WRAP_OFFSET,
                            _128_CONV2D_17_OUTPUT_0_MEM_WRAP_SIZE,
                            _128_CONV2D_17_OUTPUT_0_DEV_FMT>
                            (_128_Conv2D_17_output_0,
                            _128_CONV2D_17_OUTPUT_0_STREAM);
    fclose(_128_CONV2D_17_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _129_Sigmoid_33_output_0 = (float*) (mem_0 + _129_SIGMOID_33_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_129_SIGMOID_33_NB_ELTS>
                               (_128_Conv2D_17_output_0, _129_Sigmoid_33_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _129_Sigmoid_33\n");
    FILE* _129_SIGMOID_33_OUTPUT_0_STREAM = fopen("data/export_outputs/_129_Sigmoid_33_output_0.txt", "w");
    export_cpp::saveOutputs<_129_SIGMOID_33_OUTPUT_0_NB_OUTPUTS,
                            _129_SIGMOID_33_OUTPUT_0_OUT_HEIGHT,
                            _129_SIGMOID_33_OUTPUT_0_OUT_WIDTH,
                            _129_SIGMOID_33_OUTPUT_0_MEM_CONT_OFFSET,
                            _129_SIGMOID_33_OUTPUT_0_MEM_CONT_SIZE,
                            _129_SIGMOID_33_OUTPUT_0_MEM_WRAP_OFFSET,
                            _129_SIGMOID_33_OUTPUT_0_MEM_WRAP_SIZE,
                            _129_SIGMOID_33_OUTPUT_0_DEV_FMT>
                            (_129_Sigmoid_33_output_0,
                            _129_SIGMOID_33_OUTPUT_0_STREAM);
    fclose(_129_SIGMOID_33_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _130_Mul_33_output_0 = (float*) (mem_0 + _130_MUL_33_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_130_MUL_33_NB_MAT,
                                 _130_MUL_33_ELEM_OP,
                                 _130_MUL_33_INPUT1_CONT_SIZE,
                                 _130_MUL_33_INPUT2_CONT_SIZE,
                                 _130_MUL_33_OUTPUT_CONT_SIZE,
                                 _130_MUL_33_OFFSET_IN1,
                                 _130_MUL_33_OFFSET_IN2,
                                 _130_MUL_33_ACTIVATION,
                                 _128_CONV2D_17_OUTPUT_0_MEM_CONT_OFFSET,
                                 _128_CONV2D_17_OUTPUT_0_MEM_CONT_SIZE,
                                 _128_CONV2D_17_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _128_CONV2D_17_OUTPUT_0_MEM_WRAP_SIZE,
                                 _128_CONV2D_17_OUTPUT_0_MEM_STRIDE,
                                 _129_SIGMOID_33_OUTPUT_0_MEM_CONT_OFFSET,
                                 _129_SIGMOID_33_OUTPUT_0_MEM_CONT_SIZE,
                                 _129_SIGMOID_33_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _129_SIGMOID_33_OUTPUT_0_MEM_WRAP_SIZE,
                                 _129_SIGMOID_33_OUTPUT_0_MEM_STRIDE,
                                 _130_MUL_33_OUTPUT_0_MEM_CONT_OFFSET,
                                 _130_MUL_33_OUTPUT_0_MEM_CONT_SIZE,
                                 _130_MUL_33_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _130_MUL_33_OUTPUT_0_MEM_WRAP_SIZE,
                                 _130_MUL_33_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_130_Mul_33_output_0, _130_MUL_33_RESCALING, _128_Conv2D_17_output_0, _129_Sigmoid_33_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _130_Mul_33\n");
    FILE* _130_MUL_33_OUTPUT_0_STREAM = fopen("data/export_outputs/_130_Mul_33_output_0.txt", "w");
    export_cpp::saveOutputs<_130_MUL_33_OUTPUT_0_NB_OUTPUTS,
                            _130_MUL_33_OUTPUT_0_OUT_HEIGHT,
                            _130_MUL_33_OUTPUT_0_OUT_WIDTH,
                            _130_MUL_33_OUTPUT_0_MEM_CONT_OFFSET,
                            _130_MUL_33_OUTPUT_0_MEM_CONT_SIZE,
                            _130_MUL_33_OUTPUT_0_MEM_WRAP_OFFSET,
                            _130_MUL_33_OUTPUT_0_MEM_WRAP_SIZE,
                            _130_MUL_33_OUTPUT_0_DEV_FMT>
                            (_130_Mul_33_output_0,
                            _130_MUL_33_OUTPUT_0_STREAM);
    fclose(_130_MUL_33_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _131_Resize_1_output_0 = (float*) (mem_0 + _131_RESIZE_1_OUTPUT_0_MEM_OFFSET);

    export_cpp::resize_forward<_130_MUL_33_OUTPUT_0_IN_BATCH,
                               _130_MUL_33_OUTPUT_0_NB_CHANNELS,
                               _130_MUL_33_OUTPUT_0_IN_HEIGHT, 
                               _130_MUL_33_OUTPUT_0_IN_WIDTH,
                               _131_RESIZE_1_OUTPUT_0_NB_OUTPUTS,
                               _131_RESIZE_1_OUTPUT_0_OUT_HEIGHT,
                               _131_RESIZE_1_OUTPUT_0_OUT_WIDTH,
                               _130_MUL_33_OUTPUT_0_MEM_CONT_OFFSET,
                               _130_MUL_33_OUTPUT_0_MEM_CONT_SIZE,
                               _130_MUL_33_OUTPUT_0_MEM_WRAP_OFFSET,
                               _130_MUL_33_OUTPUT_0_MEM_WRAP_SIZE,
                               _130_MUL_33_OUTPUT_0_MEM_STRIDE,
                               _131_RESIZE_1_OUTPUT_0_MEM_CONT_OFFSET,
                               _131_RESIZE_1_OUTPUT_0_MEM_CONT_SIZE,
                               _131_RESIZE_1_OUTPUT_0_MEM_WRAP_OFFSET,
                               _131_RESIZE_1_OUTPUT_0_MEM_WRAP_SIZE,
                               _131_RESIZE_1_OUTPUT_0_MEM_STRIDE,
                               _131_RESIZE_1_ANTIALIAS,
                               _131_RESIZE_1_EXCLUDE_OUTSIDE,
                               float,
                               float,
                               _131_RESIZE_1_INTERPOLATION_MODE,
                               _131_RESIZE_1_COORDTRANS>
                               (_130_Mul_33_output_0, _131_Resize_1_output_0,
                                _131_RESIZE_1_CUBICA);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _131_Resize_1\n");
    FILE* _131_RESIZE_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_131_Resize_1_output_0.txt", "w");
    export_cpp::saveOutputs<_131_RESIZE_1_OUTPUT_0_NB_OUTPUTS,
                            _131_RESIZE_1_OUTPUT_0_OUT_HEIGHT,
                            _131_RESIZE_1_OUTPUT_0_OUT_WIDTH,
                            _131_RESIZE_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _131_RESIZE_1_OUTPUT_0_MEM_CONT_SIZE,
                            _131_RESIZE_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _131_RESIZE_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _131_RESIZE_1_OUTPUT_0_DEV_FMT>
                            (_131_Resize_1_output_0,
                            _131_RESIZE_1_OUTPUT_0_STREAM);
    fclose(_131_RESIZE_1_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _132_Concat_8_output_0 = (float*) (mem_0 + _132_CONCAT_8_OUTPUT_0_MEM_OFFSET);

    const float* _132_Concat_8_inputs[] = { _131_Resize_1_output_0, _45_Mul_12_output_0 };
    export_cpp::concat_forward<_132_CONCAT_8_AXIS_SIZE_POST,
                               _132_CONCAT_8_AXIS_SIZE_PRE,
                               _132_CONCAT_8_AXIS_SIZE,
                               _132_CONCAT_8_TOTAL_AXIS_SIZE,
                               2> 
                              (_132_Concat_8_inputs,
                               _132_Concat_8_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _132_Concat_8\n");
    FILE* _132_CONCAT_8_OUTPUT_0_STREAM = fopen("data/export_outputs/_132_Concat_8_output_0.txt", "w");
    export_cpp::saveOutputs<_132_CONCAT_8_OUTPUT_0_NB_OUTPUTS,
                            _132_CONCAT_8_OUTPUT_0_OUT_HEIGHT,
                            _132_CONCAT_8_OUTPUT_0_OUT_WIDTH,
                            _132_CONCAT_8_OUTPUT_0_MEM_CONT_OFFSET,
                            _132_CONCAT_8_OUTPUT_0_MEM_CONT_SIZE,
                            _132_CONCAT_8_OUTPUT_0_MEM_WRAP_OFFSET,
                            _132_CONCAT_8_OUTPUT_0_MEM_WRAP_SIZE,
                            _132_CONCAT_8_OUTPUT_0_DEV_FMT>
                            (_132_Concat_8_output_0,
                            _132_CONCAT_8_OUTPUT_0_STREAM);
    fclose(_132_CONCAT_8_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _133_Conv2D_18_output_0 = (float*) (mem_0 + _133_CONV2D_18_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_132_CONCAT_8_OUTPUT_0_IN_BATCH,
                           _133_CONV2D_18_NB_CHANNELS,
                           _132_CONCAT_8_OUTPUT_0_IN_HEIGHT,
                           _132_CONCAT_8_OUTPUT_0_IN_WIDTH,
                           _133_CONV2D_18_NB_OUTPUTS,
                           _133_CONV2D_18_OUTPUT_0_OUT_HEIGHT,
                           _133_CONV2D_18_OUTPUT_0_OUT_WIDTH,
                           _133_CONV2D_18_PADDING_Y,
                           _133_CONV2D_18_PADDING_X,
                           _133_CONV2D_18_STRIDE_Y,
                           _133_CONV2D_18_STRIDE_X,
                           _133_CONV2D_18_DILATION_Y,
                           _133_CONV2D_18_DILATION_X,
                           _133_CONV2D_18_KERNEL_HEIGHT,
                           _133_CONV2D_18_KERNEL_WIDTH,
                           _133_CONV2D_18_ACTIVATION,
                           _132_CONCAT_8_OUTPUT_0_MEM_CONT_OFFSET,
                           _132_CONCAT_8_OUTPUT_0_MEM_CONT_SIZE,
                           _132_CONCAT_8_OUTPUT_0_MEM_WRAP_OFFSET,
                           _132_CONCAT_8_OUTPUT_0_MEM_WRAP_SIZE,
                           _132_CONCAT_8_OUTPUT_0_MEM_STRIDE,
                           _133_CONV2D_18_OUTPUT_0_MEM_CONT_OFFSET,
                           _133_CONV2D_18_OUTPUT_0_MEM_CONT_SIZE,
                           _133_CONV2D_18_OUTPUT_0_MEM_WRAP_OFFSET,
                           _133_CONV2D_18_OUTPUT_0_MEM_WRAP_SIZE,
                           _133_CONV2D_18_OUTPUT_0_MEM_STRIDE>
                           (_132_Concat_8_output_0, _133_Conv2D_18_output_0, _133_Conv2D_18_weights_output_0, _133_Conv2D_18_biases_output_0, _133_CONV2D_18_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _133_Conv2D_18\n");
    FILE* _133_CONV2D_18_OUTPUT_0_STREAM = fopen("data/export_outputs/_133_Conv2D_18_output_0.txt", "w");
    export_cpp::saveOutputs<_133_CONV2D_18_OUTPUT_0_NB_OUTPUTS,
                            _133_CONV2D_18_OUTPUT_0_OUT_HEIGHT,
                            _133_CONV2D_18_OUTPUT_0_OUT_WIDTH,
                            _133_CONV2D_18_OUTPUT_0_MEM_CONT_OFFSET,
                            _133_CONV2D_18_OUTPUT_0_MEM_CONT_SIZE,
                            _133_CONV2D_18_OUTPUT_0_MEM_WRAP_OFFSET,
                            _133_CONV2D_18_OUTPUT_0_MEM_WRAP_SIZE,
                            _133_CONV2D_18_OUTPUT_0_DEV_FMT>
                            (_133_Conv2D_18_output_0,
                            _133_CONV2D_18_OUTPUT_0_STREAM);
    fclose(_133_CONV2D_18_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _134_Sigmoid_34_output_0 = (float*) (mem_0 + _134_SIGMOID_34_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_134_SIGMOID_34_NB_ELTS>
                               (_133_Conv2D_18_output_0, _134_Sigmoid_34_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _134_Sigmoid_34\n");
    FILE* _134_SIGMOID_34_OUTPUT_0_STREAM = fopen("data/export_outputs/_134_Sigmoid_34_output_0.txt", "w");
    export_cpp::saveOutputs<_134_SIGMOID_34_OUTPUT_0_NB_OUTPUTS,
                            _134_SIGMOID_34_OUTPUT_0_OUT_HEIGHT,
                            _134_SIGMOID_34_OUTPUT_0_OUT_WIDTH,
                            _134_SIGMOID_34_OUTPUT_0_MEM_CONT_OFFSET,
                            _134_SIGMOID_34_OUTPUT_0_MEM_CONT_SIZE,
                            _134_SIGMOID_34_OUTPUT_0_MEM_WRAP_OFFSET,
                            _134_SIGMOID_34_OUTPUT_0_MEM_WRAP_SIZE,
                            _134_SIGMOID_34_OUTPUT_0_DEV_FMT>
                            (_134_Sigmoid_34_output_0,
                            _134_SIGMOID_34_OUTPUT_0_STREAM);
    fclose(_134_SIGMOID_34_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _135_Mul_34_output_0 = (float*) (mem_0 + _135_MUL_34_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_135_MUL_34_NB_MAT,
                                 _135_MUL_34_ELEM_OP,
                                 _135_MUL_34_INPUT1_CONT_SIZE,
                                 _135_MUL_34_INPUT2_CONT_SIZE,
                                 _135_MUL_34_OUTPUT_CONT_SIZE,
                                 _135_MUL_34_OFFSET_IN1,
                                 _135_MUL_34_OFFSET_IN2,
                                 _135_MUL_34_ACTIVATION,
                                 _133_CONV2D_18_OUTPUT_0_MEM_CONT_OFFSET,
                                 _133_CONV2D_18_OUTPUT_0_MEM_CONT_SIZE,
                                 _133_CONV2D_18_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _133_CONV2D_18_OUTPUT_0_MEM_WRAP_SIZE,
                                 _133_CONV2D_18_OUTPUT_0_MEM_STRIDE,
                                 _134_SIGMOID_34_OUTPUT_0_MEM_CONT_OFFSET,
                                 _134_SIGMOID_34_OUTPUT_0_MEM_CONT_SIZE,
                                 _134_SIGMOID_34_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _134_SIGMOID_34_OUTPUT_0_MEM_WRAP_SIZE,
                                 _134_SIGMOID_34_OUTPUT_0_MEM_STRIDE,
                                 _135_MUL_34_OUTPUT_0_MEM_CONT_OFFSET,
                                 _135_MUL_34_OUTPUT_0_MEM_CONT_SIZE,
                                 _135_MUL_34_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _135_MUL_34_OUTPUT_0_MEM_WRAP_SIZE,
                                 _135_MUL_34_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_135_Mul_34_output_0, _135_MUL_34_RESCALING, _133_Conv2D_18_output_0, _134_Sigmoid_34_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _135_Mul_34\n");
    FILE* _135_MUL_34_OUTPUT_0_STREAM = fopen("data/export_outputs/_135_Mul_34_output_0.txt", "w");
    export_cpp::saveOutputs<_135_MUL_34_OUTPUT_0_NB_OUTPUTS,
                            _135_MUL_34_OUTPUT_0_OUT_HEIGHT,
                            _135_MUL_34_OUTPUT_0_OUT_WIDTH,
                            _135_MUL_34_OUTPUT_0_MEM_CONT_OFFSET,
                            _135_MUL_34_OUTPUT_0_MEM_CONT_SIZE,
                            _135_MUL_34_OUTPUT_0_MEM_WRAP_OFFSET,
                            _135_MUL_34_OUTPUT_0_MEM_WRAP_SIZE,
                            _135_MUL_34_OUTPUT_0_DEV_FMT>
                            (_135_Mul_34_output_0,
                            _135_MUL_34_OUTPUT_0_STREAM);
    fclose(_135_MUL_34_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _136_Split_6_output_0 = (float*) (mem_0 + _136_SPLIT_6_OUTPUT_0_MEM_OFFSET);
    float* _136_Split_6_output_1 = (float*) (mem_0 + _136_SPLIT_6_OUTPUT_1_MEM_OFFSET);



    float* __136_Split_6_outputs[_136_SPLIT_6_NB_OUTPUTS] = {
        
        _136_Split_6_output_0,
        
        _136_Split_6_output_1
        
    };

    export_cpp::split_forward<_136_SPLIT_6_NB_OUTPUTS,
                              _136_SPLIT_6_AXIS_DIM,
                              _136_SPLIT_6_PRE_AXIS_ELEMS,
                              _136_SPLIT_6_POST_AXIS_ELEMS,
                              _136_SPLIT_6_SPLITS,
                              float>
                              (_135_Mul_34_output_0,
                               __136_Split_6_outputs);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _136_Split_6\n");
    FILE* _136_SPLIT_6_OUTPUT_0_STREAM = fopen("data/export_outputs/_136_Split_6_output_0.txt", "w");
    export_cpp::saveOutputs<_136_SPLIT_6_OUTPUT_0_NB_OUTPUTS,
                            _136_SPLIT_6_OUTPUT_0_OUT_HEIGHT,
                            _136_SPLIT_6_OUTPUT_0_OUT_WIDTH,
                            _136_SPLIT_6_OUTPUT_0_MEM_CONT_OFFSET,
                            _136_SPLIT_6_OUTPUT_0_MEM_CONT_SIZE,
                            _136_SPLIT_6_OUTPUT_0_MEM_WRAP_OFFSET,
                            _136_SPLIT_6_OUTPUT_0_MEM_WRAP_SIZE,
                            _136_SPLIT_6_OUTPUT_0_DEV_FMT>
                            (_136_Split_6_output_0,
                            _136_SPLIT_6_OUTPUT_0_STREAM);
    fclose(_136_SPLIT_6_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _137_PadConv_17_output_0 = (float*) (mem_0 + _137_PADCONV_17_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_136_SPLIT_6_OUTPUT_1_IN_BATCH,
                           _137_PADCONV_17_NB_CHANNELS,
                           _136_SPLIT_6_OUTPUT_1_IN_HEIGHT,
                           _136_SPLIT_6_OUTPUT_1_IN_WIDTH,
                           _137_PADCONV_17_NB_OUTPUTS,
                           _137_PADCONV_17_OUTPUT_0_OUT_HEIGHT,
                           _137_PADCONV_17_OUTPUT_0_OUT_WIDTH,
                           _137_PADCONV_17_PADDING_Y,
                           _137_PADCONV_17_PADDING_X,
                           _137_PADCONV_17_STRIDE_Y,
                           _137_PADCONV_17_STRIDE_X,
                           _137_PADCONV_17_DILATION_Y,
                           _137_PADCONV_17_DILATION_X,
                           _137_PADCONV_17_KERNEL_HEIGHT,
                           _137_PADCONV_17_KERNEL_WIDTH,
                           _137_PADCONV_17_ACTIVATION,
                           _136_SPLIT_6_OUTPUT_1_MEM_CONT_OFFSET,
                           _136_SPLIT_6_OUTPUT_1_MEM_CONT_SIZE,
                           _136_SPLIT_6_OUTPUT_1_MEM_WRAP_OFFSET,
                           _136_SPLIT_6_OUTPUT_1_MEM_WRAP_SIZE,
                           _136_SPLIT_6_OUTPUT_1_MEM_STRIDE,
                           _137_PADCONV_17_OUTPUT_0_MEM_CONT_OFFSET,
                           _137_PADCONV_17_OUTPUT_0_MEM_CONT_SIZE,
                           _137_PADCONV_17_OUTPUT_0_MEM_WRAP_OFFSET,
                           _137_PADCONV_17_OUTPUT_0_MEM_WRAP_SIZE,
                           _137_PADCONV_17_OUTPUT_0_MEM_STRIDE>
                           (_136_Split_6_output_1, _137_PadConv_17_output_0, _137_PadConv_17_weights_output_0, _137_PadConv_17_biases_output_0, _137_PADCONV_17_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _137_PadConv_17\n");
    FILE* _137_PADCONV_17_OUTPUT_0_STREAM = fopen("data/export_outputs/_137_PadConv_17_output_0.txt", "w");
    export_cpp::saveOutputs<_137_PADCONV_17_OUTPUT_0_NB_OUTPUTS,
                            _137_PADCONV_17_OUTPUT_0_OUT_HEIGHT,
                            _137_PADCONV_17_OUTPUT_0_OUT_WIDTH,
                            _137_PADCONV_17_OUTPUT_0_MEM_CONT_OFFSET,
                            _137_PADCONV_17_OUTPUT_0_MEM_CONT_SIZE,
                            _137_PADCONV_17_OUTPUT_0_MEM_WRAP_OFFSET,
                            _137_PADCONV_17_OUTPUT_0_MEM_WRAP_SIZE,
                            _137_PADCONV_17_OUTPUT_0_DEV_FMT>
                            (_137_PadConv_17_output_0,
                            _137_PADCONV_17_OUTPUT_0_STREAM);
    fclose(_137_PADCONV_17_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _138_Sigmoid_35_output_0 = (float*) (mem_0 + _138_SIGMOID_35_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_138_SIGMOID_35_NB_ELTS>
                               (_137_PadConv_17_output_0, _138_Sigmoid_35_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _138_Sigmoid_35\n");
    FILE* _138_SIGMOID_35_OUTPUT_0_STREAM = fopen("data/export_outputs/_138_Sigmoid_35_output_0.txt", "w");
    export_cpp::saveOutputs<_138_SIGMOID_35_OUTPUT_0_NB_OUTPUTS,
                            _138_SIGMOID_35_OUTPUT_0_OUT_HEIGHT,
                            _138_SIGMOID_35_OUTPUT_0_OUT_WIDTH,
                            _138_SIGMOID_35_OUTPUT_0_MEM_CONT_OFFSET,
                            _138_SIGMOID_35_OUTPUT_0_MEM_CONT_SIZE,
                            _138_SIGMOID_35_OUTPUT_0_MEM_WRAP_OFFSET,
                            _138_SIGMOID_35_OUTPUT_0_MEM_WRAP_SIZE,
                            _138_SIGMOID_35_OUTPUT_0_DEV_FMT>
                            (_138_Sigmoid_35_output_0,
                            _138_SIGMOID_35_OUTPUT_0_STREAM);
    fclose(_138_SIGMOID_35_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _139_Mul_35_output_0 = (float*) (mem_0 + _139_MUL_35_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_139_MUL_35_NB_MAT,
                                 _139_MUL_35_ELEM_OP,
                                 _139_MUL_35_INPUT1_CONT_SIZE,
                                 _139_MUL_35_INPUT2_CONT_SIZE,
                                 _139_MUL_35_OUTPUT_CONT_SIZE,
                                 _139_MUL_35_OFFSET_IN1,
                                 _139_MUL_35_OFFSET_IN2,
                                 _139_MUL_35_ACTIVATION,
                                 _137_PADCONV_17_OUTPUT_0_MEM_CONT_OFFSET,
                                 _137_PADCONV_17_OUTPUT_0_MEM_CONT_SIZE,
                                 _137_PADCONV_17_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _137_PADCONV_17_OUTPUT_0_MEM_WRAP_SIZE,
                                 _137_PADCONV_17_OUTPUT_0_MEM_STRIDE,
                                 _138_SIGMOID_35_OUTPUT_0_MEM_CONT_OFFSET,
                                 _138_SIGMOID_35_OUTPUT_0_MEM_CONT_SIZE,
                                 _138_SIGMOID_35_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _138_SIGMOID_35_OUTPUT_0_MEM_WRAP_SIZE,
                                 _138_SIGMOID_35_OUTPUT_0_MEM_STRIDE,
                                 _139_MUL_35_OUTPUT_0_MEM_CONT_OFFSET,
                                 _139_MUL_35_OUTPUT_0_MEM_CONT_SIZE,
                                 _139_MUL_35_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _139_MUL_35_OUTPUT_0_MEM_WRAP_SIZE,
                                 _139_MUL_35_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_139_Mul_35_output_0, _139_MUL_35_RESCALING, _137_PadConv_17_output_0, _138_Sigmoid_35_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _139_Mul_35\n");
    FILE* _139_MUL_35_OUTPUT_0_STREAM = fopen("data/export_outputs/_139_Mul_35_output_0.txt", "w");
    export_cpp::saveOutputs<_139_MUL_35_OUTPUT_0_NB_OUTPUTS,
                            _139_MUL_35_OUTPUT_0_OUT_HEIGHT,
                            _139_MUL_35_OUTPUT_0_OUT_WIDTH,
                            _139_MUL_35_OUTPUT_0_MEM_CONT_OFFSET,
                            _139_MUL_35_OUTPUT_0_MEM_CONT_SIZE,
                            _139_MUL_35_OUTPUT_0_MEM_WRAP_OFFSET,
                            _139_MUL_35_OUTPUT_0_MEM_WRAP_SIZE,
                            _139_MUL_35_OUTPUT_0_DEV_FMT>
                            (_139_Mul_35_output_0,
                            _139_MUL_35_OUTPUT_0_STREAM);
    fclose(_139_MUL_35_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _140_PadConv_18_output_0 = (float*) (mem_0 + _140_PADCONV_18_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_139_MUL_35_OUTPUT_0_IN_BATCH,
                           _140_PADCONV_18_NB_CHANNELS,
                           _139_MUL_35_OUTPUT_0_IN_HEIGHT,
                           _139_MUL_35_OUTPUT_0_IN_WIDTH,
                           _140_PADCONV_18_NB_OUTPUTS,
                           _140_PADCONV_18_OUTPUT_0_OUT_HEIGHT,
                           _140_PADCONV_18_OUTPUT_0_OUT_WIDTH,
                           _140_PADCONV_18_PADDING_Y,
                           _140_PADCONV_18_PADDING_X,
                           _140_PADCONV_18_STRIDE_Y,
                           _140_PADCONV_18_STRIDE_X,
                           _140_PADCONV_18_DILATION_Y,
                           _140_PADCONV_18_DILATION_X,
                           _140_PADCONV_18_KERNEL_HEIGHT,
                           _140_PADCONV_18_KERNEL_WIDTH,
                           _140_PADCONV_18_ACTIVATION,
                           _139_MUL_35_OUTPUT_0_MEM_CONT_OFFSET,
                           _139_MUL_35_OUTPUT_0_MEM_CONT_SIZE,
                           _139_MUL_35_OUTPUT_0_MEM_WRAP_OFFSET,
                           _139_MUL_35_OUTPUT_0_MEM_WRAP_SIZE,
                           _139_MUL_35_OUTPUT_0_MEM_STRIDE,
                           _140_PADCONV_18_OUTPUT_0_MEM_CONT_OFFSET,
                           _140_PADCONV_18_OUTPUT_0_MEM_CONT_SIZE,
                           _140_PADCONV_18_OUTPUT_0_MEM_WRAP_OFFSET,
                           _140_PADCONV_18_OUTPUT_0_MEM_WRAP_SIZE,
                           _140_PADCONV_18_OUTPUT_0_MEM_STRIDE>
                           (_139_Mul_35_output_0, _140_PadConv_18_output_0, _140_PadConv_18_weights_output_0, _140_PadConv_18_biases_output_0, _140_PADCONV_18_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _140_PadConv_18\n");
    FILE* _140_PADCONV_18_OUTPUT_0_STREAM = fopen("data/export_outputs/_140_PadConv_18_output_0.txt", "w");
    export_cpp::saveOutputs<_140_PADCONV_18_OUTPUT_0_NB_OUTPUTS,
                            _140_PADCONV_18_OUTPUT_0_OUT_HEIGHT,
                            _140_PADCONV_18_OUTPUT_0_OUT_WIDTH,
                            _140_PADCONV_18_OUTPUT_0_MEM_CONT_OFFSET,
                            _140_PADCONV_18_OUTPUT_0_MEM_CONT_SIZE,
                            _140_PADCONV_18_OUTPUT_0_MEM_WRAP_OFFSET,
                            _140_PADCONV_18_OUTPUT_0_MEM_WRAP_SIZE,
                            _140_PADCONV_18_OUTPUT_0_DEV_FMT>
                            (_140_PadConv_18_output_0,
                            _140_PADCONV_18_OUTPUT_0_STREAM);
    fclose(_140_PADCONV_18_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _141_Sigmoid_36_output_0 = (float*) (mem_0 + _141_SIGMOID_36_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_141_SIGMOID_36_NB_ELTS>
                               (_140_PadConv_18_output_0, _141_Sigmoid_36_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _141_Sigmoid_36\n");
    FILE* _141_SIGMOID_36_OUTPUT_0_STREAM = fopen("data/export_outputs/_141_Sigmoid_36_output_0.txt", "w");
    export_cpp::saveOutputs<_141_SIGMOID_36_OUTPUT_0_NB_OUTPUTS,
                            _141_SIGMOID_36_OUTPUT_0_OUT_HEIGHT,
                            _141_SIGMOID_36_OUTPUT_0_OUT_WIDTH,
                            _141_SIGMOID_36_OUTPUT_0_MEM_CONT_OFFSET,
                            _141_SIGMOID_36_OUTPUT_0_MEM_CONT_SIZE,
                            _141_SIGMOID_36_OUTPUT_0_MEM_WRAP_OFFSET,
                            _141_SIGMOID_36_OUTPUT_0_MEM_WRAP_SIZE,
                            _141_SIGMOID_36_OUTPUT_0_DEV_FMT>
                            (_141_Sigmoid_36_output_0,
                            _141_SIGMOID_36_OUTPUT_0_STREAM);
    fclose(_141_SIGMOID_36_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _142_Mul_36_output_0 = (float*) (mem_0 + _142_MUL_36_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_142_MUL_36_NB_MAT,
                                 _142_MUL_36_ELEM_OP,
                                 _142_MUL_36_INPUT1_CONT_SIZE,
                                 _142_MUL_36_INPUT2_CONT_SIZE,
                                 _142_MUL_36_OUTPUT_CONT_SIZE,
                                 _142_MUL_36_OFFSET_IN1,
                                 _142_MUL_36_OFFSET_IN2,
                                 _142_MUL_36_ACTIVATION,
                                 _140_PADCONV_18_OUTPUT_0_MEM_CONT_OFFSET,
                                 _140_PADCONV_18_OUTPUT_0_MEM_CONT_SIZE,
                                 _140_PADCONV_18_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _140_PADCONV_18_OUTPUT_0_MEM_WRAP_SIZE,
                                 _140_PADCONV_18_OUTPUT_0_MEM_STRIDE,
                                 _141_SIGMOID_36_OUTPUT_0_MEM_CONT_OFFSET,
                                 _141_SIGMOID_36_OUTPUT_0_MEM_CONT_SIZE,
                                 _141_SIGMOID_36_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _141_SIGMOID_36_OUTPUT_0_MEM_WRAP_SIZE,
                                 _141_SIGMOID_36_OUTPUT_0_MEM_STRIDE,
                                 _142_MUL_36_OUTPUT_0_MEM_CONT_OFFSET,
                                 _142_MUL_36_OUTPUT_0_MEM_CONT_SIZE,
                                 _142_MUL_36_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _142_MUL_36_OUTPUT_0_MEM_WRAP_SIZE,
                                 _142_MUL_36_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_142_Mul_36_output_0, _142_MUL_36_RESCALING, _140_PadConv_18_output_0, _141_Sigmoid_36_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _142_Mul_36\n");
    FILE* _142_MUL_36_OUTPUT_0_STREAM = fopen("data/export_outputs/_142_Mul_36_output_0.txt", "w");
    export_cpp::saveOutputs<_142_MUL_36_OUTPUT_0_NB_OUTPUTS,
                            _142_MUL_36_OUTPUT_0_OUT_HEIGHT,
                            _142_MUL_36_OUTPUT_0_OUT_WIDTH,
                            _142_MUL_36_OUTPUT_0_MEM_CONT_OFFSET,
                            _142_MUL_36_OUTPUT_0_MEM_CONT_SIZE,
                            _142_MUL_36_OUTPUT_0_MEM_WRAP_OFFSET,
                            _142_MUL_36_OUTPUT_0_MEM_WRAP_SIZE,
                            _142_MUL_36_OUTPUT_0_DEV_FMT>
                            (_142_Mul_36_output_0,
                            _142_MUL_36_OUTPUT_0_STREAM);
    fclose(_142_MUL_36_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _143_Concat_9_output_0 = (float*) (mem_0 + _143_CONCAT_9_OUTPUT_0_MEM_OFFSET);

    const float* _143_Concat_9_inputs[] = { _136_Split_6_output_0, _136_Split_6_output_1, _142_Mul_36_output_0 };
    export_cpp::concat_forward<_143_CONCAT_9_AXIS_SIZE_POST,
                               _143_CONCAT_9_AXIS_SIZE_PRE,
                               _143_CONCAT_9_AXIS_SIZE,
                               _143_CONCAT_9_TOTAL_AXIS_SIZE,
                               3> 
                              (_143_Concat_9_inputs,
                               _143_Concat_9_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _143_Concat_9\n");
    FILE* _143_CONCAT_9_OUTPUT_0_STREAM = fopen("data/export_outputs/_143_Concat_9_output_0.txt", "w");
    export_cpp::saveOutputs<_143_CONCAT_9_OUTPUT_0_NB_OUTPUTS,
                            _143_CONCAT_9_OUTPUT_0_OUT_HEIGHT,
                            _143_CONCAT_9_OUTPUT_0_OUT_WIDTH,
                            _143_CONCAT_9_OUTPUT_0_MEM_CONT_OFFSET,
                            _143_CONCAT_9_OUTPUT_0_MEM_CONT_SIZE,
                            _143_CONCAT_9_OUTPUT_0_MEM_WRAP_OFFSET,
                            _143_CONCAT_9_OUTPUT_0_MEM_WRAP_SIZE,
                            _143_CONCAT_9_OUTPUT_0_DEV_FMT>
                            (_143_Concat_9_output_0,
                            _143_CONCAT_9_OUTPUT_0_STREAM);
    fclose(_143_CONCAT_9_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _144_Conv2D_19_output_0 = (float*) (mem_0 + _144_CONV2D_19_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_143_CONCAT_9_OUTPUT_0_IN_BATCH,
                           _144_CONV2D_19_NB_CHANNELS,
                           _143_CONCAT_9_OUTPUT_0_IN_HEIGHT,
                           _143_CONCAT_9_OUTPUT_0_IN_WIDTH,
                           _144_CONV2D_19_NB_OUTPUTS,
                           _144_CONV2D_19_OUTPUT_0_OUT_HEIGHT,
                           _144_CONV2D_19_OUTPUT_0_OUT_WIDTH,
                           _144_CONV2D_19_PADDING_Y,
                           _144_CONV2D_19_PADDING_X,
                           _144_CONV2D_19_STRIDE_Y,
                           _144_CONV2D_19_STRIDE_X,
                           _144_CONV2D_19_DILATION_Y,
                           _144_CONV2D_19_DILATION_X,
                           _144_CONV2D_19_KERNEL_HEIGHT,
                           _144_CONV2D_19_KERNEL_WIDTH,
                           _144_CONV2D_19_ACTIVATION,
                           _143_CONCAT_9_OUTPUT_0_MEM_CONT_OFFSET,
                           _143_CONCAT_9_OUTPUT_0_MEM_CONT_SIZE,
                           _143_CONCAT_9_OUTPUT_0_MEM_WRAP_OFFSET,
                           _143_CONCAT_9_OUTPUT_0_MEM_WRAP_SIZE,
                           _143_CONCAT_9_OUTPUT_0_MEM_STRIDE,
                           _144_CONV2D_19_OUTPUT_0_MEM_CONT_OFFSET,
                           _144_CONV2D_19_OUTPUT_0_MEM_CONT_SIZE,
                           _144_CONV2D_19_OUTPUT_0_MEM_WRAP_OFFSET,
                           _144_CONV2D_19_OUTPUT_0_MEM_WRAP_SIZE,
                           _144_CONV2D_19_OUTPUT_0_MEM_STRIDE>
                           (_143_Concat_9_output_0, _144_Conv2D_19_output_0, _144_Conv2D_19_weights_output_0, _144_Conv2D_19_biases_output_0, _144_CONV2D_19_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _144_Conv2D_19\n");
    FILE* _144_CONV2D_19_OUTPUT_0_STREAM = fopen("data/export_outputs/_144_Conv2D_19_output_0.txt", "w");
    export_cpp::saveOutputs<_144_CONV2D_19_OUTPUT_0_NB_OUTPUTS,
                            _144_CONV2D_19_OUTPUT_0_OUT_HEIGHT,
                            _144_CONV2D_19_OUTPUT_0_OUT_WIDTH,
                            _144_CONV2D_19_OUTPUT_0_MEM_CONT_OFFSET,
                            _144_CONV2D_19_OUTPUT_0_MEM_CONT_SIZE,
                            _144_CONV2D_19_OUTPUT_0_MEM_WRAP_OFFSET,
                            _144_CONV2D_19_OUTPUT_0_MEM_WRAP_SIZE,
                            _144_CONV2D_19_OUTPUT_0_DEV_FMT>
                            (_144_Conv2D_19_output_0,
                            _144_CONV2D_19_OUTPUT_0_STREAM);
    fclose(_144_CONV2D_19_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _145_Sigmoid_37_output_0 = (float*) (mem_0 + _145_SIGMOID_37_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_145_SIGMOID_37_NB_ELTS>
                               (_144_Conv2D_19_output_0, _145_Sigmoid_37_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _145_Sigmoid_37\n");
    FILE* _145_SIGMOID_37_OUTPUT_0_STREAM = fopen("data/export_outputs/_145_Sigmoid_37_output_0.txt", "w");
    export_cpp::saveOutputs<_145_SIGMOID_37_OUTPUT_0_NB_OUTPUTS,
                            _145_SIGMOID_37_OUTPUT_0_OUT_HEIGHT,
                            _145_SIGMOID_37_OUTPUT_0_OUT_WIDTH,
                            _145_SIGMOID_37_OUTPUT_0_MEM_CONT_OFFSET,
                            _145_SIGMOID_37_OUTPUT_0_MEM_CONT_SIZE,
                            _145_SIGMOID_37_OUTPUT_0_MEM_WRAP_OFFSET,
                            _145_SIGMOID_37_OUTPUT_0_MEM_WRAP_SIZE,
                            _145_SIGMOID_37_OUTPUT_0_DEV_FMT>
                            (_145_Sigmoid_37_output_0,
                            _145_SIGMOID_37_OUTPUT_0_STREAM);
    fclose(_145_SIGMOID_37_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _146_Mul_37_output_0 = (float*) (mem_0 + _146_MUL_37_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_146_MUL_37_NB_MAT,
                                 _146_MUL_37_ELEM_OP,
                                 _146_MUL_37_INPUT1_CONT_SIZE,
                                 _146_MUL_37_INPUT2_CONT_SIZE,
                                 _146_MUL_37_OUTPUT_CONT_SIZE,
                                 _146_MUL_37_OFFSET_IN1,
                                 _146_MUL_37_OFFSET_IN2,
                                 _146_MUL_37_ACTIVATION,
                                 _144_CONV2D_19_OUTPUT_0_MEM_CONT_OFFSET,
                                 _144_CONV2D_19_OUTPUT_0_MEM_CONT_SIZE,
                                 _144_CONV2D_19_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _144_CONV2D_19_OUTPUT_0_MEM_WRAP_SIZE,
                                 _144_CONV2D_19_OUTPUT_0_MEM_STRIDE,
                                 _145_SIGMOID_37_OUTPUT_0_MEM_CONT_OFFSET,
                                 _145_SIGMOID_37_OUTPUT_0_MEM_CONT_SIZE,
                                 _145_SIGMOID_37_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _145_SIGMOID_37_OUTPUT_0_MEM_WRAP_SIZE,
                                 _145_SIGMOID_37_OUTPUT_0_MEM_STRIDE,
                                 _146_MUL_37_OUTPUT_0_MEM_CONT_OFFSET,
                                 _146_MUL_37_OUTPUT_0_MEM_CONT_SIZE,
                                 _146_MUL_37_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _146_MUL_37_OUTPUT_0_MEM_WRAP_SIZE,
                                 _146_MUL_37_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_146_Mul_37_output_0, _146_MUL_37_RESCALING, _144_Conv2D_19_output_0, _145_Sigmoid_37_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _146_Mul_37\n");
    FILE* _146_MUL_37_OUTPUT_0_STREAM = fopen("data/export_outputs/_146_Mul_37_output_0.txt", "w");
    export_cpp::saveOutputs<_146_MUL_37_OUTPUT_0_NB_OUTPUTS,
                            _146_MUL_37_OUTPUT_0_OUT_HEIGHT,
                            _146_MUL_37_OUTPUT_0_OUT_WIDTH,
                            _146_MUL_37_OUTPUT_0_MEM_CONT_OFFSET,
                            _146_MUL_37_OUTPUT_0_MEM_CONT_SIZE,
                            _146_MUL_37_OUTPUT_0_MEM_WRAP_OFFSET,
                            _146_MUL_37_OUTPUT_0_MEM_WRAP_SIZE,
                            _146_MUL_37_OUTPUT_0_DEV_FMT>
                            (_146_Mul_37_output_0,
                            _146_MUL_37_OUTPUT_0_STREAM);
    fclose(_146_MUL_37_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _147_PadConvDw_2_output_0 = (float*) (mem_0 + _147_PADCONVDW_2_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_146_MUL_37_OUTPUT_0_IN_BATCH,
                           _147_PADCONVDW_2_NB_CHANNELS,
                           _146_MUL_37_OUTPUT_0_IN_HEIGHT,
                           _146_MUL_37_OUTPUT_0_IN_WIDTH,
                           _147_PADCONVDW_2_NB_OUTPUTS,
                           _147_PADCONVDW_2_OUTPUT_0_OUT_HEIGHT,
                           _147_PADCONVDW_2_OUTPUT_0_OUT_WIDTH,
                           _147_PADCONVDW_2_PADDING_Y,
                           _147_PADCONVDW_2_PADDING_X,
                           _147_PADCONVDW_2_STRIDE_Y,
                           _147_PADCONVDW_2_STRIDE_X,
                           _147_PADCONVDW_2_DILATION_Y,
                           _147_PADCONVDW_2_DILATION_X,
                           _147_PADCONVDW_2_KERNEL_HEIGHT,
                           _147_PADCONVDW_2_KERNEL_WIDTH,
                           _147_PADCONVDW_2_ACTIVATION,
                           _146_MUL_37_OUTPUT_0_MEM_CONT_OFFSET,
                           _146_MUL_37_OUTPUT_0_MEM_CONT_SIZE,
                           _146_MUL_37_OUTPUT_0_MEM_WRAP_OFFSET,
                           _146_MUL_37_OUTPUT_0_MEM_WRAP_SIZE,
                           _146_MUL_37_OUTPUT_0_MEM_STRIDE,
                           _147_PADCONVDW_2_OUTPUT_0_MEM_CONT_OFFSET,
                           _147_PADCONVDW_2_OUTPUT_0_MEM_CONT_SIZE,
                           _147_PADCONVDW_2_OUTPUT_0_MEM_WRAP_OFFSET,
                           _147_PADCONVDW_2_OUTPUT_0_MEM_WRAP_SIZE,
                           _147_PADCONVDW_2_OUTPUT_0_MEM_STRIDE>
                           (_146_Mul_37_output_0, _147_PadConvDw_2_output_0, _147_PadConvDw_2_weights_output_0, _147_PadConvDw_2_biases_output_0, _147_PADCONVDW_2_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _147_PadConvDw_2\n");
    FILE* _147_PADCONVDW_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_147_PadConvDw_2_output_0.txt", "w");
    export_cpp::saveOutputs<_147_PADCONVDW_2_OUTPUT_0_NB_OUTPUTS,
                            _147_PADCONVDW_2_OUTPUT_0_OUT_HEIGHT,
                            _147_PADCONVDW_2_OUTPUT_0_OUT_WIDTH,
                            _147_PADCONVDW_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _147_PADCONVDW_2_OUTPUT_0_MEM_CONT_SIZE,
                            _147_PADCONVDW_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _147_PADCONVDW_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _147_PADCONVDW_2_OUTPUT_0_DEV_FMT>
                            (_147_PadConvDw_2_output_0,
                            _147_PADCONVDW_2_OUTPUT_0_STREAM);
    fclose(_147_PADCONVDW_2_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _148_PadConv_19_output_0 = (float*) (mem_0 + _148_PADCONV_19_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_146_MUL_37_OUTPUT_0_IN_BATCH,
                           _148_PADCONV_19_NB_CHANNELS,
                           _146_MUL_37_OUTPUT_0_IN_HEIGHT,
                           _146_MUL_37_OUTPUT_0_IN_WIDTH,
                           _148_PADCONV_19_NB_OUTPUTS,
                           _148_PADCONV_19_OUTPUT_0_OUT_HEIGHT,
                           _148_PADCONV_19_OUTPUT_0_OUT_WIDTH,
                           _148_PADCONV_19_PADDING_Y,
                           _148_PADCONV_19_PADDING_X,
                           _148_PADCONV_19_STRIDE_Y,
                           _148_PADCONV_19_STRIDE_X,
                           _148_PADCONV_19_DILATION_Y,
                           _148_PADCONV_19_DILATION_X,
                           _148_PADCONV_19_KERNEL_HEIGHT,
                           _148_PADCONV_19_KERNEL_WIDTH,
                           _148_PADCONV_19_ACTIVATION,
                           _146_MUL_37_OUTPUT_0_MEM_CONT_OFFSET,
                           _146_MUL_37_OUTPUT_0_MEM_CONT_SIZE,
                           _146_MUL_37_OUTPUT_0_MEM_WRAP_OFFSET,
                           _146_MUL_37_OUTPUT_0_MEM_WRAP_SIZE,
                           _146_MUL_37_OUTPUT_0_MEM_STRIDE,
                           _148_PADCONV_19_OUTPUT_0_MEM_CONT_OFFSET,
                           _148_PADCONV_19_OUTPUT_0_MEM_CONT_SIZE,
                           _148_PADCONV_19_OUTPUT_0_MEM_WRAP_OFFSET,
                           _148_PADCONV_19_OUTPUT_0_MEM_WRAP_SIZE,
                           _148_PADCONV_19_OUTPUT_0_MEM_STRIDE>
                           (_146_Mul_37_output_0, _148_PadConv_19_output_0, _148_PadConv_19_weights_output_0, _148_PadConv_19_biases_output_0, _148_PADCONV_19_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _148_PadConv_19\n");
    FILE* _148_PADCONV_19_OUTPUT_0_STREAM = fopen("data/export_outputs/_148_PadConv_19_output_0.txt", "w");
    export_cpp::saveOutputs<_148_PADCONV_19_OUTPUT_0_NB_OUTPUTS,
                            _148_PADCONV_19_OUTPUT_0_OUT_HEIGHT,
                            _148_PADCONV_19_OUTPUT_0_OUT_WIDTH,
                            _148_PADCONV_19_OUTPUT_0_MEM_CONT_OFFSET,
                            _148_PADCONV_19_OUTPUT_0_MEM_CONT_SIZE,
                            _148_PADCONV_19_OUTPUT_0_MEM_WRAP_OFFSET,
                            _148_PADCONV_19_OUTPUT_0_MEM_WRAP_SIZE,
                            _148_PADCONV_19_OUTPUT_0_DEV_FMT>
                            (_148_PadConv_19_output_0,
                            _148_PADCONV_19_OUTPUT_0_STREAM);
    fclose(_148_PADCONV_19_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _149_PadConv_20_output_0 = (float*) (mem_0 + _149_PADCONV_20_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_146_MUL_37_OUTPUT_0_IN_BATCH,
                           _149_PADCONV_20_NB_CHANNELS,
                           _146_MUL_37_OUTPUT_0_IN_HEIGHT,
                           _146_MUL_37_OUTPUT_0_IN_WIDTH,
                           _149_PADCONV_20_NB_OUTPUTS,
                           _149_PADCONV_20_OUTPUT_0_OUT_HEIGHT,
                           _149_PADCONV_20_OUTPUT_0_OUT_WIDTH,
                           _149_PADCONV_20_PADDING_Y,
                           _149_PADCONV_20_PADDING_X,
                           _149_PADCONV_20_STRIDE_Y,
                           _149_PADCONV_20_STRIDE_X,
                           _149_PADCONV_20_DILATION_Y,
                           _149_PADCONV_20_DILATION_X,
                           _149_PADCONV_20_KERNEL_HEIGHT,
                           _149_PADCONV_20_KERNEL_WIDTH,
                           _149_PADCONV_20_ACTIVATION,
                           _146_MUL_37_OUTPUT_0_MEM_CONT_OFFSET,
                           _146_MUL_37_OUTPUT_0_MEM_CONT_SIZE,
                           _146_MUL_37_OUTPUT_0_MEM_WRAP_OFFSET,
                           _146_MUL_37_OUTPUT_0_MEM_WRAP_SIZE,
                           _146_MUL_37_OUTPUT_0_MEM_STRIDE,
                           _149_PADCONV_20_OUTPUT_0_MEM_CONT_OFFSET,
                           _149_PADCONV_20_OUTPUT_0_MEM_CONT_SIZE,
                           _149_PADCONV_20_OUTPUT_0_MEM_WRAP_OFFSET,
                           _149_PADCONV_20_OUTPUT_0_MEM_WRAP_SIZE,
                           _149_PADCONV_20_OUTPUT_0_MEM_STRIDE>
                           (_146_Mul_37_output_0, _149_PadConv_20_output_0, _149_PadConv_20_weights_output_0, _149_PadConv_20_biases_output_0, _149_PADCONV_20_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _149_PadConv_20\n");
    FILE* _149_PADCONV_20_OUTPUT_0_STREAM = fopen("data/export_outputs/_149_PadConv_20_output_0.txt", "w");
    export_cpp::saveOutputs<_149_PADCONV_20_OUTPUT_0_NB_OUTPUTS,
                            _149_PADCONV_20_OUTPUT_0_OUT_HEIGHT,
                            _149_PADCONV_20_OUTPUT_0_OUT_WIDTH,
                            _149_PADCONV_20_OUTPUT_0_MEM_CONT_OFFSET,
                            _149_PADCONV_20_OUTPUT_0_MEM_CONT_SIZE,
                            _149_PADCONV_20_OUTPUT_0_MEM_WRAP_OFFSET,
                            _149_PADCONV_20_OUTPUT_0_MEM_WRAP_SIZE,
                            _149_PADCONV_20_OUTPUT_0_DEV_FMT>
                            (_149_PadConv_20_output_0,
                            _149_PADCONV_20_OUTPUT_0_STREAM);
    fclose(_149_PADCONV_20_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _150_Sigmoid_38_output_0 = (float*) (mem_0 + _150_SIGMOID_38_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_150_SIGMOID_38_NB_ELTS>
                               (_149_PadConv_20_output_0, _150_Sigmoid_38_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _150_Sigmoid_38\n");
    FILE* _150_SIGMOID_38_OUTPUT_0_STREAM = fopen("data/export_outputs/_150_Sigmoid_38_output_0.txt", "w");
    export_cpp::saveOutputs<_150_SIGMOID_38_OUTPUT_0_NB_OUTPUTS,
                            _150_SIGMOID_38_OUTPUT_0_OUT_HEIGHT,
                            _150_SIGMOID_38_OUTPUT_0_OUT_WIDTH,
                            _150_SIGMOID_38_OUTPUT_0_MEM_CONT_OFFSET,
                            _150_SIGMOID_38_OUTPUT_0_MEM_CONT_SIZE,
                            _150_SIGMOID_38_OUTPUT_0_MEM_WRAP_OFFSET,
                            _150_SIGMOID_38_OUTPUT_0_MEM_WRAP_SIZE,
                            _150_SIGMOID_38_OUTPUT_0_DEV_FMT>
                            (_150_Sigmoid_38_output_0,
                            _150_SIGMOID_38_OUTPUT_0_STREAM);
    fclose(_150_SIGMOID_38_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _151_Sigmoid_39_output_0 = (float*) (mem_0 + _151_SIGMOID_39_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_151_SIGMOID_39_NB_ELTS>
                               (_148_PadConv_19_output_0, _151_Sigmoid_39_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _151_Sigmoid_39\n");
    FILE* _151_SIGMOID_39_OUTPUT_0_STREAM = fopen("data/export_outputs/_151_Sigmoid_39_output_0.txt", "w");
    export_cpp::saveOutputs<_151_SIGMOID_39_OUTPUT_0_NB_OUTPUTS,
                            _151_SIGMOID_39_OUTPUT_0_OUT_HEIGHT,
                            _151_SIGMOID_39_OUTPUT_0_OUT_WIDTH,
                            _151_SIGMOID_39_OUTPUT_0_MEM_CONT_OFFSET,
                            _151_SIGMOID_39_OUTPUT_0_MEM_CONT_SIZE,
                            _151_SIGMOID_39_OUTPUT_0_MEM_WRAP_OFFSET,
                            _151_SIGMOID_39_OUTPUT_0_MEM_WRAP_SIZE,
                            _151_SIGMOID_39_OUTPUT_0_DEV_FMT>
                            (_151_Sigmoid_39_output_0,
                            _151_SIGMOID_39_OUTPUT_0_STREAM);
    fclose(_151_SIGMOID_39_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _152_Sigmoid_40_output_0 = (float*) (mem_0 + _152_SIGMOID_40_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_152_SIGMOID_40_NB_ELTS>
                               (_147_PadConvDw_2_output_0, _152_Sigmoid_40_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _152_Sigmoid_40\n");
    FILE* _152_SIGMOID_40_OUTPUT_0_STREAM = fopen("data/export_outputs/_152_Sigmoid_40_output_0.txt", "w");
    export_cpp::saveOutputs<_152_SIGMOID_40_OUTPUT_0_NB_OUTPUTS,
                            _152_SIGMOID_40_OUTPUT_0_OUT_HEIGHT,
                            _152_SIGMOID_40_OUTPUT_0_OUT_WIDTH,
                            _152_SIGMOID_40_OUTPUT_0_MEM_CONT_OFFSET,
                            _152_SIGMOID_40_OUTPUT_0_MEM_CONT_SIZE,
                            _152_SIGMOID_40_OUTPUT_0_MEM_WRAP_OFFSET,
                            _152_SIGMOID_40_OUTPUT_0_MEM_WRAP_SIZE,
                            _152_SIGMOID_40_OUTPUT_0_DEV_FMT>
                            (_152_Sigmoid_40_output_0,
                            _152_SIGMOID_40_OUTPUT_0_STREAM);
    fclose(_152_SIGMOID_40_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _153_Mul_38_output_0 = (float*) (mem_0 + _153_MUL_38_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_153_MUL_38_NB_MAT,
                                 _153_MUL_38_ELEM_OP,
                                 _153_MUL_38_INPUT1_CONT_SIZE,
                                 _153_MUL_38_INPUT2_CONT_SIZE,
                                 _153_MUL_38_OUTPUT_CONT_SIZE,
                                 _153_MUL_38_OFFSET_IN1,
                                 _153_MUL_38_OFFSET_IN2,
                                 _153_MUL_38_ACTIVATION,
                                 _149_PADCONV_20_OUTPUT_0_MEM_CONT_OFFSET,
                                 _149_PADCONV_20_OUTPUT_0_MEM_CONT_SIZE,
                                 _149_PADCONV_20_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _149_PADCONV_20_OUTPUT_0_MEM_WRAP_SIZE,
                                 _149_PADCONV_20_OUTPUT_0_MEM_STRIDE,
                                 _150_SIGMOID_38_OUTPUT_0_MEM_CONT_OFFSET,
                                 _150_SIGMOID_38_OUTPUT_0_MEM_CONT_SIZE,
                                 _150_SIGMOID_38_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _150_SIGMOID_38_OUTPUT_0_MEM_WRAP_SIZE,
                                 _150_SIGMOID_38_OUTPUT_0_MEM_STRIDE,
                                 _153_MUL_38_OUTPUT_0_MEM_CONT_OFFSET,
                                 _153_MUL_38_OUTPUT_0_MEM_CONT_SIZE,
                                 _153_MUL_38_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _153_MUL_38_OUTPUT_0_MEM_WRAP_SIZE,
                                 _153_MUL_38_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_153_Mul_38_output_0, _153_MUL_38_RESCALING, _149_PadConv_20_output_0, _150_Sigmoid_38_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _153_Mul_38\n");
    FILE* _153_MUL_38_OUTPUT_0_STREAM = fopen("data/export_outputs/_153_Mul_38_output_0.txt", "w");
    export_cpp::saveOutputs<_153_MUL_38_OUTPUT_0_NB_OUTPUTS,
                            _153_MUL_38_OUTPUT_0_OUT_HEIGHT,
                            _153_MUL_38_OUTPUT_0_OUT_WIDTH,
                            _153_MUL_38_OUTPUT_0_MEM_CONT_OFFSET,
                            _153_MUL_38_OUTPUT_0_MEM_CONT_SIZE,
                            _153_MUL_38_OUTPUT_0_MEM_WRAP_OFFSET,
                            _153_MUL_38_OUTPUT_0_MEM_WRAP_SIZE,
                            _153_MUL_38_OUTPUT_0_DEV_FMT>
                            (_153_Mul_38_output_0,
                            _153_MUL_38_OUTPUT_0_STREAM);
    fclose(_153_MUL_38_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _154_Mul_39_output_0 = (float*) (mem_0 + _154_MUL_39_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_154_MUL_39_NB_MAT,
                                 _154_MUL_39_ELEM_OP,
                                 _154_MUL_39_INPUT1_CONT_SIZE,
                                 _154_MUL_39_INPUT2_CONT_SIZE,
                                 _154_MUL_39_OUTPUT_CONT_SIZE,
                                 _154_MUL_39_OFFSET_IN1,
                                 _154_MUL_39_OFFSET_IN2,
                                 _154_MUL_39_ACTIVATION,
                                 _148_PADCONV_19_OUTPUT_0_MEM_CONT_OFFSET,
                                 _148_PADCONV_19_OUTPUT_0_MEM_CONT_SIZE,
                                 _148_PADCONV_19_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _148_PADCONV_19_OUTPUT_0_MEM_WRAP_SIZE,
                                 _148_PADCONV_19_OUTPUT_0_MEM_STRIDE,
                                 _151_SIGMOID_39_OUTPUT_0_MEM_CONT_OFFSET,
                                 _151_SIGMOID_39_OUTPUT_0_MEM_CONT_SIZE,
                                 _151_SIGMOID_39_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _151_SIGMOID_39_OUTPUT_0_MEM_WRAP_SIZE,
                                 _151_SIGMOID_39_OUTPUT_0_MEM_STRIDE,
                                 _154_MUL_39_OUTPUT_0_MEM_CONT_OFFSET,
                                 _154_MUL_39_OUTPUT_0_MEM_CONT_SIZE,
                                 _154_MUL_39_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _154_MUL_39_OUTPUT_0_MEM_WRAP_SIZE,
                                 _154_MUL_39_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_154_Mul_39_output_0, _154_MUL_39_RESCALING, _148_PadConv_19_output_0, _151_Sigmoid_39_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _154_Mul_39\n");
    FILE* _154_MUL_39_OUTPUT_0_STREAM = fopen("data/export_outputs/_154_Mul_39_output_0.txt", "w");
    export_cpp::saveOutputs<_154_MUL_39_OUTPUT_0_NB_OUTPUTS,
                            _154_MUL_39_OUTPUT_0_OUT_HEIGHT,
                            _154_MUL_39_OUTPUT_0_OUT_WIDTH,
                            _154_MUL_39_OUTPUT_0_MEM_CONT_OFFSET,
                            _154_MUL_39_OUTPUT_0_MEM_CONT_SIZE,
                            _154_MUL_39_OUTPUT_0_MEM_WRAP_OFFSET,
                            _154_MUL_39_OUTPUT_0_MEM_WRAP_SIZE,
                            _154_MUL_39_OUTPUT_0_DEV_FMT>
                            (_154_Mul_39_output_0,
                            _154_MUL_39_OUTPUT_0_STREAM);
    fclose(_154_MUL_39_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _155_Mul_40_output_0 = (float*) (mem_0 + _155_MUL_40_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_155_MUL_40_NB_MAT,
                                 _155_MUL_40_ELEM_OP,
                                 _155_MUL_40_INPUT1_CONT_SIZE,
                                 _155_MUL_40_INPUT2_CONT_SIZE,
                                 _155_MUL_40_OUTPUT_CONT_SIZE,
                                 _155_MUL_40_OFFSET_IN1,
                                 _155_MUL_40_OFFSET_IN2,
                                 _155_MUL_40_ACTIVATION,
                                 _147_PADCONVDW_2_OUTPUT_0_MEM_CONT_OFFSET,
                                 _147_PADCONVDW_2_OUTPUT_0_MEM_CONT_SIZE,
                                 _147_PADCONVDW_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _147_PADCONVDW_2_OUTPUT_0_MEM_WRAP_SIZE,
                                 _147_PADCONVDW_2_OUTPUT_0_MEM_STRIDE,
                                 _152_SIGMOID_40_OUTPUT_0_MEM_CONT_OFFSET,
                                 _152_SIGMOID_40_OUTPUT_0_MEM_CONT_SIZE,
                                 _152_SIGMOID_40_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _152_SIGMOID_40_OUTPUT_0_MEM_WRAP_SIZE,
                                 _152_SIGMOID_40_OUTPUT_0_MEM_STRIDE,
                                 _155_MUL_40_OUTPUT_0_MEM_CONT_OFFSET,
                                 _155_MUL_40_OUTPUT_0_MEM_CONT_SIZE,
                                 _155_MUL_40_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _155_MUL_40_OUTPUT_0_MEM_WRAP_SIZE,
                                 _155_MUL_40_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_155_Mul_40_output_0, _155_MUL_40_RESCALING, _147_PadConvDw_2_output_0, _152_Sigmoid_40_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _155_Mul_40\n");
    FILE* _155_MUL_40_OUTPUT_0_STREAM = fopen("data/export_outputs/_155_Mul_40_output_0.txt", "w");
    export_cpp::saveOutputs<_155_MUL_40_OUTPUT_0_NB_OUTPUTS,
                            _155_MUL_40_OUTPUT_0_OUT_HEIGHT,
                            _155_MUL_40_OUTPUT_0_OUT_WIDTH,
                            _155_MUL_40_OUTPUT_0_MEM_CONT_OFFSET,
                            _155_MUL_40_OUTPUT_0_MEM_CONT_SIZE,
                            _155_MUL_40_OUTPUT_0_MEM_WRAP_OFFSET,
                            _155_MUL_40_OUTPUT_0_MEM_WRAP_SIZE,
                            _155_MUL_40_OUTPUT_0_DEV_FMT>
                            (_155_Mul_40_output_0,
                            _155_MUL_40_OUTPUT_0_STREAM);
    fclose(_155_MUL_40_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _156_Concat_10_output_0 = (float*) (mem_0 + _156_CONCAT_10_OUTPUT_0_MEM_OFFSET);

    const float* _156_Concat_10_inputs[] = { _153_Mul_38_output_0, _130_Mul_33_output_0 };
    export_cpp::concat_forward<_156_CONCAT_10_AXIS_SIZE_POST,
                               _156_CONCAT_10_AXIS_SIZE_PRE,
                               _156_CONCAT_10_AXIS_SIZE,
                               _156_CONCAT_10_TOTAL_AXIS_SIZE,
                               2> 
                              (_156_Concat_10_inputs,
                               _156_Concat_10_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _156_Concat_10\n");
    FILE* _156_CONCAT_10_OUTPUT_0_STREAM = fopen("data/export_outputs/_156_Concat_10_output_0.txt", "w");
    export_cpp::saveOutputs<_156_CONCAT_10_OUTPUT_0_NB_OUTPUTS,
                            _156_CONCAT_10_OUTPUT_0_OUT_HEIGHT,
                            _156_CONCAT_10_OUTPUT_0_OUT_WIDTH,
                            _156_CONCAT_10_OUTPUT_0_MEM_CONT_OFFSET,
                            _156_CONCAT_10_OUTPUT_0_MEM_CONT_SIZE,
                            _156_CONCAT_10_OUTPUT_0_MEM_WRAP_OFFSET,
                            _156_CONCAT_10_OUTPUT_0_MEM_WRAP_SIZE,
                            _156_CONCAT_10_OUTPUT_0_DEV_FMT>
                            (_156_Concat_10_output_0,
                            _156_CONCAT_10_OUTPUT_0_STREAM);
    fclose(_156_CONCAT_10_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _157_PadConv_21_output_0 = (float*) (mem_0 + _157_PADCONV_21_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_154_MUL_39_OUTPUT_0_IN_BATCH,
                           _157_PADCONV_21_NB_CHANNELS,
                           _154_MUL_39_OUTPUT_0_IN_HEIGHT,
                           _154_MUL_39_OUTPUT_0_IN_WIDTH,
                           _157_PADCONV_21_NB_OUTPUTS,
                           _157_PADCONV_21_OUTPUT_0_OUT_HEIGHT,
                           _157_PADCONV_21_OUTPUT_0_OUT_WIDTH,
                           _157_PADCONV_21_PADDING_Y,
                           _157_PADCONV_21_PADDING_X,
                           _157_PADCONV_21_STRIDE_Y,
                           _157_PADCONV_21_STRIDE_X,
                           _157_PADCONV_21_DILATION_Y,
                           _157_PADCONV_21_DILATION_X,
                           _157_PADCONV_21_KERNEL_HEIGHT,
                           _157_PADCONV_21_KERNEL_WIDTH,
                           _157_PADCONV_21_ACTIVATION,
                           _154_MUL_39_OUTPUT_0_MEM_CONT_OFFSET,
                           _154_MUL_39_OUTPUT_0_MEM_CONT_SIZE,
                           _154_MUL_39_OUTPUT_0_MEM_WRAP_OFFSET,
                           _154_MUL_39_OUTPUT_0_MEM_WRAP_SIZE,
                           _154_MUL_39_OUTPUT_0_MEM_STRIDE,
                           _157_PADCONV_21_OUTPUT_0_MEM_CONT_OFFSET,
                           _157_PADCONV_21_OUTPUT_0_MEM_CONT_SIZE,
                           _157_PADCONV_21_OUTPUT_0_MEM_WRAP_OFFSET,
                           _157_PADCONV_21_OUTPUT_0_MEM_WRAP_SIZE,
                           _157_PADCONV_21_OUTPUT_0_MEM_STRIDE>
                           (_154_Mul_39_output_0, _157_PadConv_21_output_0, _157_PadConv_21_weights_output_0, _157_PadConv_21_biases_output_0, _157_PADCONV_21_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _157_PadConv_21\n");
    FILE* _157_PADCONV_21_OUTPUT_0_STREAM = fopen("data/export_outputs/_157_PadConv_21_output_0.txt", "w");
    export_cpp::saveOutputs<_157_PADCONV_21_OUTPUT_0_NB_OUTPUTS,
                            _157_PADCONV_21_OUTPUT_0_OUT_HEIGHT,
                            _157_PADCONV_21_OUTPUT_0_OUT_WIDTH,
                            _157_PADCONV_21_OUTPUT_0_MEM_CONT_OFFSET,
                            _157_PADCONV_21_OUTPUT_0_MEM_CONT_SIZE,
                            _157_PADCONV_21_OUTPUT_0_MEM_WRAP_OFFSET,
                            _157_PADCONV_21_OUTPUT_0_MEM_WRAP_SIZE,
                            _157_PADCONV_21_OUTPUT_0_DEV_FMT>
                            (_157_PadConv_21_output_0,
                            _157_PADCONV_21_OUTPUT_0_STREAM);
    fclose(_157_PADCONV_21_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _158_Conv2D_20_output_0 = (float*) (mem_0 + _158_CONV2D_20_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_155_MUL_40_OUTPUT_0_IN_BATCH,
                           _158_CONV2D_20_NB_CHANNELS,
                           _155_MUL_40_OUTPUT_0_IN_HEIGHT,
                           _155_MUL_40_OUTPUT_0_IN_WIDTH,
                           _158_CONV2D_20_NB_OUTPUTS,
                           _158_CONV2D_20_OUTPUT_0_OUT_HEIGHT,
                           _158_CONV2D_20_OUTPUT_0_OUT_WIDTH,
                           _158_CONV2D_20_PADDING_Y,
                           _158_CONV2D_20_PADDING_X,
                           _158_CONV2D_20_STRIDE_Y,
                           _158_CONV2D_20_STRIDE_X,
                           _158_CONV2D_20_DILATION_Y,
                           _158_CONV2D_20_DILATION_X,
                           _158_CONV2D_20_KERNEL_HEIGHT,
                           _158_CONV2D_20_KERNEL_WIDTH,
                           _158_CONV2D_20_ACTIVATION,
                           _155_MUL_40_OUTPUT_0_MEM_CONT_OFFSET,
                           _155_MUL_40_OUTPUT_0_MEM_CONT_SIZE,
                           _155_MUL_40_OUTPUT_0_MEM_WRAP_OFFSET,
                           _155_MUL_40_OUTPUT_0_MEM_WRAP_SIZE,
                           _155_MUL_40_OUTPUT_0_MEM_STRIDE,
                           _158_CONV2D_20_OUTPUT_0_MEM_CONT_OFFSET,
                           _158_CONV2D_20_OUTPUT_0_MEM_CONT_SIZE,
                           _158_CONV2D_20_OUTPUT_0_MEM_WRAP_OFFSET,
                           _158_CONV2D_20_OUTPUT_0_MEM_WRAP_SIZE,
                           _158_CONV2D_20_OUTPUT_0_MEM_STRIDE>
                           (_155_Mul_40_output_0, _158_Conv2D_20_output_0, _158_Conv2D_20_weights_output_0, _158_Conv2D_20_biases_output_0, _158_CONV2D_20_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _158_Conv2D_20\n");
    FILE* _158_CONV2D_20_OUTPUT_0_STREAM = fopen("data/export_outputs/_158_Conv2D_20_output_0.txt", "w");
    export_cpp::saveOutputs<_158_CONV2D_20_OUTPUT_0_NB_OUTPUTS,
                            _158_CONV2D_20_OUTPUT_0_OUT_HEIGHT,
                            _158_CONV2D_20_OUTPUT_0_OUT_WIDTH,
                            _158_CONV2D_20_OUTPUT_0_MEM_CONT_OFFSET,
                            _158_CONV2D_20_OUTPUT_0_MEM_CONT_SIZE,
                            _158_CONV2D_20_OUTPUT_0_MEM_WRAP_OFFSET,
                            _158_CONV2D_20_OUTPUT_0_MEM_WRAP_SIZE,
                            _158_CONV2D_20_OUTPUT_0_DEV_FMT>
                            (_158_Conv2D_20_output_0,
                            _158_CONV2D_20_OUTPUT_0_STREAM);
    fclose(_158_CONV2D_20_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _159_Sigmoid_41_output_0 = (float*) (mem_0 + _159_SIGMOID_41_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_159_SIGMOID_41_NB_ELTS>
                               (_157_PadConv_21_output_0, _159_Sigmoid_41_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _159_Sigmoid_41\n");
    FILE* _159_SIGMOID_41_OUTPUT_0_STREAM = fopen("data/export_outputs/_159_Sigmoid_41_output_0.txt", "w");
    export_cpp::saveOutputs<_159_SIGMOID_41_OUTPUT_0_NB_OUTPUTS,
                            _159_SIGMOID_41_OUTPUT_0_OUT_HEIGHT,
                            _159_SIGMOID_41_OUTPUT_0_OUT_WIDTH,
                            _159_SIGMOID_41_OUTPUT_0_MEM_CONT_OFFSET,
                            _159_SIGMOID_41_OUTPUT_0_MEM_CONT_SIZE,
                            _159_SIGMOID_41_OUTPUT_0_MEM_WRAP_OFFSET,
                            _159_SIGMOID_41_OUTPUT_0_MEM_WRAP_SIZE,
                            _159_SIGMOID_41_OUTPUT_0_DEV_FMT>
                            (_159_Sigmoid_41_output_0,
                            _159_SIGMOID_41_OUTPUT_0_STREAM);
    fclose(_159_SIGMOID_41_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _160_Sigmoid_42_output_0 = (float*) (mem_0 + _160_SIGMOID_42_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_160_SIGMOID_42_NB_ELTS>
                               (_158_Conv2D_20_output_0, _160_Sigmoid_42_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _160_Sigmoid_42\n");
    FILE* _160_SIGMOID_42_OUTPUT_0_STREAM = fopen("data/export_outputs/_160_Sigmoid_42_output_0.txt", "w");
    export_cpp::saveOutputs<_160_SIGMOID_42_OUTPUT_0_NB_OUTPUTS,
                            _160_SIGMOID_42_OUTPUT_0_OUT_HEIGHT,
                            _160_SIGMOID_42_OUTPUT_0_OUT_WIDTH,
                            _160_SIGMOID_42_OUTPUT_0_MEM_CONT_OFFSET,
                            _160_SIGMOID_42_OUTPUT_0_MEM_CONT_SIZE,
                            _160_SIGMOID_42_OUTPUT_0_MEM_WRAP_OFFSET,
                            _160_SIGMOID_42_OUTPUT_0_MEM_WRAP_SIZE,
                            _160_SIGMOID_42_OUTPUT_0_DEV_FMT>
                            (_160_Sigmoid_42_output_0,
                            _160_SIGMOID_42_OUTPUT_0_STREAM);
    fclose(_160_SIGMOID_42_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _161_Conv2D_21_output_0 = (float*) (mem_0 + _161_CONV2D_21_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_156_CONCAT_10_OUTPUT_0_IN_BATCH,
                           _161_CONV2D_21_NB_CHANNELS,
                           _156_CONCAT_10_OUTPUT_0_IN_HEIGHT,
                           _156_CONCAT_10_OUTPUT_0_IN_WIDTH,
                           _161_CONV2D_21_NB_OUTPUTS,
                           _161_CONV2D_21_OUTPUT_0_OUT_HEIGHT,
                           _161_CONV2D_21_OUTPUT_0_OUT_WIDTH,
                           _161_CONV2D_21_PADDING_Y,
                           _161_CONV2D_21_PADDING_X,
                           _161_CONV2D_21_STRIDE_Y,
                           _161_CONV2D_21_STRIDE_X,
                           _161_CONV2D_21_DILATION_Y,
                           _161_CONV2D_21_DILATION_X,
                           _161_CONV2D_21_KERNEL_HEIGHT,
                           _161_CONV2D_21_KERNEL_WIDTH,
                           _161_CONV2D_21_ACTIVATION,
                           _156_CONCAT_10_OUTPUT_0_MEM_CONT_OFFSET,
                           _156_CONCAT_10_OUTPUT_0_MEM_CONT_SIZE,
                           _156_CONCAT_10_OUTPUT_0_MEM_WRAP_OFFSET,
                           _156_CONCAT_10_OUTPUT_0_MEM_WRAP_SIZE,
                           _156_CONCAT_10_OUTPUT_0_MEM_STRIDE,
                           _161_CONV2D_21_OUTPUT_0_MEM_CONT_OFFSET,
                           _161_CONV2D_21_OUTPUT_0_MEM_CONT_SIZE,
                           _161_CONV2D_21_OUTPUT_0_MEM_WRAP_OFFSET,
                           _161_CONV2D_21_OUTPUT_0_MEM_WRAP_SIZE,
                           _161_CONV2D_21_OUTPUT_0_MEM_STRIDE>
                           (_156_Concat_10_output_0, _161_Conv2D_21_output_0, _161_Conv2D_21_weights_output_0, _161_Conv2D_21_biases_output_0, _161_CONV2D_21_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _161_Conv2D_21\n");
    FILE* _161_CONV2D_21_OUTPUT_0_STREAM = fopen("data/export_outputs/_161_Conv2D_21_output_0.txt", "w");
    export_cpp::saveOutputs<_161_CONV2D_21_OUTPUT_0_NB_OUTPUTS,
                            _161_CONV2D_21_OUTPUT_0_OUT_HEIGHT,
                            _161_CONV2D_21_OUTPUT_0_OUT_WIDTH,
                            _161_CONV2D_21_OUTPUT_0_MEM_CONT_OFFSET,
                            _161_CONV2D_21_OUTPUT_0_MEM_CONT_SIZE,
                            _161_CONV2D_21_OUTPUT_0_MEM_WRAP_OFFSET,
                            _161_CONV2D_21_OUTPUT_0_MEM_WRAP_SIZE,
                            _161_CONV2D_21_OUTPUT_0_DEV_FMT>
                            (_161_Conv2D_21_output_0,
                            _161_CONV2D_21_OUTPUT_0_STREAM);
    fclose(_161_CONV2D_21_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _162_Sigmoid_43_output_0 = (float*) (mem_0 + _162_SIGMOID_43_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_162_SIGMOID_43_NB_ELTS>
                               (_161_Conv2D_21_output_0, _162_Sigmoid_43_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _162_Sigmoid_43\n");
    FILE* _162_SIGMOID_43_OUTPUT_0_STREAM = fopen("data/export_outputs/_162_Sigmoid_43_output_0.txt", "w");
    export_cpp::saveOutputs<_162_SIGMOID_43_OUTPUT_0_NB_OUTPUTS,
                            _162_SIGMOID_43_OUTPUT_0_OUT_HEIGHT,
                            _162_SIGMOID_43_OUTPUT_0_OUT_WIDTH,
                            _162_SIGMOID_43_OUTPUT_0_MEM_CONT_OFFSET,
                            _162_SIGMOID_43_OUTPUT_0_MEM_CONT_SIZE,
                            _162_SIGMOID_43_OUTPUT_0_MEM_WRAP_OFFSET,
                            _162_SIGMOID_43_OUTPUT_0_MEM_WRAP_SIZE,
                            _162_SIGMOID_43_OUTPUT_0_DEV_FMT>
                            (_162_Sigmoid_43_output_0,
                            _162_SIGMOID_43_OUTPUT_0_STREAM);
    fclose(_162_SIGMOID_43_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _163_Mul_41_output_0 = (float*) (mem_0 + _163_MUL_41_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_163_MUL_41_NB_MAT,
                                 _163_MUL_41_ELEM_OP,
                                 _163_MUL_41_INPUT1_CONT_SIZE,
                                 _163_MUL_41_INPUT2_CONT_SIZE,
                                 _163_MUL_41_OUTPUT_CONT_SIZE,
                                 _163_MUL_41_OFFSET_IN1,
                                 _163_MUL_41_OFFSET_IN2,
                                 _163_MUL_41_ACTIVATION,
                                 _157_PADCONV_21_OUTPUT_0_MEM_CONT_OFFSET,
                                 _157_PADCONV_21_OUTPUT_0_MEM_CONT_SIZE,
                                 _157_PADCONV_21_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _157_PADCONV_21_OUTPUT_0_MEM_WRAP_SIZE,
                                 _157_PADCONV_21_OUTPUT_0_MEM_STRIDE,
                                 _159_SIGMOID_41_OUTPUT_0_MEM_CONT_OFFSET,
                                 _159_SIGMOID_41_OUTPUT_0_MEM_CONT_SIZE,
                                 _159_SIGMOID_41_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _159_SIGMOID_41_OUTPUT_0_MEM_WRAP_SIZE,
                                 _159_SIGMOID_41_OUTPUT_0_MEM_STRIDE,
                                 _163_MUL_41_OUTPUT_0_MEM_CONT_OFFSET,
                                 _163_MUL_41_OUTPUT_0_MEM_CONT_SIZE,
                                 _163_MUL_41_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _163_MUL_41_OUTPUT_0_MEM_WRAP_SIZE,
                                 _163_MUL_41_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_163_Mul_41_output_0, _163_MUL_41_RESCALING, _157_PadConv_21_output_0, _159_Sigmoid_41_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _163_Mul_41\n");
    FILE* _163_MUL_41_OUTPUT_0_STREAM = fopen("data/export_outputs/_163_Mul_41_output_0.txt", "w");
    export_cpp::saveOutputs<_163_MUL_41_OUTPUT_0_NB_OUTPUTS,
                            _163_MUL_41_OUTPUT_0_OUT_HEIGHT,
                            _163_MUL_41_OUTPUT_0_OUT_WIDTH,
                            _163_MUL_41_OUTPUT_0_MEM_CONT_OFFSET,
                            _163_MUL_41_OUTPUT_0_MEM_CONT_SIZE,
                            _163_MUL_41_OUTPUT_0_MEM_WRAP_OFFSET,
                            _163_MUL_41_OUTPUT_0_MEM_WRAP_SIZE,
                            _163_MUL_41_OUTPUT_0_DEV_FMT>
                            (_163_Mul_41_output_0,
                            _163_MUL_41_OUTPUT_0_STREAM);
    fclose(_163_MUL_41_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _164_Mul_42_output_0 = (float*) (mem_0 + _164_MUL_42_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_164_MUL_42_NB_MAT,
                                 _164_MUL_42_ELEM_OP,
                                 _164_MUL_42_INPUT1_CONT_SIZE,
                                 _164_MUL_42_INPUT2_CONT_SIZE,
                                 _164_MUL_42_OUTPUT_CONT_SIZE,
                                 _164_MUL_42_OFFSET_IN1,
                                 _164_MUL_42_OFFSET_IN2,
                                 _164_MUL_42_ACTIVATION,
                                 _158_CONV2D_20_OUTPUT_0_MEM_CONT_OFFSET,
                                 _158_CONV2D_20_OUTPUT_0_MEM_CONT_SIZE,
                                 _158_CONV2D_20_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _158_CONV2D_20_OUTPUT_0_MEM_WRAP_SIZE,
                                 _158_CONV2D_20_OUTPUT_0_MEM_STRIDE,
                                 _160_SIGMOID_42_OUTPUT_0_MEM_CONT_OFFSET,
                                 _160_SIGMOID_42_OUTPUT_0_MEM_CONT_SIZE,
                                 _160_SIGMOID_42_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _160_SIGMOID_42_OUTPUT_0_MEM_WRAP_SIZE,
                                 _160_SIGMOID_42_OUTPUT_0_MEM_STRIDE,
                                 _164_MUL_42_OUTPUT_0_MEM_CONT_OFFSET,
                                 _164_MUL_42_OUTPUT_0_MEM_CONT_SIZE,
                                 _164_MUL_42_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _164_MUL_42_OUTPUT_0_MEM_WRAP_SIZE,
                                 _164_MUL_42_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_164_Mul_42_output_0, _164_MUL_42_RESCALING, _158_Conv2D_20_output_0, _160_Sigmoid_42_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _164_Mul_42\n");
    FILE* _164_MUL_42_OUTPUT_0_STREAM = fopen("data/export_outputs/_164_Mul_42_output_0.txt", "w");
    export_cpp::saveOutputs<_164_MUL_42_OUTPUT_0_NB_OUTPUTS,
                            _164_MUL_42_OUTPUT_0_OUT_HEIGHT,
                            _164_MUL_42_OUTPUT_0_OUT_WIDTH,
                            _164_MUL_42_OUTPUT_0_MEM_CONT_OFFSET,
                            _164_MUL_42_OUTPUT_0_MEM_CONT_SIZE,
                            _164_MUL_42_OUTPUT_0_MEM_WRAP_OFFSET,
                            _164_MUL_42_OUTPUT_0_MEM_WRAP_SIZE,
                            _164_MUL_42_OUTPUT_0_DEV_FMT>
                            (_164_Mul_42_output_0,
                            _164_MUL_42_OUTPUT_0_STREAM);
    fclose(_164_MUL_42_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _165_Mul_43_output_0 = (float*) (mem_0 + _165_MUL_43_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_165_MUL_43_NB_MAT,
                                 _165_MUL_43_ELEM_OP,
                                 _165_MUL_43_INPUT1_CONT_SIZE,
                                 _165_MUL_43_INPUT2_CONT_SIZE,
                                 _165_MUL_43_OUTPUT_CONT_SIZE,
                                 _165_MUL_43_OFFSET_IN1,
                                 _165_MUL_43_OFFSET_IN2,
                                 _165_MUL_43_ACTIVATION,
                                 _161_CONV2D_21_OUTPUT_0_MEM_CONT_OFFSET,
                                 _161_CONV2D_21_OUTPUT_0_MEM_CONT_SIZE,
                                 _161_CONV2D_21_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _161_CONV2D_21_OUTPUT_0_MEM_WRAP_SIZE,
                                 _161_CONV2D_21_OUTPUT_0_MEM_STRIDE,
                                 _162_SIGMOID_43_OUTPUT_0_MEM_CONT_OFFSET,
                                 _162_SIGMOID_43_OUTPUT_0_MEM_CONT_SIZE,
                                 _162_SIGMOID_43_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _162_SIGMOID_43_OUTPUT_0_MEM_WRAP_SIZE,
                                 _162_SIGMOID_43_OUTPUT_0_MEM_STRIDE,
                                 _165_MUL_43_OUTPUT_0_MEM_CONT_OFFSET,
                                 _165_MUL_43_OUTPUT_0_MEM_CONT_SIZE,
                                 _165_MUL_43_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _165_MUL_43_OUTPUT_0_MEM_WRAP_SIZE,
                                 _165_MUL_43_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_165_Mul_43_output_0, _165_MUL_43_RESCALING, _161_Conv2D_21_output_0, _162_Sigmoid_43_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _165_Mul_43\n");
    FILE* _165_MUL_43_OUTPUT_0_STREAM = fopen("data/export_outputs/_165_Mul_43_output_0.txt", "w");
    export_cpp::saveOutputs<_165_MUL_43_OUTPUT_0_NB_OUTPUTS,
                            _165_MUL_43_OUTPUT_0_OUT_HEIGHT,
                            _165_MUL_43_OUTPUT_0_OUT_WIDTH,
                            _165_MUL_43_OUTPUT_0_MEM_CONT_OFFSET,
                            _165_MUL_43_OUTPUT_0_MEM_CONT_SIZE,
                            _165_MUL_43_OUTPUT_0_MEM_WRAP_OFFSET,
                            _165_MUL_43_OUTPUT_0_MEM_WRAP_SIZE,
                            _165_MUL_43_OUTPUT_0_DEV_FMT>
                            (_165_Mul_43_output_0,
                            _165_MUL_43_OUTPUT_0_STREAM);
    fclose(_165_MUL_43_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _166_PadConvDw_3_output_0 = (float*) (mem_0 + _166_PADCONVDW_3_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_164_MUL_42_OUTPUT_0_IN_BATCH,
                           _166_PADCONVDW_3_NB_CHANNELS,
                           _164_MUL_42_OUTPUT_0_IN_HEIGHT,
                           _164_MUL_42_OUTPUT_0_IN_WIDTH,
                           _166_PADCONVDW_3_NB_OUTPUTS,
                           _166_PADCONVDW_3_OUTPUT_0_OUT_HEIGHT,
                           _166_PADCONVDW_3_OUTPUT_0_OUT_WIDTH,
                           _166_PADCONVDW_3_PADDING_Y,
                           _166_PADCONVDW_3_PADDING_X,
                           _166_PADCONVDW_3_STRIDE_Y,
                           _166_PADCONVDW_3_STRIDE_X,
                           _166_PADCONVDW_3_DILATION_Y,
                           _166_PADCONVDW_3_DILATION_X,
                           _166_PADCONVDW_3_KERNEL_HEIGHT,
                           _166_PADCONVDW_3_KERNEL_WIDTH,
                           _166_PADCONVDW_3_ACTIVATION,
                           _164_MUL_42_OUTPUT_0_MEM_CONT_OFFSET,
                           _164_MUL_42_OUTPUT_0_MEM_CONT_SIZE,
                           _164_MUL_42_OUTPUT_0_MEM_WRAP_OFFSET,
                           _164_MUL_42_OUTPUT_0_MEM_WRAP_SIZE,
                           _164_MUL_42_OUTPUT_0_MEM_STRIDE,
                           _166_PADCONVDW_3_OUTPUT_0_MEM_CONT_OFFSET,
                           _166_PADCONVDW_3_OUTPUT_0_MEM_CONT_SIZE,
                           _166_PADCONVDW_3_OUTPUT_0_MEM_WRAP_OFFSET,
                           _166_PADCONVDW_3_OUTPUT_0_MEM_WRAP_SIZE,
                           _166_PADCONVDW_3_OUTPUT_0_MEM_STRIDE>
                           (_164_Mul_42_output_0, _166_PadConvDw_3_output_0, _166_PadConvDw_3_weights_output_0, _166_PadConvDw_3_biases_output_0, _166_PADCONVDW_3_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _166_PadConvDw_3\n");
    FILE* _166_PADCONVDW_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_166_PadConvDw_3_output_0.txt", "w");
    export_cpp::saveOutputs<_166_PADCONVDW_3_OUTPUT_0_NB_OUTPUTS,
                            _166_PADCONVDW_3_OUTPUT_0_OUT_HEIGHT,
                            _166_PADCONVDW_3_OUTPUT_0_OUT_WIDTH,
                            _166_PADCONVDW_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _166_PADCONVDW_3_OUTPUT_0_MEM_CONT_SIZE,
                            _166_PADCONVDW_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _166_PADCONVDW_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _166_PADCONVDW_3_OUTPUT_0_DEV_FMT>
                            (_166_PadConvDw_3_output_0,
                            _166_PADCONVDW_3_OUTPUT_0_STREAM);
    fclose(_166_PADCONVDW_3_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _167_Conv2D_22_output_0 = (float*) (mem_0 + _167_CONV2D_22_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_163_MUL_41_OUTPUT_0_IN_BATCH,
                           _167_CONV2D_22_NB_CHANNELS,
                           _163_MUL_41_OUTPUT_0_IN_HEIGHT,
                           _163_MUL_41_OUTPUT_0_IN_WIDTH,
                           _167_CONV2D_22_NB_OUTPUTS,
                           _167_CONV2D_22_OUTPUT_0_OUT_HEIGHT,
                           _167_CONV2D_22_OUTPUT_0_OUT_WIDTH,
                           _167_CONV2D_22_PADDING_Y,
                           _167_CONV2D_22_PADDING_X,
                           _167_CONV2D_22_STRIDE_Y,
                           _167_CONV2D_22_STRIDE_X,
                           _167_CONV2D_22_DILATION_Y,
                           _167_CONV2D_22_DILATION_X,
                           _167_CONV2D_22_KERNEL_HEIGHT,
                           _167_CONV2D_22_KERNEL_WIDTH,
                           _167_CONV2D_22_ACTIVATION,
                           _163_MUL_41_OUTPUT_0_MEM_CONT_OFFSET,
                           _163_MUL_41_OUTPUT_0_MEM_CONT_SIZE,
                           _163_MUL_41_OUTPUT_0_MEM_WRAP_OFFSET,
                           _163_MUL_41_OUTPUT_0_MEM_WRAP_SIZE,
                           _163_MUL_41_OUTPUT_0_MEM_STRIDE,
                           _167_CONV2D_22_OUTPUT_0_MEM_CONT_OFFSET,
                           _167_CONV2D_22_OUTPUT_0_MEM_CONT_SIZE,
                           _167_CONV2D_22_OUTPUT_0_MEM_WRAP_OFFSET,
                           _167_CONV2D_22_OUTPUT_0_MEM_WRAP_SIZE,
                           _167_CONV2D_22_OUTPUT_0_MEM_STRIDE>
                           (_163_Mul_41_output_0, _167_Conv2D_22_output_0, _167_Conv2D_22_weights_output_0, _167_Conv2D_22_biases_output_0, _167_CONV2D_22_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _167_Conv2D_22\n");
    FILE* _167_CONV2D_22_OUTPUT_0_STREAM = fopen("data/export_outputs/_167_Conv2D_22_output_0.txt", "w");
    export_cpp::saveOutputs<_167_CONV2D_22_OUTPUT_0_NB_OUTPUTS,
                            _167_CONV2D_22_OUTPUT_0_OUT_HEIGHT,
                            _167_CONV2D_22_OUTPUT_0_OUT_WIDTH,
                            _167_CONV2D_22_OUTPUT_0_MEM_CONT_OFFSET,
                            _167_CONV2D_22_OUTPUT_0_MEM_CONT_SIZE,
                            _167_CONV2D_22_OUTPUT_0_MEM_WRAP_OFFSET,
                            _167_CONV2D_22_OUTPUT_0_MEM_WRAP_SIZE,
                            _167_CONV2D_22_OUTPUT_0_DEV_FMT>
                            (_167_Conv2D_22_output_0,
                            _167_CONV2D_22_OUTPUT_0_STREAM);
    fclose(_167_CONV2D_22_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _168_Split_7_output_0 = (float*) (mem_0 + _168_SPLIT_7_OUTPUT_0_MEM_OFFSET);
    float* _168_Split_7_output_1 = (float*) (mem_0 + _168_SPLIT_7_OUTPUT_1_MEM_OFFSET);



    float* __168_Split_7_outputs[_168_SPLIT_7_NB_OUTPUTS] = {
        
        _168_Split_7_output_0,
        
        _168_Split_7_output_1
        
    };

    export_cpp::split_forward<_168_SPLIT_7_NB_OUTPUTS,
                              _168_SPLIT_7_AXIS_DIM,
                              _168_SPLIT_7_PRE_AXIS_ELEMS,
                              _168_SPLIT_7_POST_AXIS_ELEMS,
                              _168_SPLIT_7_SPLITS,
                              float>
                              (_165_Mul_43_output_0,
                               __168_Split_7_outputs);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _168_Split_7\n");
    FILE* _168_SPLIT_7_OUTPUT_0_STREAM = fopen("data/export_outputs/_168_Split_7_output_0.txt", "w");
    export_cpp::saveOutputs<_168_SPLIT_7_OUTPUT_0_NB_OUTPUTS,
                            _168_SPLIT_7_OUTPUT_0_OUT_HEIGHT,
                            _168_SPLIT_7_OUTPUT_0_OUT_WIDTH,
                            _168_SPLIT_7_OUTPUT_0_MEM_CONT_OFFSET,
                            _168_SPLIT_7_OUTPUT_0_MEM_CONT_SIZE,
                            _168_SPLIT_7_OUTPUT_0_MEM_WRAP_OFFSET,
                            _168_SPLIT_7_OUTPUT_0_MEM_WRAP_SIZE,
                            _168_SPLIT_7_OUTPUT_0_DEV_FMT>
                            (_168_Split_7_output_0,
                            _168_SPLIT_7_OUTPUT_0_STREAM);
    fclose(_168_SPLIT_7_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _169_Sigmoid_44_output_0 = (float*) (mem_0 + _169_SIGMOID_44_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_169_SIGMOID_44_NB_ELTS>
                               (_166_PadConvDw_3_output_0, _169_Sigmoid_44_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _169_Sigmoid_44\n");
    FILE* _169_SIGMOID_44_OUTPUT_0_STREAM = fopen("data/export_outputs/_169_Sigmoid_44_output_0.txt", "w");
    export_cpp::saveOutputs<_169_SIGMOID_44_OUTPUT_0_NB_OUTPUTS,
                            _169_SIGMOID_44_OUTPUT_0_OUT_HEIGHT,
                            _169_SIGMOID_44_OUTPUT_0_OUT_WIDTH,
                            _169_SIGMOID_44_OUTPUT_0_MEM_CONT_OFFSET,
                            _169_SIGMOID_44_OUTPUT_0_MEM_CONT_SIZE,
                            _169_SIGMOID_44_OUTPUT_0_MEM_WRAP_OFFSET,
                            _169_SIGMOID_44_OUTPUT_0_MEM_WRAP_SIZE,
                            _169_SIGMOID_44_OUTPUT_0_DEV_FMT>
                            (_169_Sigmoid_44_output_0,
                            _169_SIGMOID_44_OUTPUT_0_STREAM);
    fclose(_169_SIGMOID_44_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _170_Transpose_0_output_0 = (float*) (mem_0 + _170_TRANSPOSE_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::transpose_ND_forward<float,
                                     _170_TRANSPOSE_0_NB_DIMS,
                                     _170_TRANSPOSE_0_NB_ELTS,
                                     _170_TRANSPOSE_0_PERMUTE,
                                     _170_TRANSPOSE_0_DIMS,
                                     _170_TRANSPOSE_0_OUT_STRIDES,
                                    _167_CONV2D_22_OUTPUT_0_MEM_CONT_OFFSET,
                                    _167_CONV2D_22_OUTPUT_0_MEM_CONT_SIZE,
                                    _167_CONV2D_22_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _167_CONV2D_22_OUTPUT_0_MEM_WRAP_SIZE,
                                    _167_CONV2D_22_OUTPUT_0_MEM_STRIDE,
                                    _170_TRANSPOSE_0_OUTPUT_0_MEM_CONT_OFFSET,
                                    _170_TRANSPOSE_0_OUTPUT_0_MEM_CONT_SIZE,
                                    _170_TRANSPOSE_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _170_TRANSPOSE_0_OUTPUT_0_MEM_WRAP_SIZE,
                                    _170_TRANSPOSE_0_OUTPUT_0_MEM_STRIDE>
                                     (_167_Conv2D_22_output_0,
                                     _170_Transpose_0_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _170_Transpose_0\n");
    FILE* _170_TRANSPOSE_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_170_Transpose_0_output_0.txt", "w");
    export_cpp::saveOutputs<_170_TRANSPOSE_0_OUTPUT_0_NB_OUTPUTS,
                            _170_TRANSPOSE_0_OUTPUT_0_OUT_HEIGHT,
                            _170_TRANSPOSE_0_OUTPUT_0_OUT_WIDTH,
                            _170_TRANSPOSE_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _170_TRANSPOSE_0_OUTPUT_0_MEM_CONT_SIZE,
                            _170_TRANSPOSE_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _170_TRANSPOSE_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _170_TRANSPOSE_0_OUTPUT_0_DEV_FMT>
                            (_170_Transpose_0_output_0,
                            _170_TRANSPOSE_0_OUTPUT_0_STREAM);
    fclose(_170_TRANSPOSE_0_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _171_Mul_44_output_0 = (float*) (mem_0 + _171_MUL_44_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_171_MUL_44_NB_MAT,
                                 _171_MUL_44_ELEM_OP,
                                 _171_MUL_44_INPUT1_CONT_SIZE,
                                 _171_MUL_44_INPUT2_CONT_SIZE,
                                 _171_MUL_44_OUTPUT_CONT_SIZE,
                                 _171_MUL_44_OFFSET_IN1,
                                 _171_MUL_44_OFFSET_IN2,
                                 _171_MUL_44_ACTIVATION,
                                 _166_PADCONVDW_3_OUTPUT_0_MEM_CONT_OFFSET,
                                 _166_PADCONVDW_3_OUTPUT_0_MEM_CONT_SIZE,
                                 _166_PADCONVDW_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _166_PADCONVDW_3_OUTPUT_0_MEM_WRAP_SIZE,
                                 _166_PADCONVDW_3_OUTPUT_0_MEM_STRIDE,
                                 _169_SIGMOID_44_OUTPUT_0_MEM_CONT_OFFSET,
                                 _169_SIGMOID_44_OUTPUT_0_MEM_CONT_SIZE,
                                 _169_SIGMOID_44_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _169_SIGMOID_44_OUTPUT_0_MEM_WRAP_SIZE,
                                 _169_SIGMOID_44_OUTPUT_0_MEM_STRIDE,
                                 _171_MUL_44_OUTPUT_0_MEM_CONT_OFFSET,
                                 _171_MUL_44_OUTPUT_0_MEM_CONT_SIZE,
                                 _171_MUL_44_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _171_MUL_44_OUTPUT_0_MEM_WRAP_SIZE,
                                 _171_MUL_44_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_171_Mul_44_output_0, _171_MUL_44_RESCALING, _166_PadConvDw_3_output_0, _169_Sigmoid_44_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _171_Mul_44\n");
    FILE* _171_MUL_44_OUTPUT_0_STREAM = fopen("data/export_outputs/_171_Mul_44_output_0.txt", "w");
    export_cpp::saveOutputs<_171_MUL_44_OUTPUT_0_NB_OUTPUTS,
                            _171_MUL_44_OUTPUT_0_OUT_HEIGHT,
                            _171_MUL_44_OUTPUT_0_OUT_WIDTH,
                            _171_MUL_44_OUTPUT_0_MEM_CONT_OFFSET,
                            _171_MUL_44_OUTPUT_0_MEM_CONT_SIZE,
                            _171_MUL_44_OUTPUT_0_MEM_WRAP_OFFSET,
                            _171_MUL_44_OUTPUT_0_MEM_WRAP_SIZE,
                            _171_MUL_44_OUTPUT_0_DEV_FMT>
                            (_171_Mul_44_output_0,
                            _171_MUL_44_OUTPUT_0_STREAM);
    fclose(_171_MUL_44_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _172_PadConv_22_output_0 = (float*) (mem_0 + _172_PADCONV_22_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_168_SPLIT_7_OUTPUT_1_IN_BATCH,
                           _172_PADCONV_22_NB_CHANNELS,
                           _168_SPLIT_7_OUTPUT_1_IN_HEIGHT,
                           _168_SPLIT_7_OUTPUT_1_IN_WIDTH,
                           _172_PADCONV_22_NB_OUTPUTS,
                           _172_PADCONV_22_OUTPUT_0_OUT_HEIGHT,
                           _172_PADCONV_22_OUTPUT_0_OUT_WIDTH,
                           _172_PADCONV_22_PADDING_Y,
                           _172_PADCONV_22_PADDING_X,
                           _172_PADCONV_22_STRIDE_Y,
                           _172_PADCONV_22_STRIDE_X,
                           _172_PADCONV_22_DILATION_Y,
                           _172_PADCONV_22_DILATION_X,
                           _172_PADCONV_22_KERNEL_HEIGHT,
                           _172_PADCONV_22_KERNEL_WIDTH,
                           _172_PADCONV_22_ACTIVATION,
                           _168_SPLIT_7_OUTPUT_1_MEM_CONT_OFFSET,
                           _168_SPLIT_7_OUTPUT_1_MEM_CONT_SIZE,
                           _168_SPLIT_7_OUTPUT_1_MEM_WRAP_OFFSET,
                           _168_SPLIT_7_OUTPUT_1_MEM_WRAP_SIZE,
                           _168_SPLIT_7_OUTPUT_1_MEM_STRIDE,
                           _172_PADCONV_22_OUTPUT_0_MEM_CONT_OFFSET,
                           _172_PADCONV_22_OUTPUT_0_MEM_CONT_SIZE,
                           _172_PADCONV_22_OUTPUT_0_MEM_WRAP_OFFSET,
                           _172_PADCONV_22_OUTPUT_0_MEM_WRAP_SIZE,
                           _172_PADCONV_22_OUTPUT_0_MEM_STRIDE>
                           (_168_Split_7_output_1, _172_PadConv_22_output_0, _172_PadConv_22_weights_output_0, _172_PadConv_22_biases_output_0, _172_PADCONV_22_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _172_PadConv_22\n");
    FILE* _172_PADCONV_22_OUTPUT_0_STREAM = fopen("data/export_outputs/_172_PadConv_22_output_0.txt", "w");
    export_cpp::saveOutputs<_172_PADCONV_22_OUTPUT_0_NB_OUTPUTS,
                            _172_PADCONV_22_OUTPUT_0_OUT_HEIGHT,
                            _172_PADCONV_22_OUTPUT_0_OUT_WIDTH,
                            _172_PADCONV_22_OUTPUT_0_MEM_CONT_OFFSET,
                            _172_PADCONV_22_OUTPUT_0_MEM_CONT_SIZE,
                            _172_PADCONV_22_OUTPUT_0_MEM_WRAP_OFFSET,
                            _172_PADCONV_22_OUTPUT_0_MEM_WRAP_SIZE,
                            _172_PADCONV_22_OUTPUT_0_DEV_FMT>
                            (_172_PadConv_22_output_0,
                            _172_PADCONV_22_OUTPUT_0_STREAM);
    fclose(_172_PADCONV_22_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _173_Reshape_0_output_0 = (float*) (mem_0 + _173_RESHAPE_0_OUTPUT_0_MEM_OFFSET);


    export_cpp::identity_forward<_173_RESHAPE_0_NB_ELTS,
                                 _170_TRANSPOSE_0_OUTPUT_0_MEM_CONT_OFFSET,
                                 _170_TRANSPOSE_0_OUTPUT_0_MEM_CONT_SIZE,
                                 _170_TRANSPOSE_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _170_TRANSPOSE_0_OUTPUT_0_MEM_WRAP_SIZE,
                                 _170_TRANSPOSE_0_OUTPUT_0_MEM_STRIDE,
                                 _173_RESHAPE_0_OUTPUT_0_MEM_CONT_OFFSET,
                                 _173_RESHAPE_0_OUTPUT_0_MEM_CONT_SIZE,
                                 _173_RESHAPE_0_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _173_RESHAPE_0_OUTPUT_0_MEM_WRAP_SIZE,
                                 _173_RESHAPE_0_OUTPUT_0_MEM_STRIDE>
                                (_170_Transpose_0_output_0, _173_Reshape_0_output_0);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _173_Reshape_0\n");
    FILE* _173_RESHAPE_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_173_Reshape_0_output_0.txt", "w");
    export_cpp::saveOutputs<_173_RESHAPE_0_OUTPUT_0_NB_OUTPUTS,
                            _173_RESHAPE_0_OUTPUT_0_OUT_HEIGHT,
                            _173_RESHAPE_0_OUTPUT_0_OUT_WIDTH,
                            _173_RESHAPE_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _173_RESHAPE_0_OUTPUT_0_MEM_CONT_SIZE,
                            _173_RESHAPE_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _173_RESHAPE_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _173_RESHAPE_0_OUTPUT_0_DEV_FMT>
                            (_173_Reshape_0_output_0,
                            _173_RESHAPE_0_OUTPUT_0_STREAM);
    fclose(_173_RESHAPE_0_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _174_Sigmoid_45_output_0 = (float*) (mem_0 + _174_SIGMOID_45_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_174_SIGMOID_45_NB_ELTS>
                               (_172_PadConv_22_output_0, _174_Sigmoid_45_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _174_Sigmoid_45\n");
    FILE* _174_SIGMOID_45_OUTPUT_0_STREAM = fopen("data/export_outputs/_174_Sigmoid_45_output_0.txt", "w");
    export_cpp::saveOutputs<_174_SIGMOID_45_OUTPUT_0_NB_OUTPUTS,
                            _174_SIGMOID_45_OUTPUT_0_OUT_HEIGHT,
                            _174_SIGMOID_45_OUTPUT_0_OUT_WIDTH,
                            _174_SIGMOID_45_OUTPUT_0_MEM_CONT_OFFSET,
                            _174_SIGMOID_45_OUTPUT_0_MEM_CONT_SIZE,
                            _174_SIGMOID_45_OUTPUT_0_MEM_WRAP_OFFSET,
                            _174_SIGMOID_45_OUTPUT_0_MEM_WRAP_SIZE,
                            _174_SIGMOID_45_OUTPUT_0_DEV_FMT>
                            (_174_Sigmoid_45_output_0,
                            _174_SIGMOID_45_OUTPUT_0_STREAM);
    fclose(_174_SIGMOID_45_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _175_Conv2D_23_output_0 = (float*) (mem_0 + _175_CONV2D_23_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_171_MUL_44_OUTPUT_0_IN_BATCH,
                           _175_CONV2D_23_NB_CHANNELS,
                           _171_MUL_44_OUTPUT_0_IN_HEIGHT,
                           _171_MUL_44_OUTPUT_0_IN_WIDTH,
                           _175_CONV2D_23_NB_OUTPUTS,
                           _175_CONV2D_23_OUTPUT_0_OUT_HEIGHT,
                           _175_CONV2D_23_OUTPUT_0_OUT_WIDTH,
                           _175_CONV2D_23_PADDING_Y,
                           _175_CONV2D_23_PADDING_X,
                           _175_CONV2D_23_STRIDE_Y,
                           _175_CONV2D_23_STRIDE_X,
                           _175_CONV2D_23_DILATION_Y,
                           _175_CONV2D_23_DILATION_X,
                           _175_CONV2D_23_KERNEL_HEIGHT,
                           _175_CONV2D_23_KERNEL_WIDTH,
                           _175_CONV2D_23_ACTIVATION,
                           _171_MUL_44_OUTPUT_0_MEM_CONT_OFFSET,
                           _171_MUL_44_OUTPUT_0_MEM_CONT_SIZE,
                           _171_MUL_44_OUTPUT_0_MEM_WRAP_OFFSET,
                           _171_MUL_44_OUTPUT_0_MEM_WRAP_SIZE,
                           _171_MUL_44_OUTPUT_0_MEM_STRIDE,
                           _175_CONV2D_23_OUTPUT_0_MEM_CONT_OFFSET,
                           _175_CONV2D_23_OUTPUT_0_MEM_CONT_SIZE,
                           _175_CONV2D_23_OUTPUT_0_MEM_WRAP_OFFSET,
                           _175_CONV2D_23_OUTPUT_0_MEM_WRAP_SIZE,
                           _175_CONV2D_23_OUTPUT_0_MEM_STRIDE>
                           (_171_Mul_44_output_0, _175_Conv2D_23_output_0, _175_Conv2D_23_weights_output_0, _175_Conv2D_23_biases_output_0, _175_CONV2D_23_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _175_Conv2D_23\n");
    FILE* _175_CONV2D_23_OUTPUT_0_STREAM = fopen("data/export_outputs/_175_Conv2D_23_output_0.txt", "w");
    export_cpp::saveOutputs<_175_CONV2D_23_OUTPUT_0_NB_OUTPUTS,
                            _175_CONV2D_23_OUTPUT_0_OUT_HEIGHT,
                            _175_CONV2D_23_OUTPUT_0_OUT_WIDTH,
                            _175_CONV2D_23_OUTPUT_0_MEM_CONT_OFFSET,
                            _175_CONV2D_23_OUTPUT_0_MEM_CONT_SIZE,
                            _175_CONV2D_23_OUTPUT_0_MEM_WRAP_OFFSET,
                            _175_CONV2D_23_OUTPUT_0_MEM_WRAP_SIZE,
                            _175_CONV2D_23_OUTPUT_0_DEV_FMT>
                            (_175_Conv2D_23_output_0,
                            _175_CONV2D_23_OUTPUT_0_STREAM);
    fclose(_175_CONV2D_23_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _176_Mul_45_output_0 = (float*) (mem_0 + _176_MUL_45_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_176_MUL_45_NB_MAT,
                                 _176_MUL_45_ELEM_OP,
                                 _176_MUL_45_INPUT1_CONT_SIZE,
                                 _176_MUL_45_INPUT2_CONT_SIZE,
                                 _176_MUL_45_OUTPUT_CONT_SIZE,
                                 _176_MUL_45_OFFSET_IN1,
                                 _176_MUL_45_OFFSET_IN2,
                                 _176_MUL_45_ACTIVATION,
                                 _172_PADCONV_22_OUTPUT_0_MEM_CONT_OFFSET,
                                 _172_PADCONV_22_OUTPUT_0_MEM_CONT_SIZE,
                                 _172_PADCONV_22_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _172_PADCONV_22_OUTPUT_0_MEM_WRAP_SIZE,
                                 _172_PADCONV_22_OUTPUT_0_MEM_STRIDE,
                                 _174_SIGMOID_45_OUTPUT_0_MEM_CONT_OFFSET,
                                 _174_SIGMOID_45_OUTPUT_0_MEM_CONT_SIZE,
                                 _174_SIGMOID_45_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _174_SIGMOID_45_OUTPUT_0_MEM_WRAP_SIZE,
                                 _174_SIGMOID_45_OUTPUT_0_MEM_STRIDE,
                                 _176_MUL_45_OUTPUT_0_MEM_CONT_OFFSET,
                                 _176_MUL_45_OUTPUT_0_MEM_CONT_SIZE,
                                 _176_MUL_45_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _176_MUL_45_OUTPUT_0_MEM_WRAP_SIZE,
                                 _176_MUL_45_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_176_Mul_45_output_0, _176_MUL_45_RESCALING, _172_PadConv_22_output_0, _174_Sigmoid_45_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _176_Mul_45\n");
    FILE* _176_MUL_45_OUTPUT_0_STREAM = fopen("data/export_outputs/_176_Mul_45_output_0.txt", "w");
    export_cpp::saveOutputs<_176_MUL_45_OUTPUT_0_NB_OUTPUTS,
                            _176_MUL_45_OUTPUT_0_OUT_HEIGHT,
                            _176_MUL_45_OUTPUT_0_OUT_WIDTH,
                            _176_MUL_45_OUTPUT_0_MEM_CONT_OFFSET,
                            _176_MUL_45_OUTPUT_0_MEM_CONT_SIZE,
                            _176_MUL_45_OUTPUT_0_MEM_WRAP_OFFSET,
                            _176_MUL_45_OUTPUT_0_MEM_WRAP_SIZE,
                            _176_MUL_45_OUTPUT_0_DEV_FMT>
                            (_176_Mul_45_output_0,
                            _176_MUL_45_OUTPUT_0_STREAM);
    fclose(_176_MUL_45_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _177_Sigmoid_46_output_0 = (float*) (mem_0 + _177_SIGMOID_46_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_177_SIGMOID_46_NB_ELTS>
                               (_175_Conv2D_23_output_0, _177_Sigmoid_46_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _177_Sigmoid_46\n");
    FILE* _177_SIGMOID_46_OUTPUT_0_STREAM = fopen("data/export_outputs/_177_Sigmoid_46_output_0.txt", "w");
    export_cpp::saveOutputs<_177_SIGMOID_46_OUTPUT_0_NB_OUTPUTS,
                            _177_SIGMOID_46_OUTPUT_0_OUT_HEIGHT,
                            _177_SIGMOID_46_OUTPUT_0_OUT_WIDTH,
                            _177_SIGMOID_46_OUTPUT_0_MEM_CONT_OFFSET,
                            _177_SIGMOID_46_OUTPUT_0_MEM_CONT_SIZE,
                            _177_SIGMOID_46_OUTPUT_0_MEM_WRAP_OFFSET,
                            _177_SIGMOID_46_OUTPUT_0_MEM_WRAP_SIZE,
                            _177_SIGMOID_46_OUTPUT_0_DEV_FMT>
                            (_177_Sigmoid_46_output_0,
                            _177_SIGMOID_46_OUTPUT_0_STREAM);
    fclose(_177_SIGMOID_46_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _178_Mul_46_output_0 = (float*) (mem_0 + _178_MUL_46_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_178_MUL_46_NB_MAT,
                                 _178_MUL_46_ELEM_OP,
                                 _178_MUL_46_INPUT1_CONT_SIZE,
                                 _178_MUL_46_INPUT2_CONT_SIZE,
                                 _178_MUL_46_OUTPUT_CONT_SIZE,
                                 _178_MUL_46_OFFSET_IN1,
                                 _178_MUL_46_OFFSET_IN2,
                                 _178_MUL_46_ACTIVATION,
                                 _175_CONV2D_23_OUTPUT_0_MEM_CONT_OFFSET,
                                 _175_CONV2D_23_OUTPUT_0_MEM_CONT_SIZE,
                                 _175_CONV2D_23_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _175_CONV2D_23_OUTPUT_0_MEM_WRAP_SIZE,
                                 _175_CONV2D_23_OUTPUT_0_MEM_STRIDE,
                                 _177_SIGMOID_46_OUTPUT_0_MEM_CONT_OFFSET,
                                 _177_SIGMOID_46_OUTPUT_0_MEM_CONT_SIZE,
                                 _177_SIGMOID_46_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _177_SIGMOID_46_OUTPUT_0_MEM_WRAP_SIZE,
                                 _177_SIGMOID_46_OUTPUT_0_MEM_STRIDE,
                                 _178_MUL_46_OUTPUT_0_MEM_CONT_OFFSET,
                                 _178_MUL_46_OUTPUT_0_MEM_CONT_SIZE,
                                 _178_MUL_46_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _178_MUL_46_OUTPUT_0_MEM_WRAP_SIZE,
                                 _178_MUL_46_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_178_Mul_46_output_0, _178_MUL_46_RESCALING, _175_Conv2D_23_output_0, _177_Sigmoid_46_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _178_Mul_46\n");
    FILE* _178_MUL_46_OUTPUT_0_STREAM = fopen("data/export_outputs/_178_Mul_46_output_0.txt", "w");
    export_cpp::saveOutputs<_178_MUL_46_OUTPUT_0_NB_OUTPUTS,
                            _178_MUL_46_OUTPUT_0_OUT_HEIGHT,
                            _178_MUL_46_OUTPUT_0_OUT_WIDTH,
                            _178_MUL_46_OUTPUT_0_MEM_CONT_OFFSET,
                            _178_MUL_46_OUTPUT_0_MEM_CONT_SIZE,
                            _178_MUL_46_OUTPUT_0_MEM_WRAP_OFFSET,
                            _178_MUL_46_OUTPUT_0_MEM_WRAP_SIZE,
                            _178_MUL_46_OUTPUT_0_DEV_FMT>
                            (_178_Mul_46_output_0,
                            _178_MUL_46_OUTPUT_0_STREAM);
    fclose(_178_MUL_46_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _179_PadConv_23_output_0 = (float*) (mem_0 + _179_PADCONV_23_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_176_MUL_45_OUTPUT_0_IN_BATCH,
                           _179_PADCONV_23_NB_CHANNELS,
                           _176_MUL_45_OUTPUT_0_IN_HEIGHT,
                           _176_MUL_45_OUTPUT_0_IN_WIDTH,
                           _179_PADCONV_23_NB_OUTPUTS,
                           _179_PADCONV_23_OUTPUT_0_OUT_HEIGHT,
                           _179_PADCONV_23_OUTPUT_0_OUT_WIDTH,
                           _179_PADCONV_23_PADDING_Y,
                           _179_PADCONV_23_PADDING_X,
                           _179_PADCONV_23_STRIDE_Y,
                           _179_PADCONV_23_STRIDE_X,
                           _179_PADCONV_23_DILATION_Y,
                           _179_PADCONV_23_DILATION_X,
                           _179_PADCONV_23_KERNEL_HEIGHT,
                           _179_PADCONV_23_KERNEL_WIDTH,
                           _179_PADCONV_23_ACTIVATION,
                           _176_MUL_45_OUTPUT_0_MEM_CONT_OFFSET,
                           _176_MUL_45_OUTPUT_0_MEM_CONT_SIZE,
                           _176_MUL_45_OUTPUT_0_MEM_WRAP_OFFSET,
                           _176_MUL_45_OUTPUT_0_MEM_WRAP_SIZE,
                           _176_MUL_45_OUTPUT_0_MEM_STRIDE,
                           _179_PADCONV_23_OUTPUT_0_MEM_CONT_OFFSET,
                           _179_PADCONV_23_OUTPUT_0_MEM_CONT_SIZE,
                           _179_PADCONV_23_OUTPUT_0_MEM_WRAP_OFFSET,
                           _179_PADCONV_23_OUTPUT_0_MEM_WRAP_SIZE,
                           _179_PADCONV_23_OUTPUT_0_MEM_STRIDE>
                           (_176_Mul_45_output_0, _179_PadConv_23_output_0, _179_PadConv_23_weights_output_0, _179_PadConv_23_biases_output_0, _179_PADCONV_23_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _179_PadConv_23\n");
    FILE* _179_PADCONV_23_OUTPUT_0_STREAM = fopen("data/export_outputs/_179_PadConv_23_output_0.txt", "w");
    export_cpp::saveOutputs<_179_PADCONV_23_OUTPUT_0_NB_OUTPUTS,
                            _179_PADCONV_23_OUTPUT_0_OUT_HEIGHT,
                            _179_PADCONV_23_OUTPUT_0_OUT_WIDTH,
                            _179_PADCONV_23_OUTPUT_0_MEM_CONT_OFFSET,
                            _179_PADCONV_23_OUTPUT_0_MEM_CONT_SIZE,
                            _179_PADCONV_23_OUTPUT_0_MEM_WRAP_OFFSET,
                            _179_PADCONV_23_OUTPUT_0_MEM_WRAP_SIZE,
                            _179_PADCONV_23_OUTPUT_0_DEV_FMT>
                            (_179_PadConv_23_output_0,
                            _179_PADCONV_23_OUTPUT_0_STREAM);
    fclose(_179_PADCONV_23_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _180_Sigmoid_47_output_0 = (float*) (mem_0 + _180_SIGMOID_47_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_180_SIGMOID_47_NB_ELTS>
                               (_179_PadConv_23_output_0, _180_Sigmoid_47_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _180_Sigmoid_47\n");
    FILE* _180_SIGMOID_47_OUTPUT_0_STREAM = fopen("data/export_outputs/_180_Sigmoid_47_output_0.txt", "w");
    export_cpp::saveOutputs<_180_SIGMOID_47_OUTPUT_0_NB_OUTPUTS,
                            _180_SIGMOID_47_OUTPUT_0_OUT_HEIGHT,
                            _180_SIGMOID_47_OUTPUT_0_OUT_WIDTH,
                            _180_SIGMOID_47_OUTPUT_0_MEM_CONT_OFFSET,
                            _180_SIGMOID_47_OUTPUT_0_MEM_CONT_SIZE,
                            _180_SIGMOID_47_OUTPUT_0_MEM_WRAP_OFFSET,
                            _180_SIGMOID_47_OUTPUT_0_MEM_WRAP_SIZE,
                            _180_SIGMOID_47_OUTPUT_0_DEV_FMT>
                            (_180_Sigmoid_47_output_0,
                            _180_SIGMOID_47_OUTPUT_0_STREAM);
    fclose(_180_SIGMOID_47_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _181_Conv2D_24_output_0 = (float*) (mem_0 + _181_CONV2D_24_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_178_MUL_46_OUTPUT_0_IN_BATCH,
                           _181_CONV2D_24_NB_CHANNELS,
                           _178_MUL_46_OUTPUT_0_IN_HEIGHT,
                           _178_MUL_46_OUTPUT_0_IN_WIDTH,
                           _181_CONV2D_24_NB_OUTPUTS,
                           _181_CONV2D_24_OUTPUT_0_OUT_HEIGHT,
                           _181_CONV2D_24_OUTPUT_0_OUT_WIDTH,
                           _181_CONV2D_24_PADDING_Y,
                           _181_CONV2D_24_PADDING_X,
                           _181_CONV2D_24_STRIDE_Y,
                           _181_CONV2D_24_STRIDE_X,
                           _181_CONV2D_24_DILATION_Y,
                           _181_CONV2D_24_DILATION_X,
                           _181_CONV2D_24_KERNEL_HEIGHT,
                           _181_CONV2D_24_KERNEL_WIDTH,
                           _181_CONV2D_24_ACTIVATION,
                           _178_MUL_46_OUTPUT_0_MEM_CONT_OFFSET,
                           _178_MUL_46_OUTPUT_0_MEM_CONT_SIZE,
                           _178_MUL_46_OUTPUT_0_MEM_WRAP_OFFSET,
                           _178_MUL_46_OUTPUT_0_MEM_WRAP_SIZE,
                           _178_MUL_46_OUTPUT_0_MEM_STRIDE,
                           _181_CONV2D_24_OUTPUT_0_MEM_CONT_OFFSET,
                           _181_CONV2D_24_OUTPUT_0_MEM_CONT_SIZE,
                           _181_CONV2D_24_OUTPUT_0_MEM_WRAP_OFFSET,
                           _181_CONV2D_24_OUTPUT_0_MEM_WRAP_SIZE,
                           _181_CONV2D_24_OUTPUT_0_MEM_STRIDE>
                           (_178_Mul_46_output_0, _181_Conv2D_24_output_0, _181_Conv2D_24_weights_output_0, _181_Conv2D_24_biases_output_0, _181_CONV2D_24_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _181_Conv2D_24\n");
    FILE* _181_CONV2D_24_OUTPUT_0_STREAM = fopen("data/export_outputs/_181_Conv2D_24_output_0.txt", "w");
    export_cpp::saveOutputs<_181_CONV2D_24_OUTPUT_0_NB_OUTPUTS,
                            _181_CONV2D_24_OUTPUT_0_OUT_HEIGHT,
                            _181_CONV2D_24_OUTPUT_0_OUT_WIDTH,
                            _181_CONV2D_24_OUTPUT_0_MEM_CONT_OFFSET,
                            _181_CONV2D_24_OUTPUT_0_MEM_CONT_SIZE,
                            _181_CONV2D_24_OUTPUT_0_MEM_WRAP_OFFSET,
                            _181_CONV2D_24_OUTPUT_0_MEM_WRAP_SIZE,
                            _181_CONV2D_24_OUTPUT_0_DEV_FMT>
                            (_181_Conv2D_24_output_0,
                            _181_CONV2D_24_OUTPUT_0_STREAM);
    fclose(_181_CONV2D_24_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _182_Mul_47_output_0 = (float*) (mem_0 + _182_MUL_47_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_182_MUL_47_NB_MAT,
                                 _182_MUL_47_ELEM_OP,
                                 _182_MUL_47_INPUT1_CONT_SIZE,
                                 _182_MUL_47_INPUT2_CONT_SIZE,
                                 _182_MUL_47_OUTPUT_CONT_SIZE,
                                 _182_MUL_47_OFFSET_IN1,
                                 _182_MUL_47_OFFSET_IN2,
                                 _182_MUL_47_ACTIVATION,
                                 _179_PADCONV_23_OUTPUT_0_MEM_CONT_OFFSET,
                                 _179_PADCONV_23_OUTPUT_0_MEM_CONT_SIZE,
                                 _179_PADCONV_23_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _179_PADCONV_23_OUTPUT_0_MEM_WRAP_SIZE,
                                 _179_PADCONV_23_OUTPUT_0_MEM_STRIDE,
                                 _180_SIGMOID_47_OUTPUT_0_MEM_CONT_OFFSET,
                                 _180_SIGMOID_47_OUTPUT_0_MEM_CONT_SIZE,
                                 _180_SIGMOID_47_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _180_SIGMOID_47_OUTPUT_0_MEM_WRAP_SIZE,
                                 _180_SIGMOID_47_OUTPUT_0_MEM_STRIDE,
                                 _182_MUL_47_OUTPUT_0_MEM_CONT_OFFSET,
                                 _182_MUL_47_OUTPUT_0_MEM_CONT_SIZE,
                                 _182_MUL_47_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _182_MUL_47_OUTPUT_0_MEM_WRAP_SIZE,
                                 _182_MUL_47_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_182_Mul_47_output_0, _182_MUL_47_RESCALING, _179_PadConv_23_output_0, _180_Sigmoid_47_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _182_Mul_47\n");
    FILE* _182_MUL_47_OUTPUT_0_STREAM = fopen("data/export_outputs/_182_Mul_47_output_0.txt", "w");
    export_cpp::saveOutputs<_182_MUL_47_OUTPUT_0_NB_OUTPUTS,
                            _182_MUL_47_OUTPUT_0_OUT_HEIGHT,
                            _182_MUL_47_OUTPUT_0_OUT_WIDTH,
                            _182_MUL_47_OUTPUT_0_MEM_CONT_OFFSET,
                            _182_MUL_47_OUTPUT_0_MEM_CONT_SIZE,
                            _182_MUL_47_OUTPUT_0_MEM_WRAP_OFFSET,
                            _182_MUL_47_OUTPUT_0_MEM_WRAP_SIZE,
                            _182_MUL_47_OUTPUT_0_DEV_FMT>
                            (_182_Mul_47_output_0,
                            _182_MUL_47_OUTPUT_0_STREAM);
    fclose(_182_MUL_47_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _183_Transpose_1_output_0 = (float*) (mem_0 + _183_TRANSPOSE_1_OUTPUT_0_MEM_OFFSET);

    export_cpp::transpose_ND_forward<float,
                                     _183_TRANSPOSE_1_NB_DIMS,
                                     _183_TRANSPOSE_1_NB_ELTS,
                                     _183_TRANSPOSE_1_PERMUTE,
                                     _183_TRANSPOSE_1_DIMS,
                                     _183_TRANSPOSE_1_OUT_STRIDES,
                                    _181_CONV2D_24_OUTPUT_0_MEM_CONT_OFFSET,
                                    _181_CONV2D_24_OUTPUT_0_MEM_CONT_SIZE,
                                    _181_CONV2D_24_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _181_CONV2D_24_OUTPUT_0_MEM_WRAP_SIZE,
                                    _181_CONV2D_24_OUTPUT_0_MEM_STRIDE,
                                    _183_TRANSPOSE_1_OUTPUT_0_MEM_CONT_OFFSET,
                                    _183_TRANSPOSE_1_OUTPUT_0_MEM_CONT_SIZE,
                                    _183_TRANSPOSE_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _183_TRANSPOSE_1_OUTPUT_0_MEM_WRAP_SIZE,
                                    _183_TRANSPOSE_1_OUTPUT_0_MEM_STRIDE>
                                     (_181_Conv2D_24_output_0,
                                     _183_Transpose_1_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _183_Transpose_1\n");
    FILE* _183_TRANSPOSE_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_183_Transpose_1_output_0.txt", "w");
    export_cpp::saveOutputs<_183_TRANSPOSE_1_OUTPUT_0_NB_OUTPUTS,
                            _183_TRANSPOSE_1_OUTPUT_0_OUT_HEIGHT,
                            _183_TRANSPOSE_1_OUTPUT_0_OUT_WIDTH,
                            _183_TRANSPOSE_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _183_TRANSPOSE_1_OUTPUT_0_MEM_CONT_SIZE,
                            _183_TRANSPOSE_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _183_TRANSPOSE_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _183_TRANSPOSE_1_OUTPUT_0_DEV_FMT>
                            (_183_Transpose_1_output_0,
                            _183_TRANSPOSE_1_OUTPUT_0_STREAM);
    fclose(_183_TRANSPOSE_1_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _184_Concat_11_output_0 = (float*) (mem_0 + _184_CONCAT_11_OUTPUT_0_MEM_OFFSET);

    const float* _184_Concat_11_inputs[] = { _168_Split_7_output_0, _168_Split_7_output_1, _182_Mul_47_output_0 };
    export_cpp::concat_forward<_184_CONCAT_11_AXIS_SIZE_POST,
                               _184_CONCAT_11_AXIS_SIZE_PRE,
                               _184_CONCAT_11_AXIS_SIZE,
                               _184_CONCAT_11_TOTAL_AXIS_SIZE,
                               3> 
                              (_184_Concat_11_inputs,
                               _184_Concat_11_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _184_Concat_11\n");
    FILE* _184_CONCAT_11_OUTPUT_0_STREAM = fopen("data/export_outputs/_184_Concat_11_output_0.txt", "w");
    export_cpp::saveOutputs<_184_CONCAT_11_OUTPUT_0_NB_OUTPUTS,
                            _184_CONCAT_11_OUTPUT_0_OUT_HEIGHT,
                            _184_CONCAT_11_OUTPUT_0_OUT_WIDTH,
                            _184_CONCAT_11_OUTPUT_0_MEM_CONT_OFFSET,
                            _184_CONCAT_11_OUTPUT_0_MEM_CONT_SIZE,
                            _184_CONCAT_11_OUTPUT_0_MEM_WRAP_OFFSET,
                            _184_CONCAT_11_OUTPUT_0_MEM_WRAP_SIZE,
                            _184_CONCAT_11_OUTPUT_0_DEV_FMT>
                            (_184_Concat_11_output_0,
                            _184_CONCAT_11_OUTPUT_0_STREAM);
    fclose(_184_CONCAT_11_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _185_Reshape_1_output_0 = (float*) (mem_0 + _185_RESHAPE_1_OUTPUT_0_MEM_OFFSET);


    export_cpp::identity_forward<_185_RESHAPE_1_NB_ELTS,
                                 _183_TRANSPOSE_1_OUTPUT_0_MEM_CONT_OFFSET,
                                 _183_TRANSPOSE_1_OUTPUT_0_MEM_CONT_SIZE,
                                 _183_TRANSPOSE_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _183_TRANSPOSE_1_OUTPUT_0_MEM_WRAP_SIZE,
                                 _183_TRANSPOSE_1_OUTPUT_0_MEM_STRIDE,
                                 _185_RESHAPE_1_OUTPUT_0_MEM_CONT_OFFSET,
                                 _185_RESHAPE_1_OUTPUT_0_MEM_CONT_SIZE,
                                 _185_RESHAPE_1_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _185_RESHAPE_1_OUTPUT_0_MEM_WRAP_SIZE,
                                 _185_RESHAPE_1_OUTPUT_0_MEM_STRIDE>
                                (_183_Transpose_1_output_0, _185_Reshape_1_output_0);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _185_Reshape_1\n");
    FILE* _185_RESHAPE_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_185_Reshape_1_output_0.txt", "w");
    export_cpp::saveOutputs<_185_RESHAPE_1_OUTPUT_0_NB_OUTPUTS,
                            _185_RESHAPE_1_OUTPUT_0_OUT_HEIGHT,
                            _185_RESHAPE_1_OUTPUT_0_OUT_WIDTH,
                            _185_RESHAPE_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _185_RESHAPE_1_OUTPUT_0_MEM_CONT_SIZE,
                            _185_RESHAPE_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _185_RESHAPE_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _185_RESHAPE_1_OUTPUT_0_DEV_FMT>
                            (_185_Reshape_1_output_0,
                            _185_RESHAPE_1_OUTPUT_0_STREAM);
    fclose(_185_RESHAPE_1_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _186_Conv2D_25_output_0 = (float*) (mem_0 + _186_CONV2D_25_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_184_CONCAT_11_OUTPUT_0_IN_BATCH,
                           _186_CONV2D_25_NB_CHANNELS,
                           _184_CONCAT_11_OUTPUT_0_IN_HEIGHT,
                           _184_CONCAT_11_OUTPUT_0_IN_WIDTH,
                           _186_CONV2D_25_NB_OUTPUTS,
                           _186_CONV2D_25_OUTPUT_0_OUT_HEIGHT,
                           _186_CONV2D_25_OUTPUT_0_OUT_WIDTH,
                           _186_CONV2D_25_PADDING_Y,
                           _186_CONV2D_25_PADDING_X,
                           _186_CONV2D_25_STRIDE_Y,
                           _186_CONV2D_25_STRIDE_X,
                           _186_CONV2D_25_DILATION_Y,
                           _186_CONV2D_25_DILATION_X,
                           _186_CONV2D_25_KERNEL_HEIGHT,
                           _186_CONV2D_25_KERNEL_WIDTH,
                           _186_CONV2D_25_ACTIVATION,
                           _184_CONCAT_11_OUTPUT_0_MEM_CONT_OFFSET,
                           _184_CONCAT_11_OUTPUT_0_MEM_CONT_SIZE,
                           _184_CONCAT_11_OUTPUT_0_MEM_WRAP_OFFSET,
                           _184_CONCAT_11_OUTPUT_0_MEM_WRAP_SIZE,
                           _184_CONCAT_11_OUTPUT_0_MEM_STRIDE,
                           _186_CONV2D_25_OUTPUT_0_MEM_CONT_OFFSET,
                           _186_CONV2D_25_OUTPUT_0_MEM_CONT_SIZE,
                           _186_CONV2D_25_OUTPUT_0_MEM_WRAP_OFFSET,
                           _186_CONV2D_25_OUTPUT_0_MEM_WRAP_SIZE,
                           _186_CONV2D_25_OUTPUT_0_MEM_STRIDE>
                           (_184_Concat_11_output_0, _186_Conv2D_25_output_0, _186_Conv2D_25_weights_output_0, _186_Conv2D_25_biases_output_0, _186_CONV2D_25_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _186_Conv2D_25\n");
    FILE* _186_CONV2D_25_OUTPUT_0_STREAM = fopen("data/export_outputs/_186_Conv2D_25_output_0.txt", "w");
    export_cpp::saveOutputs<_186_CONV2D_25_OUTPUT_0_NB_OUTPUTS,
                            _186_CONV2D_25_OUTPUT_0_OUT_HEIGHT,
                            _186_CONV2D_25_OUTPUT_0_OUT_WIDTH,
                            _186_CONV2D_25_OUTPUT_0_MEM_CONT_OFFSET,
                            _186_CONV2D_25_OUTPUT_0_MEM_CONT_SIZE,
                            _186_CONV2D_25_OUTPUT_0_MEM_WRAP_OFFSET,
                            _186_CONV2D_25_OUTPUT_0_MEM_WRAP_SIZE,
                            _186_CONV2D_25_OUTPUT_0_DEV_FMT>
                            (_186_Conv2D_25_output_0,
                            _186_CONV2D_25_OUTPUT_0_STREAM);
    fclose(_186_CONV2D_25_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _187_Sigmoid_48_output_0 = (float*) (mem_0 + _187_SIGMOID_48_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_187_SIGMOID_48_NB_ELTS>
                               (_186_Conv2D_25_output_0, _187_Sigmoid_48_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _187_Sigmoid_48\n");
    FILE* _187_SIGMOID_48_OUTPUT_0_STREAM = fopen("data/export_outputs/_187_Sigmoid_48_output_0.txt", "w");
    export_cpp::saveOutputs<_187_SIGMOID_48_OUTPUT_0_NB_OUTPUTS,
                            _187_SIGMOID_48_OUTPUT_0_OUT_HEIGHT,
                            _187_SIGMOID_48_OUTPUT_0_OUT_WIDTH,
                            _187_SIGMOID_48_OUTPUT_0_MEM_CONT_OFFSET,
                            _187_SIGMOID_48_OUTPUT_0_MEM_CONT_SIZE,
                            _187_SIGMOID_48_OUTPUT_0_MEM_WRAP_OFFSET,
                            _187_SIGMOID_48_OUTPUT_0_MEM_WRAP_SIZE,
                            _187_SIGMOID_48_OUTPUT_0_DEV_FMT>
                            (_187_Sigmoid_48_output_0,
                            _187_SIGMOID_48_OUTPUT_0_STREAM);
    fclose(_187_SIGMOID_48_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _188_Mul_48_output_0 = (float*) (mem_0 + _188_MUL_48_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_188_MUL_48_NB_MAT,
                                 _188_MUL_48_ELEM_OP,
                                 _188_MUL_48_INPUT1_CONT_SIZE,
                                 _188_MUL_48_INPUT2_CONT_SIZE,
                                 _188_MUL_48_OUTPUT_CONT_SIZE,
                                 _188_MUL_48_OFFSET_IN1,
                                 _188_MUL_48_OFFSET_IN2,
                                 _188_MUL_48_ACTIVATION,
                                 _186_CONV2D_25_OUTPUT_0_MEM_CONT_OFFSET,
                                 _186_CONV2D_25_OUTPUT_0_MEM_CONT_SIZE,
                                 _186_CONV2D_25_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _186_CONV2D_25_OUTPUT_0_MEM_WRAP_SIZE,
                                 _186_CONV2D_25_OUTPUT_0_MEM_STRIDE,
                                 _187_SIGMOID_48_OUTPUT_0_MEM_CONT_OFFSET,
                                 _187_SIGMOID_48_OUTPUT_0_MEM_CONT_SIZE,
                                 _187_SIGMOID_48_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _187_SIGMOID_48_OUTPUT_0_MEM_WRAP_SIZE,
                                 _187_SIGMOID_48_OUTPUT_0_MEM_STRIDE,
                                 _188_MUL_48_OUTPUT_0_MEM_CONT_OFFSET,
                                 _188_MUL_48_OUTPUT_0_MEM_CONT_SIZE,
                                 _188_MUL_48_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _188_MUL_48_OUTPUT_0_MEM_WRAP_SIZE,
                                 _188_MUL_48_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_188_Mul_48_output_0, _188_MUL_48_RESCALING, _186_Conv2D_25_output_0, _187_Sigmoid_48_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _188_Mul_48\n");
    FILE* _188_MUL_48_OUTPUT_0_STREAM = fopen("data/export_outputs/_188_Mul_48_output_0.txt", "w");
    export_cpp::saveOutputs<_188_MUL_48_OUTPUT_0_NB_OUTPUTS,
                            _188_MUL_48_OUTPUT_0_OUT_HEIGHT,
                            _188_MUL_48_OUTPUT_0_OUT_WIDTH,
                            _188_MUL_48_OUTPUT_0_MEM_CONT_OFFSET,
                            _188_MUL_48_OUTPUT_0_MEM_CONT_SIZE,
                            _188_MUL_48_OUTPUT_0_MEM_WRAP_OFFSET,
                            _188_MUL_48_OUTPUT_0_MEM_WRAP_SIZE,
                            _188_MUL_48_OUTPUT_0_DEV_FMT>
                            (_188_Mul_48_output_0,
                            _188_MUL_48_OUTPUT_0_STREAM);
    fclose(_188_MUL_48_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _189_PadConv_24_output_0 = (float*) (mem_0 + _189_PADCONV_24_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_188_MUL_48_OUTPUT_0_IN_BATCH,
                           _189_PADCONV_24_NB_CHANNELS,
                           _188_MUL_48_OUTPUT_0_IN_HEIGHT,
                           _188_MUL_48_OUTPUT_0_IN_WIDTH,
                           _189_PADCONV_24_NB_OUTPUTS,
                           _189_PADCONV_24_OUTPUT_0_OUT_HEIGHT,
                           _189_PADCONV_24_OUTPUT_0_OUT_WIDTH,
                           _189_PADCONV_24_PADDING_Y,
                           _189_PADCONV_24_PADDING_X,
                           _189_PADCONV_24_STRIDE_Y,
                           _189_PADCONV_24_STRIDE_X,
                           _189_PADCONV_24_DILATION_Y,
                           _189_PADCONV_24_DILATION_X,
                           _189_PADCONV_24_KERNEL_HEIGHT,
                           _189_PADCONV_24_KERNEL_WIDTH,
                           _189_PADCONV_24_ACTIVATION,
                           _188_MUL_48_OUTPUT_0_MEM_CONT_OFFSET,
                           _188_MUL_48_OUTPUT_0_MEM_CONT_SIZE,
                           _188_MUL_48_OUTPUT_0_MEM_WRAP_OFFSET,
                           _188_MUL_48_OUTPUT_0_MEM_WRAP_SIZE,
                           _188_MUL_48_OUTPUT_0_MEM_STRIDE,
                           _189_PADCONV_24_OUTPUT_0_MEM_CONT_OFFSET,
                           _189_PADCONV_24_OUTPUT_0_MEM_CONT_SIZE,
                           _189_PADCONV_24_OUTPUT_0_MEM_WRAP_OFFSET,
                           _189_PADCONV_24_OUTPUT_0_MEM_WRAP_SIZE,
                           _189_PADCONV_24_OUTPUT_0_MEM_STRIDE>
                           (_188_Mul_48_output_0, _189_PadConv_24_output_0, _189_PadConv_24_weights_output_0, _189_PadConv_24_biases_output_0, _189_PADCONV_24_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _189_PadConv_24\n");
    FILE* _189_PADCONV_24_OUTPUT_0_STREAM = fopen("data/export_outputs/_189_PadConv_24_output_0.txt", "w");
    export_cpp::saveOutputs<_189_PADCONV_24_OUTPUT_0_NB_OUTPUTS,
                            _189_PADCONV_24_OUTPUT_0_OUT_HEIGHT,
                            _189_PADCONV_24_OUTPUT_0_OUT_WIDTH,
                            _189_PADCONV_24_OUTPUT_0_MEM_CONT_OFFSET,
                            _189_PADCONV_24_OUTPUT_0_MEM_CONT_SIZE,
                            _189_PADCONV_24_OUTPUT_0_MEM_WRAP_OFFSET,
                            _189_PADCONV_24_OUTPUT_0_MEM_WRAP_SIZE,
                            _189_PADCONV_24_OUTPUT_0_DEV_FMT>
                            (_189_PadConv_24_output_0,
                            _189_PADCONV_24_OUTPUT_0_STREAM);
    fclose(_189_PADCONV_24_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _190_Conv2D_26_output_0 = (float*) (mem_0 + _190_CONV2D_26_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_188_MUL_48_OUTPUT_0_IN_BATCH,
                           _190_CONV2D_26_NB_CHANNELS,
                           _188_MUL_48_OUTPUT_0_IN_HEIGHT,
                           _188_MUL_48_OUTPUT_0_IN_WIDTH,
                           _190_CONV2D_26_NB_OUTPUTS,
                           _190_CONV2D_26_OUTPUT_0_OUT_HEIGHT,
                           _190_CONV2D_26_OUTPUT_0_OUT_WIDTH,
                           _190_CONV2D_26_PADDING_Y,
                           _190_CONV2D_26_PADDING_X,
                           _190_CONV2D_26_STRIDE_Y,
                           _190_CONV2D_26_STRIDE_X,
                           _190_CONV2D_26_DILATION_Y,
                           _190_CONV2D_26_DILATION_X,
                           _190_CONV2D_26_KERNEL_HEIGHT,
                           _190_CONV2D_26_KERNEL_WIDTH,
                           _190_CONV2D_26_ACTIVATION,
                           _188_MUL_48_OUTPUT_0_MEM_CONT_OFFSET,
                           _188_MUL_48_OUTPUT_0_MEM_CONT_SIZE,
                           _188_MUL_48_OUTPUT_0_MEM_WRAP_OFFSET,
                           _188_MUL_48_OUTPUT_0_MEM_WRAP_SIZE,
                           _188_MUL_48_OUTPUT_0_MEM_STRIDE,
                           _190_CONV2D_26_OUTPUT_0_MEM_CONT_OFFSET,
                           _190_CONV2D_26_OUTPUT_0_MEM_CONT_SIZE,
                           _190_CONV2D_26_OUTPUT_0_MEM_WRAP_OFFSET,
                           _190_CONV2D_26_OUTPUT_0_MEM_WRAP_SIZE,
                           _190_CONV2D_26_OUTPUT_0_MEM_STRIDE>
                           (_188_Mul_48_output_0, _190_Conv2D_26_output_0, _190_Conv2D_26_weights_output_0, _190_Conv2D_26_biases_output_0, _190_CONV2D_26_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _190_Conv2D_26\n");
    FILE* _190_CONV2D_26_OUTPUT_0_STREAM = fopen("data/export_outputs/_190_Conv2D_26_output_0.txt", "w");
    export_cpp::saveOutputs<_190_CONV2D_26_OUTPUT_0_NB_OUTPUTS,
                            _190_CONV2D_26_OUTPUT_0_OUT_HEIGHT,
                            _190_CONV2D_26_OUTPUT_0_OUT_WIDTH,
                            _190_CONV2D_26_OUTPUT_0_MEM_CONT_OFFSET,
                            _190_CONV2D_26_OUTPUT_0_MEM_CONT_SIZE,
                            _190_CONV2D_26_OUTPUT_0_MEM_WRAP_OFFSET,
                            _190_CONV2D_26_OUTPUT_0_MEM_WRAP_SIZE,
                            _190_CONV2D_26_OUTPUT_0_DEV_FMT>
                            (_190_Conv2D_26_output_0,
                            _190_CONV2D_26_OUTPUT_0_STREAM);
    fclose(_190_CONV2D_26_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _191_PadConvDw_4_output_0 = (float*) (mem_0 + _191_PADCONVDW_4_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_188_MUL_48_OUTPUT_0_IN_BATCH,
                           _191_PADCONVDW_4_NB_CHANNELS,
                           _188_MUL_48_OUTPUT_0_IN_HEIGHT,
                           _188_MUL_48_OUTPUT_0_IN_WIDTH,
                           _191_PADCONVDW_4_NB_OUTPUTS,
                           _191_PADCONVDW_4_OUTPUT_0_OUT_HEIGHT,
                           _191_PADCONVDW_4_OUTPUT_0_OUT_WIDTH,
                           _191_PADCONVDW_4_PADDING_Y,
                           _191_PADCONVDW_4_PADDING_X,
                           _191_PADCONVDW_4_STRIDE_Y,
                           _191_PADCONVDW_4_STRIDE_X,
                           _191_PADCONVDW_4_DILATION_Y,
                           _191_PADCONVDW_4_DILATION_X,
                           _191_PADCONVDW_4_KERNEL_HEIGHT,
                           _191_PADCONVDW_4_KERNEL_WIDTH,
                           _191_PADCONVDW_4_ACTIVATION,
                           _188_MUL_48_OUTPUT_0_MEM_CONT_OFFSET,
                           _188_MUL_48_OUTPUT_0_MEM_CONT_SIZE,
                           _188_MUL_48_OUTPUT_0_MEM_WRAP_OFFSET,
                           _188_MUL_48_OUTPUT_0_MEM_WRAP_SIZE,
                           _188_MUL_48_OUTPUT_0_MEM_STRIDE,
                           _191_PADCONVDW_4_OUTPUT_0_MEM_CONT_OFFSET,
                           _191_PADCONVDW_4_OUTPUT_0_MEM_CONT_SIZE,
                           _191_PADCONVDW_4_OUTPUT_0_MEM_WRAP_OFFSET,
                           _191_PADCONVDW_4_OUTPUT_0_MEM_WRAP_SIZE,
                           _191_PADCONVDW_4_OUTPUT_0_MEM_STRIDE>
                           (_188_Mul_48_output_0, _191_PadConvDw_4_output_0, _191_PadConvDw_4_weights_output_0, _191_PadConvDw_4_biases_output_0, _191_PADCONVDW_4_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _191_PadConvDw_4\n");
    FILE* _191_PADCONVDW_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_191_PadConvDw_4_output_0.txt", "w");
    export_cpp::saveOutputs<_191_PADCONVDW_4_OUTPUT_0_NB_OUTPUTS,
                            _191_PADCONVDW_4_OUTPUT_0_OUT_HEIGHT,
                            _191_PADCONVDW_4_OUTPUT_0_OUT_WIDTH,
                            _191_PADCONVDW_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _191_PADCONVDW_4_OUTPUT_0_MEM_CONT_SIZE,
                            _191_PADCONVDW_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _191_PADCONVDW_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _191_PADCONVDW_4_OUTPUT_0_DEV_FMT>
                            (_191_PadConvDw_4_output_0,
                            _191_PADCONVDW_4_OUTPUT_0_STREAM);
    fclose(_191_PADCONVDW_4_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _192_Sigmoid_49_output_0 = (float*) (mem_0 + _192_SIGMOID_49_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_192_SIGMOID_49_NB_ELTS>
                               (_190_Conv2D_26_output_0, _192_Sigmoid_49_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _192_Sigmoid_49\n");
    FILE* _192_SIGMOID_49_OUTPUT_0_STREAM = fopen("data/export_outputs/_192_Sigmoid_49_output_0.txt", "w");
    export_cpp::saveOutputs<_192_SIGMOID_49_OUTPUT_0_NB_OUTPUTS,
                            _192_SIGMOID_49_OUTPUT_0_OUT_HEIGHT,
                            _192_SIGMOID_49_OUTPUT_0_OUT_WIDTH,
                            _192_SIGMOID_49_OUTPUT_0_MEM_CONT_OFFSET,
                            _192_SIGMOID_49_OUTPUT_0_MEM_CONT_SIZE,
                            _192_SIGMOID_49_OUTPUT_0_MEM_WRAP_OFFSET,
                            _192_SIGMOID_49_OUTPUT_0_MEM_WRAP_SIZE,
                            _192_SIGMOID_49_OUTPUT_0_DEV_FMT>
                            (_192_Sigmoid_49_output_0,
                            _192_SIGMOID_49_OUTPUT_0_STREAM);
    fclose(_192_SIGMOID_49_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _193_Sigmoid_50_output_0 = (float*) (mem_0 + _193_SIGMOID_50_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_193_SIGMOID_50_NB_ELTS>
                               (_189_PadConv_24_output_0, _193_Sigmoid_50_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _193_Sigmoid_50\n");
    FILE* _193_SIGMOID_50_OUTPUT_0_STREAM = fopen("data/export_outputs/_193_Sigmoid_50_output_0.txt", "w");
    export_cpp::saveOutputs<_193_SIGMOID_50_OUTPUT_0_NB_OUTPUTS,
                            _193_SIGMOID_50_OUTPUT_0_OUT_HEIGHT,
                            _193_SIGMOID_50_OUTPUT_0_OUT_WIDTH,
                            _193_SIGMOID_50_OUTPUT_0_MEM_CONT_OFFSET,
                            _193_SIGMOID_50_OUTPUT_0_MEM_CONT_SIZE,
                            _193_SIGMOID_50_OUTPUT_0_MEM_WRAP_OFFSET,
                            _193_SIGMOID_50_OUTPUT_0_MEM_WRAP_SIZE,
                            _193_SIGMOID_50_OUTPUT_0_DEV_FMT>
                            (_193_Sigmoid_50_output_0,
                            _193_SIGMOID_50_OUTPUT_0_STREAM);
    fclose(_193_SIGMOID_50_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _194_Sigmoid_51_output_0 = (float*) (mem_0 + _194_SIGMOID_51_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_194_SIGMOID_51_NB_ELTS>
                               (_191_PadConvDw_4_output_0, _194_Sigmoid_51_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _194_Sigmoid_51\n");
    FILE* _194_SIGMOID_51_OUTPUT_0_STREAM = fopen("data/export_outputs/_194_Sigmoid_51_output_0.txt", "w");
    export_cpp::saveOutputs<_194_SIGMOID_51_OUTPUT_0_NB_OUTPUTS,
                            _194_SIGMOID_51_OUTPUT_0_OUT_HEIGHT,
                            _194_SIGMOID_51_OUTPUT_0_OUT_WIDTH,
                            _194_SIGMOID_51_OUTPUT_0_MEM_CONT_OFFSET,
                            _194_SIGMOID_51_OUTPUT_0_MEM_CONT_SIZE,
                            _194_SIGMOID_51_OUTPUT_0_MEM_WRAP_OFFSET,
                            _194_SIGMOID_51_OUTPUT_0_MEM_WRAP_SIZE,
                            _194_SIGMOID_51_OUTPUT_0_DEV_FMT>
                            (_194_Sigmoid_51_output_0,
                            _194_SIGMOID_51_OUTPUT_0_STREAM);
    fclose(_194_SIGMOID_51_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _195_Mul_49_output_0 = (float*) (mem_0 + _195_MUL_49_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_195_MUL_49_NB_MAT,
                                 _195_MUL_49_ELEM_OP,
                                 _195_MUL_49_INPUT1_CONT_SIZE,
                                 _195_MUL_49_INPUT2_CONT_SIZE,
                                 _195_MUL_49_OUTPUT_CONT_SIZE,
                                 _195_MUL_49_OFFSET_IN1,
                                 _195_MUL_49_OFFSET_IN2,
                                 _195_MUL_49_ACTIVATION,
                                 _190_CONV2D_26_OUTPUT_0_MEM_CONT_OFFSET,
                                 _190_CONV2D_26_OUTPUT_0_MEM_CONT_SIZE,
                                 _190_CONV2D_26_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _190_CONV2D_26_OUTPUT_0_MEM_WRAP_SIZE,
                                 _190_CONV2D_26_OUTPUT_0_MEM_STRIDE,
                                 _192_SIGMOID_49_OUTPUT_0_MEM_CONT_OFFSET,
                                 _192_SIGMOID_49_OUTPUT_0_MEM_CONT_SIZE,
                                 _192_SIGMOID_49_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _192_SIGMOID_49_OUTPUT_0_MEM_WRAP_SIZE,
                                 _192_SIGMOID_49_OUTPUT_0_MEM_STRIDE,
                                 _195_MUL_49_OUTPUT_0_MEM_CONT_OFFSET,
                                 _195_MUL_49_OUTPUT_0_MEM_CONT_SIZE,
                                 _195_MUL_49_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _195_MUL_49_OUTPUT_0_MEM_WRAP_SIZE,
                                 _195_MUL_49_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_195_Mul_49_output_0, _195_MUL_49_RESCALING, _190_Conv2D_26_output_0, _192_Sigmoid_49_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _195_Mul_49\n");
    FILE* _195_MUL_49_OUTPUT_0_STREAM = fopen("data/export_outputs/_195_Mul_49_output_0.txt", "w");
    export_cpp::saveOutputs<_195_MUL_49_OUTPUT_0_NB_OUTPUTS,
                            _195_MUL_49_OUTPUT_0_OUT_HEIGHT,
                            _195_MUL_49_OUTPUT_0_OUT_WIDTH,
                            _195_MUL_49_OUTPUT_0_MEM_CONT_OFFSET,
                            _195_MUL_49_OUTPUT_0_MEM_CONT_SIZE,
                            _195_MUL_49_OUTPUT_0_MEM_WRAP_OFFSET,
                            _195_MUL_49_OUTPUT_0_MEM_WRAP_SIZE,
                            _195_MUL_49_OUTPUT_0_DEV_FMT>
                            (_195_Mul_49_output_0,
                            _195_MUL_49_OUTPUT_0_STREAM);
    fclose(_195_MUL_49_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _196_Mul_50_output_0 = (float*) (mem_0 + _196_MUL_50_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_196_MUL_50_NB_MAT,
                                 _196_MUL_50_ELEM_OP,
                                 _196_MUL_50_INPUT1_CONT_SIZE,
                                 _196_MUL_50_INPUT2_CONT_SIZE,
                                 _196_MUL_50_OUTPUT_CONT_SIZE,
                                 _196_MUL_50_OFFSET_IN1,
                                 _196_MUL_50_OFFSET_IN2,
                                 _196_MUL_50_ACTIVATION,
                                 _189_PADCONV_24_OUTPUT_0_MEM_CONT_OFFSET,
                                 _189_PADCONV_24_OUTPUT_0_MEM_CONT_SIZE,
                                 _189_PADCONV_24_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _189_PADCONV_24_OUTPUT_0_MEM_WRAP_SIZE,
                                 _189_PADCONV_24_OUTPUT_0_MEM_STRIDE,
                                 _193_SIGMOID_50_OUTPUT_0_MEM_CONT_OFFSET,
                                 _193_SIGMOID_50_OUTPUT_0_MEM_CONT_SIZE,
                                 _193_SIGMOID_50_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _193_SIGMOID_50_OUTPUT_0_MEM_WRAP_SIZE,
                                 _193_SIGMOID_50_OUTPUT_0_MEM_STRIDE,
                                 _196_MUL_50_OUTPUT_0_MEM_CONT_OFFSET,
                                 _196_MUL_50_OUTPUT_0_MEM_CONT_SIZE,
                                 _196_MUL_50_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _196_MUL_50_OUTPUT_0_MEM_WRAP_SIZE,
                                 _196_MUL_50_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_196_Mul_50_output_0, _196_MUL_50_RESCALING, _189_PadConv_24_output_0, _193_Sigmoid_50_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _196_Mul_50\n");
    FILE* _196_MUL_50_OUTPUT_0_STREAM = fopen("data/export_outputs/_196_Mul_50_output_0.txt", "w");
    export_cpp::saveOutputs<_196_MUL_50_OUTPUT_0_NB_OUTPUTS,
                            _196_MUL_50_OUTPUT_0_OUT_HEIGHT,
                            _196_MUL_50_OUTPUT_0_OUT_WIDTH,
                            _196_MUL_50_OUTPUT_0_MEM_CONT_OFFSET,
                            _196_MUL_50_OUTPUT_0_MEM_CONT_SIZE,
                            _196_MUL_50_OUTPUT_0_MEM_WRAP_OFFSET,
                            _196_MUL_50_OUTPUT_0_MEM_WRAP_SIZE,
                            _196_MUL_50_OUTPUT_0_DEV_FMT>
                            (_196_Mul_50_output_0,
                            _196_MUL_50_OUTPUT_0_STREAM);
    fclose(_196_MUL_50_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _197_Mul_51_output_0 = (float*) (mem_0 + _197_MUL_51_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_197_MUL_51_NB_MAT,
                                 _197_MUL_51_ELEM_OP,
                                 _197_MUL_51_INPUT1_CONT_SIZE,
                                 _197_MUL_51_INPUT2_CONT_SIZE,
                                 _197_MUL_51_OUTPUT_CONT_SIZE,
                                 _197_MUL_51_OFFSET_IN1,
                                 _197_MUL_51_OFFSET_IN2,
                                 _197_MUL_51_ACTIVATION,
                                 _191_PADCONVDW_4_OUTPUT_0_MEM_CONT_OFFSET,
                                 _191_PADCONVDW_4_OUTPUT_0_MEM_CONT_SIZE,
                                 _191_PADCONVDW_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _191_PADCONVDW_4_OUTPUT_0_MEM_WRAP_SIZE,
                                 _191_PADCONVDW_4_OUTPUT_0_MEM_STRIDE,
                                 _194_SIGMOID_51_OUTPUT_0_MEM_CONT_OFFSET,
                                 _194_SIGMOID_51_OUTPUT_0_MEM_CONT_SIZE,
                                 _194_SIGMOID_51_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _194_SIGMOID_51_OUTPUT_0_MEM_WRAP_SIZE,
                                 _194_SIGMOID_51_OUTPUT_0_MEM_STRIDE,
                                 _197_MUL_51_OUTPUT_0_MEM_CONT_OFFSET,
                                 _197_MUL_51_OUTPUT_0_MEM_CONT_SIZE,
                                 _197_MUL_51_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _197_MUL_51_OUTPUT_0_MEM_WRAP_SIZE,
                                 _197_MUL_51_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_197_Mul_51_output_0, _197_MUL_51_RESCALING, _191_PadConvDw_4_output_0, _194_Sigmoid_51_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _197_Mul_51\n");
    FILE* _197_MUL_51_OUTPUT_0_STREAM = fopen("data/export_outputs/_197_Mul_51_output_0.txt", "w");
    export_cpp::saveOutputs<_197_MUL_51_OUTPUT_0_NB_OUTPUTS,
                            _197_MUL_51_OUTPUT_0_OUT_HEIGHT,
                            _197_MUL_51_OUTPUT_0_OUT_WIDTH,
                            _197_MUL_51_OUTPUT_0_MEM_CONT_OFFSET,
                            _197_MUL_51_OUTPUT_0_MEM_CONT_SIZE,
                            _197_MUL_51_OUTPUT_0_MEM_WRAP_OFFSET,
                            _197_MUL_51_OUTPUT_0_MEM_WRAP_SIZE,
                            _197_MUL_51_OUTPUT_0_DEV_FMT>
                            (_197_Mul_51_output_0,
                            _197_MUL_51_OUTPUT_0_STREAM);
    fclose(_197_MUL_51_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _198_PadConvDw_5_output_0 = (float*) (mem_0 + _198_PADCONVDW_5_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_195_MUL_49_OUTPUT_0_IN_BATCH,
                           _198_PADCONVDW_5_NB_CHANNELS,
                           _195_MUL_49_OUTPUT_0_IN_HEIGHT,
                           _195_MUL_49_OUTPUT_0_IN_WIDTH,
                           _198_PADCONVDW_5_NB_OUTPUTS,
                           _198_PADCONVDW_5_OUTPUT_0_OUT_HEIGHT,
                           _198_PADCONVDW_5_OUTPUT_0_OUT_WIDTH,
                           _198_PADCONVDW_5_PADDING_Y,
                           _198_PADCONVDW_5_PADDING_X,
                           _198_PADCONVDW_5_STRIDE_Y,
                           _198_PADCONVDW_5_STRIDE_X,
                           _198_PADCONVDW_5_DILATION_Y,
                           _198_PADCONVDW_5_DILATION_X,
                           _198_PADCONVDW_5_KERNEL_HEIGHT,
                           _198_PADCONVDW_5_KERNEL_WIDTH,
                           _198_PADCONVDW_5_ACTIVATION,
                           _195_MUL_49_OUTPUT_0_MEM_CONT_OFFSET,
                           _195_MUL_49_OUTPUT_0_MEM_CONT_SIZE,
                           _195_MUL_49_OUTPUT_0_MEM_WRAP_OFFSET,
                           _195_MUL_49_OUTPUT_0_MEM_WRAP_SIZE,
                           _195_MUL_49_OUTPUT_0_MEM_STRIDE,
                           _198_PADCONVDW_5_OUTPUT_0_MEM_CONT_OFFSET,
                           _198_PADCONVDW_5_OUTPUT_0_MEM_CONT_SIZE,
                           _198_PADCONVDW_5_OUTPUT_0_MEM_WRAP_OFFSET,
                           _198_PADCONVDW_5_OUTPUT_0_MEM_WRAP_SIZE,
                           _198_PADCONVDW_5_OUTPUT_0_MEM_STRIDE>
                           (_195_Mul_49_output_0, _198_PadConvDw_5_output_0, _198_PadConvDw_5_weights_output_0, _198_PadConvDw_5_biases_output_0, _198_PADCONVDW_5_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _198_PadConvDw_5\n");
    FILE* _198_PADCONVDW_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_198_PadConvDw_5_output_0.txt", "w");
    export_cpp::saveOutputs<_198_PADCONVDW_5_OUTPUT_0_NB_OUTPUTS,
                            _198_PADCONVDW_5_OUTPUT_0_OUT_HEIGHT,
                            _198_PADCONVDW_5_OUTPUT_0_OUT_WIDTH,
                            _198_PADCONVDW_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _198_PADCONVDW_5_OUTPUT_0_MEM_CONT_SIZE,
                            _198_PADCONVDW_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _198_PADCONVDW_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _198_PADCONVDW_5_OUTPUT_0_DEV_FMT>
                            (_198_PadConvDw_5_output_0,
                            _198_PADCONVDW_5_OUTPUT_0_STREAM);
    fclose(_198_PADCONVDW_5_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _199_PadConv_25_output_0 = (float*) (mem_0 + _199_PADCONV_25_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_196_MUL_50_OUTPUT_0_IN_BATCH,
                           _199_PADCONV_25_NB_CHANNELS,
                           _196_MUL_50_OUTPUT_0_IN_HEIGHT,
                           _196_MUL_50_OUTPUT_0_IN_WIDTH,
                           _199_PADCONV_25_NB_OUTPUTS,
                           _199_PADCONV_25_OUTPUT_0_OUT_HEIGHT,
                           _199_PADCONV_25_OUTPUT_0_OUT_WIDTH,
                           _199_PADCONV_25_PADDING_Y,
                           _199_PADCONV_25_PADDING_X,
                           _199_PADCONV_25_STRIDE_Y,
                           _199_PADCONV_25_STRIDE_X,
                           _199_PADCONV_25_DILATION_Y,
                           _199_PADCONV_25_DILATION_X,
                           _199_PADCONV_25_KERNEL_HEIGHT,
                           _199_PADCONV_25_KERNEL_WIDTH,
                           _199_PADCONV_25_ACTIVATION,
                           _196_MUL_50_OUTPUT_0_MEM_CONT_OFFSET,
                           _196_MUL_50_OUTPUT_0_MEM_CONT_SIZE,
                           _196_MUL_50_OUTPUT_0_MEM_WRAP_OFFSET,
                           _196_MUL_50_OUTPUT_0_MEM_WRAP_SIZE,
                           _196_MUL_50_OUTPUT_0_MEM_STRIDE,
                           _199_PADCONV_25_OUTPUT_0_MEM_CONT_OFFSET,
                           _199_PADCONV_25_OUTPUT_0_MEM_CONT_SIZE,
                           _199_PADCONV_25_OUTPUT_0_MEM_WRAP_OFFSET,
                           _199_PADCONV_25_OUTPUT_0_MEM_WRAP_SIZE,
                           _199_PADCONV_25_OUTPUT_0_MEM_STRIDE>
                           (_196_Mul_50_output_0, _199_PadConv_25_output_0, _199_PadConv_25_weights_output_0, _199_PadConv_25_biases_output_0, _199_PADCONV_25_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _199_PadConv_25\n");
    FILE* _199_PADCONV_25_OUTPUT_0_STREAM = fopen("data/export_outputs/_199_PadConv_25_output_0.txt", "w");
    export_cpp::saveOutputs<_199_PADCONV_25_OUTPUT_0_NB_OUTPUTS,
                            _199_PADCONV_25_OUTPUT_0_OUT_HEIGHT,
                            _199_PADCONV_25_OUTPUT_0_OUT_WIDTH,
                            _199_PADCONV_25_OUTPUT_0_MEM_CONT_OFFSET,
                            _199_PADCONV_25_OUTPUT_0_MEM_CONT_SIZE,
                            _199_PADCONV_25_OUTPUT_0_MEM_WRAP_OFFSET,
                            _199_PADCONV_25_OUTPUT_0_MEM_WRAP_SIZE,
                            _199_PADCONV_25_OUTPUT_0_DEV_FMT>
                            (_199_PadConv_25_output_0,
                            _199_PADCONV_25_OUTPUT_0_STREAM);
    fclose(_199_PADCONV_25_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _200_Conv2D_27_output_0 = (float*) (mem_0 + _200_CONV2D_27_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_197_MUL_51_OUTPUT_0_IN_BATCH,
                           _200_CONV2D_27_NB_CHANNELS,
                           _197_MUL_51_OUTPUT_0_IN_HEIGHT,
                           _197_MUL_51_OUTPUT_0_IN_WIDTH,
                           _200_CONV2D_27_NB_OUTPUTS,
                           _200_CONV2D_27_OUTPUT_0_OUT_HEIGHT,
                           _200_CONV2D_27_OUTPUT_0_OUT_WIDTH,
                           _200_CONV2D_27_PADDING_Y,
                           _200_CONV2D_27_PADDING_X,
                           _200_CONV2D_27_STRIDE_Y,
                           _200_CONV2D_27_STRIDE_X,
                           _200_CONV2D_27_DILATION_Y,
                           _200_CONV2D_27_DILATION_X,
                           _200_CONV2D_27_KERNEL_HEIGHT,
                           _200_CONV2D_27_KERNEL_WIDTH,
                           _200_CONV2D_27_ACTIVATION,
                           _197_MUL_51_OUTPUT_0_MEM_CONT_OFFSET,
                           _197_MUL_51_OUTPUT_0_MEM_CONT_SIZE,
                           _197_MUL_51_OUTPUT_0_MEM_WRAP_OFFSET,
                           _197_MUL_51_OUTPUT_0_MEM_WRAP_SIZE,
                           _197_MUL_51_OUTPUT_0_MEM_STRIDE,
                           _200_CONV2D_27_OUTPUT_0_MEM_CONT_OFFSET,
                           _200_CONV2D_27_OUTPUT_0_MEM_CONT_SIZE,
                           _200_CONV2D_27_OUTPUT_0_MEM_WRAP_OFFSET,
                           _200_CONV2D_27_OUTPUT_0_MEM_WRAP_SIZE,
                           _200_CONV2D_27_OUTPUT_0_MEM_STRIDE>
                           (_197_Mul_51_output_0, _200_Conv2D_27_output_0, _200_Conv2D_27_weights_output_0, _200_Conv2D_27_biases_output_0, _200_CONV2D_27_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _200_Conv2D_27\n");
    FILE* _200_CONV2D_27_OUTPUT_0_STREAM = fopen("data/export_outputs/_200_Conv2D_27_output_0.txt", "w");
    export_cpp::saveOutputs<_200_CONV2D_27_OUTPUT_0_NB_OUTPUTS,
                            _200_CONV2D_27_OUTPUT_0_OUT_HEIGHT,
                            _200_CONV2D_27_OUTPUT_0_OUT_WIDTH,
                            _200_CONV2D_27_OUTPUT_0_MEM_CONT_OFFSET,
                            _200_CONV2D_27_OUTPUT_0_MEM_CONT_SIZE,
                            _200_CONV2D_27_OUTPUT_0_MEM_WRAP_OFFSET,
                            _200_CONV2D_27_OUTPUT_0_MEM_WRAP_SIZE,
                            _200_CONV2D_27_OUTPUT_0_DEV_FMT>
                            (_200_Conv2D_27_output_0,
                            _200_CONV2D_27_OUTPUT_0_STREAM);
    fclose(_200_CONV2D_27_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _201_Concat_12_output_0 = (float*) (mem_0 + _201_CONCAT_12_OUTPUT_0_MEM_OFFSET);

    const float* _201_Concat_12_inputs[] = { _198_PadConvDw_5_output_0, _114_Mul_29_output_0 };
    export_cpp::concat_forward<_201_CONCAT_12_AXIS_SIZE_POST,
                               _201_CONCAT_12_AXIS_SIZE_PRE,
                               _201_CONCAT_12_AXIS_SIZE,
                               _201_CONCAT_12_TOTAL_AXIS_SIZE,
                               2> 
                              (_201_Concat_12_inputs,
                               _201_Concat_12_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _201_Concat_12\n");
    FILE* _201_CONCAT_12_OUTPUT_0_STREAM = fopen("data/export_outputs/_201_Concat_12_output_0.txt", "w");
    export_cpp::saveOutputs<_201_CONCAT_12_OUTPUT_0_NB_OUTPUTS,
                            _201_CONCAT_12_OUTPUT_0_OUT_HEIGHT,
                            _201_CONCAT_12_OUTPUT_0_OUT_WIDTH,
                            _201_CONCAT_12_OUTPUT_0_MEM_CONT_OFFSET,
                            _201_CONCAT_12_OUTPUT_0_MEM_CONT_SIZE,
                            _201_CONCAT_12_OUTPUT_0_MEM_WRAP_OFFSET,
                            _201_CONCAT_12_OUTPUT_0_MEM_WRAP_SIZE,
                            _201_CONCAT_12_OUTPUT_0_DEV_FMT>
                            (_201_Concat_12_output_0,
                            _201_CONCAT_12_OUTPUT_0_STREAM);
    fclose(_201_CONCAT_12_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _202_Sigmoid_52_output_0 = (float*) (mem_0 + _202_SIGMOID_52_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_202_SIGMOID_52_NB_ELTS>
                               (_199_PadConv_25_output_0, _202_Sigmoid_52_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _202_Sigmoid_52\n");
    FILE* _202_SIGMOID_52_OUTPUT_0_STREAM = fopen("data/export_outputs/_202_Sigmoid_52_output_0.txt", "w");
    export_cpp::saveOutputs<_202_SIGMOID_52_OUTPUT_0_NB_OUTPUTS,
                            _202_SIGMOID_52_OUTPUT_0_OUT_HEIGHT,
                            _202_SIGMOID_52_OUTPUT_0_OUT_WIDTH,
                            _202_SIGMOID_52_OUTPUT_0_MEM_CONT_OFFSET,
                            _202_SIGMOID_52_OUTPUT_0_MEM_CONT_SIZE,
                            _202_SIGMOID_52_OUTPUT_0_MEM_WRAP_OFFSET,
                            _202_SIGMOID_52_OUTPUT_0_MEM_WRAP_SIZE,
                            _202_SIGMOID_52_OUTPUT_0_DEV_FMT>
                            (_202_Sigmoid_52_output_0,
                            _202_SIGMOID_52_OUTPUT_0_STREAM);
    fclose(_202_SIGMOID_52_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _203_Sigmoid_53_output_0 = (float*) (mem_0 + _203_SIGMOID_53_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_203_SIGMOID_53_NB_ELTS>
                               (_200_Conv2D_27_output_0, _203_Sigmoid_53_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _203_Sigmoid_53\n");
    FILE* _203_SIGMOID_53_OUTPUT_0_STREAM = fopen("data/export_outputs/_203_Sigmoid_53_output_0.txt", "w");
    export_cpp::saveOutputs<_203_SIGMOID_53_OUTPUT_0_NB_OUTPUTS,
                            _203_SIGMOID_53_OUTPUT_0_OUT_HEIGHT,
                            _203_SIGMOID_53_OUTPUT_0_OUT_WIDTH,
                            _203_SIGMOID_53_OUTPUT_0_MEM_CONT_OFFSET,
                            _203_SIGMOID_53_OUTPUT_0_MEM_CONT_SIZE,
                            _203_SIGMOID_53_OUTPUT_0_MEM_WRAP_OFFSET,
                            _203_SIGMOID_53_OUTPUT_0_MEM_WRAP_SIZE,
                            _203_SIGMOID_53_OUTPUT_0_DEV_FMT>
                            (_203_Sigmoid_53_output_0,
                            _203_SIGMOID_53_OUTPUT_0_STREAM);
    fclose(_203_SIGMOID_53_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _204_Mul_52_output_0 = (float*) (mem_0 + _204_MUL_52_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_204_MUL_52_NB_MAT,
                                 _204_MUL_52_ELEM_OP,
                                 _204_MUL_52_INPUT1_CONT_SIZE,
                                 _204_MUL_52_INPUT2_CONT_SIZE,
                                 _204_MUL_52_OUTPUT_CONT_SIZE,
                                 _204_MUL_52_OFFSET_IN1,
                                 _204_MUL_52_OFFSET_IN2,
                                 _204_MUL_52_ACTIVATION,
                                 _199_PADCONV_25_OUTPUT_0_MEM_CONT_OFFSET,
                                 _199_PADCONV_25_OUTPUT_0_MEM_CONT_SIZE,
                                 _199_PADCONV_25_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _199_PADCONV_25_OUTPUT_0_MEM_WRAP_SIZE,
                                 _199_PADCONV_25_OUTPUT_0_MEM_STRIDE,
                                 _202_SIGMOID_52_OUTPUT_0_MEM_CONT_OFFSET,
                                 _202_SIGMOID_52_OUTPUT_0_MEM_CONT_SIZE,
                                 _202_SIGMOID_52_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _202_SIGMOID_52_OUTPUT_0_MEM_WRAP_SIZE,
                                 _202_SIGMOID_52_OUTPUT_0_MEM_STRIDE,
                                 _204_MUL_52_OUTPUT_0_MEM_CONT_OFFSET,
                                 _204_MUL_52_OUTPUT_0_MEM_CONT_SIZE,
                                 _204_MUL_52_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _204_MUL_52_OUTPUT_0_MEM_WRAP_SIZE,
                                 _204_MUL_52_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_204_Mul_52_output_0, _204_MUL_52_RESCALING, _199_PadConv_25_output_0, _202_Sigmoid_52_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _204_Mul_52\n");
    FILE* _204_MUL_52_OUTPUT_0_STREAM = fopen("data/export_outputs/_204_Mul_52_output_0.txt", "w");
    export_cpp::saveOutputs<_204_MUL_52_OUTPUT_0_NB_OUTPUTS,
                            _204_MUL_52_OUTPUT_0_OUT_HEIGHT,
                            _204_MUL_52_OUTPUT_0_OUT_WIDTH,
                            _204_MUL_52_OUTPUT_0_MEM_CONT_OFFSET,
                            _204_MUL_52_OUTPUT_0_MEM_CONT_SIZE,
                            _204_MUL_52_OUTPUT_0_MEM_WRAP_OFFSET,
                            _204_MUL_52_OUTPUT_0_MEM_WRAP_SIZE,
                            _204_MUL_52_OUTPUT_0_DEV_FMT>
                            (_204_Mul_52_output_0,
                            _204_MUL_52_OUTPUT_0_STREAM);
    fclose(_204_MUL_52_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _205_Mul_53_output_0 = (float*) (mem_0 + _205_MUL_53_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_205_MUL_53_NB_MAT,
                                 _205_MUL_53_ELEM_OP,
                                 _205_MUL_53_INPUT1_CONT_SIZE,
                                 _205_MUL_53_INPUT2_CONT_SIZE,
                                 _205_MUL_53_OUTPUT_CONT_SIZE,
                                 _205_MUL_53_OFFSET_IN1,
                                 _205_MUL_53_OFFSET_IN2,
                                 _205_MUL_53_ACTIVATION,
                                 _200_CONV2D_27_OUTPUT_0_MEM_CONT_OFFSET,
                                 _200_CONV2D_27_OUTPUT_0_MEM_CONT_SIZE,
                                 _200_CONV2D_27_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _200_CONV2D_27_OUTPUT_0_MEM_WRAP_SIZE,
                                 _200_CONV2D_27_OUTPUT_0_MEM_STRIDE,
                                 _203_SIGMOID_53_OUTPUT_0_MEM_CONT_OFFSET,
                                 _203_SIGMOID_53_OUTPUT_0_MEM_CONT_SIZE,
                                 _203_SIGMOID_53_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _203_SIGMOID_53_OUTPUT_0_MEM_WRAP_SIZE,
                                 _203_SIGMOID_53_OUTPUT_0_MEM_STRIDE,
                                 _205_MUL_53_OUTPUT_0_MEM_CONT_OFFSET,
                                 _205_MUL_53_OUTPUT_0_MEM_CONT_SIZE,
                                 _205_MUL_53_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _205_MUL_53_OUTPUT_0_MEM_WRAP_SIZE,
                                 _205_MUL_53_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_205_Mul_53_output_0, _205_MUL_53_RESCALING, _200_Conv2D_27_output_0, _203_Sigmoid_53_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _205_Mul_53\n");
    FILE* _205_MUL_53_OUTPUT_0_STREAM = fopen("data/export_outputs/_205_Mul_53_output_0.txt", "w");
    export_cpp::saveOutputs<_205_MUL_53_OUTPUT_0_NB_OUTPUTS,
                            _205_MUL_53_OUTPUT_0_OUT_HEIGHT,
                            _205_MUL_53_OUTPUT_0_OUT_WIDTH,
                            _205_MUL_53_OUTPUT_0_MEM_CONT_OFFSET,
                            _205_MUL_53_OUTPUT_0_MEM_CONT_SIZE,
                            _205_MUL_53_OUTPUT_0_MEM_WRAP_OFFSET,
                            _205_MUL_53_OUTPUT_0_MEM_WRAP_SIZE,
                            _205_MUL_53_OUTPUT_0_DEV_FMT>
                            (_205_Mul_53_output_0,
                            _205_MUL_53_OUTPUT_0_STREAM);
    fclose(_205_MUL_53_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _206_Conv2D_28_output_0 = (float*) (mem_0 + _206_CONV2D_28_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_201_CONCAT_12_OUTPUT_0_IN_BATCH,
                           _206_CONV2D_28_NB_CHANNELS,
                           _201_CONCAT_12_OUTPUT_0_IN_HEIGHT,
                           _201_CONCAT_12_OUTPUT_0_IN_WIDTH,
                           _206_CONV2D_28_NB_OUTPUTS,
                           _206_CONV2D_28_OUTPUT_0_OUT_HEIGHT,
                           _206_CONV2D_28_OUTPUT_0_OUT_WIDTH,
                           _206_CONV2D_28_PADDING_Y,
                           _206_CONV2D_28_PADDING_X,
                           _206_CONV2D_28_STRIDE_Y,
                           _206_CONV2D_28_STRIDE_X,
                           _206_CONV2D_28_DILATION_Y,
                           _206_CONV2D_28_DILATION_X,
                           _206_CONV2D_28_KERNEL_HEIGHT,
                           _206_CONV2D_28_KERNEL_WIDTH,
                           _206_CONV2D_28_ACTIVATION,
                           _201_CONCAT_12_OUTPUT_0_MEM_CONT_OFFSET,
                           _201_CONCAT_12_OUTPUT_0_MEM_CONT_SIZE,
                           _201_CONCAT_12_OUTPUT_0_MEM_WRAP_OFFSET,
                           _201_CONCAT_12_OUTPUT_0_MEM_WRAP_SIZE,
                           _201_CONCAT_12_OUTPUT_0_MEM_STRIDE,
                           _206_CONV2D_28_OUTPUT_0_MEM_CONT_OFFSET,
                           _206_CONV2D_28_OUTPUT_0_MEM_CONT_SIZE,
                           _206_CONV2D_28_OUTPUT_0_MEM_WRAP_OFFSET,
                           _206_CONV2D_28_OUTPUT_0_MEM_WRAP_SIZE,
                           _206_CONV2D_28_OUTPUT_0_MEM_STRIDE>
                           (_201_Concat_12_output_0, _206_Conv2D_28_output_0, _206_Conv2D_28_weights_output_0, _206_Conv2D_28_biases_output_0, _206_CONV2D_28_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _206_Conv2D_28\n");
    FILE* _206_CONV2D_28_OUTPUT_0_STREAM = fopen("data/export_outputs/_206_Conv2D_28_output_0.txt", "w");
    export_cpp::saveOutputs<_206_CONV2D_28_OUTPUT_0_NB_OUTPUTS,
                            _206_CONV2D_28_OUTPUT_0_OUT_HEIGHT,
                            _206_CONV2D_28_OUTPUT_0_OUT_WIDTH,
                            _206_CONV2D_28_OUTPUT_0_MEM_CONT_OFFSET,
                            _206_CONV2D_28_OUTPUT_0_MEM_CONT_SIZE,
                            _206_CONV2D_28_OUTPUT_0_MEM_WRAP_OFFSET,
                            _206_CONV2D_28_OUTPUT_0_MEM_WRAP_SIZE,
                            _206_CONV2D_28_OUTPUT_0_DEV_FMT>
                            (_206_Conv2D_28_output_0,
                            _206_CONV2D_28_OUTPUT_0_STREAM);
    fclose(_206_CONV2D_28_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _207_Sigmoid_54_output_0 = (float*) (mem_0 + _207_SIGMOID_54_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_207_SIGMOID_54_NB_ELTS>
                               (_206_Conv2D_28_output_0, _207_Sigmoid_54_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _207_Sigmoid_54\n");
    FILE* _207_SIGMOID_54_OUTPUT_0_STREAM = fopen("data/export_outputs/_207_Sigmoid_54_output_0.txt", "w");
    export_cpp::saveOutputs<_207_SIGMOID_54_OUTPUT_0_NB_OUTPUTS,
                            _207_SIGMOID_54_OUTPUT_0_OUT_HEIGHT,
                            _207_SIGMOID_54_OUTPUT_0_OUT_WIDTH,
                            _207_SIGMOID_54_OUTPUT_0_MEM_CONT_OFFSET,
                            _207_SIGMOID_54_OUTPUT_0_MEM_CONT_SIZE,
                            _207_SIGMOID_54_OUTPUT_0_MEM_WRAP_OFFSET,
                            _207_SIGMOID_54_OUTPUT_0_MEM_WRAP_SIZE,
                            _207_SIGMOID_54_OUTPUT_0_DEV_FMT>
                            (_207_Sigmoid_54_output_0,
                            _207_SIGMOID_54_OUTPUT_0_STREAM);
    fclose(_207_SIGMOID_54_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _208_Conv2D_29_output_0 = (float*) (mem_0 + _208_CONV2D_29_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_204_MUL_52_OUTPUT_0_IN_BATCH,
                           _208_CONV2D_29_NB_CHANNELS,
                           _204_MUL_52_OUTPUT_0_IN_HEIGHT,
                           _204_MUL_52_OUTPUT_0_IN_WIDTH,
                           _208_CONV2D_29_NB_OUTPUTS,
                           _208_CONV2D_29_OUTPUT_0_OUT_HEIGHT,
                           _208_CONV2D_29_OUTPUT_0_OUT_WIDTH,
                           _208_CONV2D_29_PADDING_Y,
                           _208_CONV2D_29_PADDING_X,
                           _208_CONV2D_29_STRIDE_Y,
                           _208_CONV2D_29_STRIDE_X,
                           _208_CONV2D_29_DILATION_Y,
                           _208_CONV2D_29_DILATION_X,
                           _208_CONV2D_29_KERNEL_HEIGHT,
                           _208_CONV2D_29_KERNEL_WIDTH,
                           _208_CONV2D_29_ACTIVATION,
                           _204_MUL_52_OUTPUT_0_MEM_CONT_OFFSET,
                           _204_MUL_52_OUTPUT_0_MEM_CONT_SIZE,
                           _204_MUL_52_OUTPUT_0_MEM_WRAP_OFFSET,
                           _204_MUL_52_OUTPUT_0_MEM_WRAP_SIZE,
                           _204_MUL_52_OUTPUT_0_MEM_STRIDE,
                           _208_CONV2D_29_OUTPUT_0_MEM_CONT_OFFSET,
                           _208_CONV2D_29_OUTPUT_0_MEM_CONT_SIZE,
                           _208_CONV2D_29_OUTPUT_0_MEM_WRAP_OFFSET,
                           _208_CONV2D_29_OUTPUT_0_MEM_WRAP_SIZE,
                           _208_CONV2D_29_OUTPUT_0_MEM_STRIDE>
                           (_204_Mul_52_output_0, _208_Conv2D_29_output_0, _208_Conv2D_29_weights_output_0, _208_Conv2D_29_biases_output_0, _208_CONV2D_29_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _208_Conv2D_29\n");
    FILE* _208_CONV2D_29_OUTPUT_0_STREAM = fopen("data/export_outputs/_208_Conv2D_29_output_0.txt", "w");
    export_cpp::saveOutputs<_208_CONV2D_29_OUTPUT_0_NB_OUTPUTS,
                            _208_CONV2D_29_OUTPUT_0_OUT_HEIGHT,
                            _208_CONV2D_29_OUTPUT_0_OUT_WIDTH,
                            _208_CONV2D_29_OUTPUT_0_MEM_CONT_OFFSET,
                            _208_CONV2D_29_OUTPUT_0_MEM_CONT_SIZE,
                            _208_CONV2D_29_OUTPUT_0_MEM_WRAP_OFFSET,
                            _208_CONV2D_29_OUTPUT_0_MEM_WRAP_SIZE,
                            _208_CONV2D_29_OUTPUT_0_DEV_FMT>
                            (_208_Conv2D_29_output_0,
                            _208_CONV2D_29_OUTPUT_0_STREAM);
    fclose(_208_CONV2D_29_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _209_PadConvDw_6_output_0 = (float*) (mem_0 + _209_PADCONVDW_6_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_205_MUL_53_OUTPUT_0_IN_BATCH,
                           _209_PADCONVDW_6_NB_CHANNELS,
                           _205_MUL_53_OUTPUT_0_IN_HEIGHT,
                           _205_MUL_53_OUTPUT_0_IN_WIDTH,
                           _209_PADCONVDW_6_NB_OUTPUTS,
                           _209_PADCONVDW_6_OUTPUT_0_OUT_HEIGHT,
                           _209_PADCONVDW_6_OUTPUT_0_OUT_WIDTH,
                           _209_PADCONVDW_6_PADDING_Y,
                           _209_PADCONVDW_6_PADDING_X,
                           _209_PADCONVDW_6_STRIDE_Y,
                           _209_PADCONVDW_6_STRIDE_X,
                           _209_PADCONVDW_6_DILATION_Y,
                           _209_PADCONVDW_6_DILATION_X,
                           _209_PADCONVDW_6_KERNEL_HEIGHT,
                           _209_PADCONVDW_6_KERNEL_WIDTH,
                           _209_PADCONVDW_6_ACTIVATION,
                           _205_MUL_53_OUTPUT_0_MEM_CONT_OFFSET,
                           _205_MUL_53_OUTPUT_0_MEM_CONT_SIZE,
                           _205_MUL_53_OUTPUT_0_MEM_WRAP_OFFSET,
                           _205_MUL_53_OUTPUT_0_MEM_WRAP_SIZE,
                           _205_MUL_53_OUTPUT_0_MEM_STRIDE,
                           _209_PADCONVDW_6_OUTPUT_0_MEM_CONT_OFFSET,
                           _209_PADCONVDW_6_OUTPUT_0_MEM_CONT_SIZE,
                           _209_PADCONVDW_6_OUTPUT_0_MEM_WRAP_OFFSET,
                           _209_PADCONVDW_6_OUTPUT_0_MEM_WRAP_SIZE,
                           _209_PADCONVDW_6_OUTPUT_0_MEM_STRIDE>
                           (_205_Mul_53_output_0, _209_PadConvDw_6_output_0, _209_PadConvDw_6_weights_output_0, _209_PadConvDw_6_biases_output_0, _209_PADCONVDW_6_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _209_PadConvDw_6\n");
    FILE* _209_PADCONVDW_6_OUTPUT_0_STREAM = fopen("data/export_outputs/_209_PadConvDw_6_output_0.txt", "w");
    export_cpp::saveOutputs<_209_PADCONVDW_6_OUTPUT_0_NB_OUTPUTS,
                            _209_PADCONVDW_6_OUTPUT_0_OUT_HEIGHT,
                            _209_PADCONVDW_6_OUTPUT_0_OUT_WIDTH,
                            _209_PADCONVDW_6_OUTPUT_0_MEM_CONT_OFFSET,
                            _209_PADCONVDW_6_OUTPUT_0_MEM_CONT_SIZE,
                            _209_PADCONVDW_6_OUTPUT_0_MEM_WRAP_OFFSET,
                            _209_PADCONVDW_6_OUTPUT_0_MEM_WRAP_SIZE,
                            _209_PADCONVDW_6_OUTPUT_0_DEV_FMT>
                            (_209_PadConvDw_6_output_0,
                            _209_PADCONVDW_6_OUTPUT_0_STREAM);
    fclose(_209_PADCONVDW_6_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _210_Transpose_2_output_0 = (float*) (mem_0 + _210_TRANSPOSE_2_OUTPUT_0_MEM_OFFSET);

    export_cpp::transpose_ND_forward<float,
                                     _210_TRANSPOSE_2_NB_DIMS,
                                     _210_TRANSPOSE_2_NB_ELTS,
                                     _210_TRANSPOSE_2_PERMUTE,
                                     _210_TRANSPOSE_2_DIMS,
                                     _210_TRANSPOSE_2_OUT_STRIDES,
                                    _208_CONV2D_29_OUTPUT_0_MEM_CONT_OFFSET,
                                    _208_CONV2D_29_OUTPUT_0_MEM_CONT_SIZE,
                                    _208_CONV2D_29_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _208_CONV2D_29_OUTPUT_0_MEM_WRAP_SIZE,
                                    _208_CONV2D_29_OUTPUT_0_MEM_STRIDE,
                                    _210_TRANSPOSE_2_OUTPUT_0_MEM_CONT_OFFSET,
                                    _210_TRANSPOSE_2_OUTPUT_0_MEM_CONT_SIZE,
                                    _210_TRANSPOSE_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _210_TRANSPOSE_2_OUTPUT_0_MEM_WRAP_SIZE,
                                    _210_TRANSPOSE_2_OUTPUT_0_MEM_STRIDE>
                                     (_208_Conv2D_29_output_0,
                                     _210_Transpose_2_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _210_Transpose_2\n");
    FILE* _210_TRANSPOSE_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_210_Transpose_2_output_0.txt", "w");
    export_cpp::saveOutputs<_210_TRANSPOSE_2_OUTPUT_0_NB_OUTPUTS,
                            _210_TRANSPOSE_2_OUTPUT_0_OUT_HEIGHT,
                            _210_TRANSPOSE_2_OUTPUT_0_OUT_WIDTH,
                            _210_TRANSPOSE_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _210_TRANSPOSE_2_OUTPUT_0_MEM_CONT_SIZE,
                            _210_TRANSPOSE_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _210_TRANSPOSE_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _210_TRANSPOSE_2_OUTPUT_0_DEV_FMT>
                            (_210_Transpose_2_output_0,
                            _210_TRANSPOSE_2_OUTPUT_0_STREAM);
    fclose(_210_TRANSPOSE_2_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _211_Mul_54_output_0 = (float*) (mem_0 + _211_MUL_54_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_211_MUL_54_NB_MAT,
                                 _211_MUL_54_ELEM_OP,
                                 _211_MUL_54_INPUT1_CONT_SIZE,
                                 _211_MUL_54_INPUT2_CONT_SIZE,
                                 _211_MUL_54_OUTPUT_CONT_SIZE,
                                 _211_MUL_54_OFFSET_IN1,
                                 _211_MUL_54_OFFSET_IN2,
                                 _211_MUL_54_ACTIVATION,
                                 _206_CONV2D_28_OUTPUT_0_MEM_CONT_OFFSET,
                                 _206_CONV2D_28_OUTPUT_0_MEM_CONT_SIZE,
                                 _206_CONV2D_28_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _206_CONV2D_28_OUTPUT_0_MEM_WRAP_SIZE,
                                 _206_CONV2D_28_OUTPUT_0_MEM_STRIDE,
                                 _207_SIGMOID_54_OUTPUT_0_MEM_CONT_OFFSET,
                                 _207_SIGMOID_54_OUTPUT_0_MEM_CONT_SIZE,
                                 _207_SIGMOID_54_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _207_SIGMOID_54_OUTPUT_0_MEM_WRAP_SIZE,
                                 _207_SIGMOID_54_OUTPUT_0_MEM_STRIDE,
                                 _211_MUL_54_OUTPUT_0_MEM_CONT_OFFSET,
                                 _211_MUL_54_OUTPUT_0_MEM_CONT_SIZE,
                                 _211_MUL_54_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _211_MUL_54_OUTPUT_0_MEM_WRAP_SIZE,
                                 _211_MUL_54_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_211_Mul_54_output_0, _211_MUL_54_RESCALING, _206_Conv2D_28_output_0, _207_Sigmoid_54_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _211_Mul_54\n");
    FILE* _211_MUL_54_OUTPUT_0_STREAM = fopen("data/export_outputs/_211_Mul_54_output_0.txt", "w");
    export_cpp::saveOutputs<_211_MUL_54_OUTPUT_0_NB_OUTPUTS,
                            _211_MUL_54_OUTPUT_0_OUT_HEIGHT,
                            _211_MUL_54_OUTPUT_0_OUT_WIDTH,
                            _211_MUL_54_OUTPUT_0_MEM_CONT_OFFSET,
                            _211_MUL_54_OUTPUT_0_MEM_CONT_SIZE,
                            _211_MUL_54_OUTPUT_0_MEM_WRAP_OFFSET,
                            _211_MUL_54_OUTPUT_0_MEM_WRAP_SIZE,
                            _211_MUL_54_OUTPUT_0_DEV_FMT>
                            (_211_Mul_54_output_0,
                            _211_MUL_54_OUTPUT_0_STREAM);
    fclose(_211_MUL_54_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _212_Sigmoid_55_output_0 = (float*) (mem_0 + _212_SIGMOID_55_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_212_SIGMOID_55_NB_ELTS>
                               (_209_PadConvDw_6_output_0, _212_Sigmoid_55_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _212_Sigmoid_55\n");
    FILE* _212_SIGMOID_55_OUTPUT_0_STREAM = fopen("data/export_outputs/_212_Sigmoid_55_output_0.txt", "w");
    export_cpp::saveOutputs<_212_SIGMOID_55_OUTPUT_0_NB_OUTPUTS,
                            _212_SIGMOID_55_OUTPUT_0_OUT_HEIGHT,
                            _212_SIGMOID_55_OUTPUT_0_OUT_WIDTH,
                            _212_SIGMOID_55_OUTPUT_0_MEM_CONT_OFFSET,
                            _212_SIGMOID_55_OUTPUT_0_MEM_CONT_SIZE,
                            _212_SIGMOID_55_OUTPUT_0_MEM_WRAP_OFFSET,
                            _212_SIGMOID_55_OUTPUT_0_MEM_WRAP_SIZE,
                            _212_SIGMOID_55_OUTPUT_0_DEV_FMT>
                            (_212_Sigmoid_55_output_0,
                            _212_SIGMOID_55_OUTPUT_0_STREAM);
    fclose(_212_SIGMOID_55_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _213_Split_8_output_0 = (float*) (mem_0 + _213_SPLIT_8_OUTPUT_0_MEM_OFFSET);
    float* _213_Split_8_output_1 = (float*) (mem_0 + _213_SPLIT_8_OUTPUT_1_MEM_OFFSET);



    float* __213_Split_8_outputs[_213_SPLIT_8_NB_OUTPUTS] = {
        
        _213_Split_8_output_0,
        
        _213_Split_8_output_1
        
    };

    export_cpp::split_forward<_213_SPLIT_8_NB_OUTPUTS,
                              _213_SPLIT_8_AXIS_DIM,
                              _213_SPLIT_8_PRE_AXIS_ELEMS,
                              _213_SPLIT_8_POST_AXIS_ELEMS,
                              _213_SPLIT_8_SPLITS,
                              float>
                              (_211_Mul_54_output_0,
                               __213_Split_8_outputs);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _213_Split_8\n");
    FILE* _213_SPLIT_8_OUTPUT_0_STREAM = fopen("data/export_outputs/_213_Split_8_output_0.txt", "w");
    export_cpp::saveOutputs<_213_SPLIT_8_OUTPUT_0_NB_OUTPUTS,
                            _213_SPLIT_8_OUTPUT_0_OUT_HEIGHT,
                            _213_SPLIT_8_OUTPUT_0_OUT_WIDTH,
                            _213_SPLIT_8_OUTPUT_0_MEM_CONT_OFFSET,
                            _213_SPLIT_8_OUTPUT_0_MEM_CONT_SIZE,
                            _213_SPLIT_8_OUTPUT_0_MEM_WRAP_OFFSET,
                            _213_SPLIT_8_OUTPUT_0_MEM_WRAP_SIZE,
                            _213_SPLIT_8_OUTPUT_0_DEV_FMT>
                            (_213_Split_8_output_0,
                            _213_SPLIT_8_OUTPUT_0_STREAM);
    fclose(_213_SPLIT_8_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _214_Mul_55_output_0 = (float*) (mem_0 + _214_MUL_55_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_214_MUL_55_NB_MAT,
                                 _214_MUL_55_ELEM_OP,
                                 _214_MUL_55_INPUT1_CONT_SIZE,
                                 _214_MUL_55_INPUT2_CONT_SIZE,
                                 _214_MUL_55_OUTPUT_CONT_SIZE,
                                 _214_MUL_55_OFFSET_IN1,
                                 _214_MUL_55_OFFSET_IN2,
                                 _214_MUL_55_ACTIVATION,
                                 _209_PADCONVDW_6_OUTPUT_0_MEM_CONT_OFFSET,
                                 _209_PADCONVDW_6_OUTPUT_0_MEM_CONT_SIZE,
                                 _209_PADCONVDW_6_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _209_PADCONVDW_6_OUTPUT_0_MEM_WRAP_SIZE,
                                 _209_PADCONVDW_6_OUTPUT_0_MEM_STRIDE,
                                 _212_SIGMOID_55_OUTPUT_0_MEM_CONT_OFFSET,
                                 _212_SIGMOID_55_OUTPUT_0_MEM_CONT_SIZE,
                                 _212_SIGMOID_55_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _212_SIGMOID_55_OUTPUT_0_MEM_WRAP_SIZE,
                                 _212_SIGMOID_55_OUTPUT_0_MEM_STRIDE,
                                 _214_MUL_55_OUTPUT_0_MEM_CONT_OFFSET,
                                 _214_MUL_55_OUTPUT_0_MEM_CONT_SIZE,
                                 _214_MUL_55_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _214_MUL_55_OUTPUT_0_MEM_WRAP_SIZE,
                                 _214_MUL_55_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_214_Mul_55_output_0, _214_MUL_55_RESCALING, _209_PadConvDw_6_output_0, _212_Sigmoid_55_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _214_Mul_55\n");
    FILE* _214_MUL_55_OUTPUT_0_STREAM = fopen("data/export_outputs/_214_Mul_55_output_0.txt", "w");
    export_cpp::saveOutputs<_214_MUL_55_OUTPUT_0_NB_OUTPUTS,
                            _214_MUL_55_OUTPUT_0_OUT_HEIGHT,
                            _214_MUL_55_OUTPUT_0_OUT_WIDTH,
                            _214_MUL_55_OUTPUT_0_MEM_CONT_OFFSET,
                            _214_MUL_55_OUTPUT_0_MEM_CONT_SIZE,
                            _214_MUL_55_OUTPUT_0_MEM_WRAP_OFFSET,
                            _214_MUL_55_OUTPUT_0_MEM_WRAP_SIZE,
                            _214_MUL_55_OUTPUT_0_DEV_FMT>
                            (_214_Mul_55_output_0,
                            _214_MUL_55_OUTPUT_0_STREAM);
    fclose(_214_MUL_55_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _215_Reshape_2_output_0 = (float*) (mem_0 + _215_RESHAPE_2_OUTPUT_0_MEM_OFFSET);


    export_cpp::identity_forward<_215_RESHAPE_2_NB_ELTS,
                                 _210_TRANSPOSE_2_OUTPUT_0_MEM_CONT_OFFSET,
                                 _210_TRANSPOSE_2_OUTPUT_0_MEM_CONT_SIZE,
                                 _210_TRANSPOSE_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _210_TRANSPOSE_2_OUTPUT_0_MEM_WRAP_SIZE,
                                 _210_TRANSPOSE_2_OUTPUT_0_MEM_STRIDE,
                                 _215_RESHAPE_2_OUTPUT_0_MEM_CONT_OFFSET,
                                 _215_RESHAPE_2_OUTPUT_0_MEM_CONT_SIZE,
                                 _215_RESHAPE_2_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _215_RESHAPE_2_OUTPUT_0_MEM_WRAP_SIZE,
                                 _215_RESHAPE_2_OUTPUT_0_MEM_STRIDE>
                                (_210_Transpose_2_output_0, _215_Reshape_2_output_0);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _215_Reshape_2\n");
    FILE* _215_RESHAPE_2_OUTPUT_0_STREAM = fopen("data/export_outputs/_215_Reshape_2_output_0.txt", "w");
    export_cpp::saveOutputs<_215_RESHAPE_2_OUTPUT_0_NB_OUTPUTS,
                            _215_RESHAPE_2_OUTPUT_0_OUT_HEIGHT,
                            _215_RESHAPE_2_OUTPUT_0_OUT_WIDTH,
                            _215_RESHAPE_2_OUTPUT_0_MEM_CONT_OFFSET,
                            _215_RESHAPE_2_OUTPUT_0_MEM_CONT_SIZE,
                            _215_RESHAPE_2_OUTPUT_0_MEM_WRAP_OFFSET,
                            _215_RESHAPE_2_OUTPUT_0_MEM_WRAP_SIZE,
                            _215_RESHAPE_2_OUTPUT_0_DEV_FMT>
                            (_215_Reshape_2_output_0,
                            _215_RESHAPE_2_OUTPUT_0_STREAM);
    fclose(_215_RESHAPE_2_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _216_Conv2D_30_output_0 = (float*) (mem_0 + _216_CONV2D_30_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_214_MUL_55_OUTPUT_0_IN_BATCH,
                           _216_CONV2D_30_NB_CHANNELS,
                           _214_MUL_55_OUTPUT_0_IN_HEIGHT,
                           _214_MUL_55_OUTPUT_0_IN_WIDTH,
                           _216_CONV2D_30_NB_OUTPUTS,
                           _216_CONV2D_30_OUTPUT_0_OUT_HEIGHT,
                           _216_CONV2D_30_OUTPUT_0_OUT_WIDTH,
                           _216_CONV2D_30_PADDING_Y,
                           _216_CONV2D_30_PADDING_X,
                           _216_CONV2D_30_STRIDE_Y,
                           _216_CONV2D_30_STRIDE_X,
                           _216_CONV2D_30_DILATION_Y,
                           _216_CONV2D_30_DILATION_X,
                           _216_CONV2D_30_KERNEL_HEIGHT,
                           _216_CONV2D_30_KERNEL_WIDTH,
                           _216_CONV2D_30_ACTIVATION,
                           _214_MUL_55_OUTPUT_0_MEM_CONT_OFFSET,
                           _214_MUL_55_OUTPUT_0_MEM_CONT_SIZE,
                           _214_MUL_55_OUTPUT_0_MEM_WRAP_OFFSET,
                           _214_MUL_55_OUTPUT_0_MEM_WRAP_SIZE,
                           _214_MUL_55_OUTPUT_0_MEM_STRIDE,
                           _216_CONV2D_30_OUTPUT_0_MEM_CONT_OFFSET,
                           _216_CONV2D_30_OUTPUT_0_MEM_CONT_SIZE,
                           _216_CONV2D_30_OUTPUT_0_MEM_WRAP_OFFSET,
                           _216_CONV2D_30_OUTPUT_0_MEM_WRAP_SIZE,
                           _216_CONV2D_30_OUTPUT_0_MEM_STRIDE>
                           (_214_Mul_55_output_0, _216_Conv2D_30_output_0, _216_Conv2D_30_weights_output_0, _216_Conv2D_30_biases_output_0, _216_CONV2D_30_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _216_Conv2D_30\n");
    FILE* _216_CONV2D_30_OUTPUT_0_STREAM = fopen("data/export_outputs/_216_Conv2D_30_output_0.txt", "w");
    export_cpp::saveOutputs<_216_CONV2D_30_OUTPUT_0_NB_OUTPUTS,
                            _216_CONV2D_30_OUTPUT_0_OUT_HEIGHT,
                            _216_CONV2D_30_OUTPUT_0_OUT_WIDTH,
                            _216_CONV2D_30_OUTPUT_0_MEM_CONT_OFFSET,
                            _216_CONV2D_30_OUTPUT_0_MEM_CONT_SIZE,
                            _216_CONV2D_30_OUTPUT_0_MEM_WRAP_OFFSET,
                            _216_CONV2D_30_OUTPUT_0_MEM_WRAP_SIZE,
                            _216_CONV2D_30_OUTPUT_0_DEV_FMT>
                            (_216_Conv2D_30_output_0,
                            _216_CONV2D_30_OUTPUT_0_STREAM);
    fclose(_216_CONV2D_30_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _217_PadConvDw_7_output_0 = (float*) (mem_0 + _217_PADCONVDW_7_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_213_SPLIT_8_OUTPUT_1_IN_BATCH,
                           _217_PADCONVDW_7_NB_CHANNELS,
                           _213_SPLIT_8_OUTPUT_1_IN_HEIGHT,
                           _213_SPLIT_8_OUTPUT_1_IN_WIDTH,
                           _217_PADCONVDW_7_NB_OUTPUTS,
                           _217_PADCONVDW_7_OUTPUT_0_OUT_HEIGHT,
                           _217_PADCONVDW_7_OUTPUT_0_OUT_WIDTH,
                           _217_PADCONVDW_7_PADDING_Y,
                           _217_PADCONVDW_7_PADDING_X,
                           _217_PADCONVDW_7_STRIDE_Y,
                           _217_PADCONVDW_7_STRIDE_X,
                           _217_PADCONVDW_7_DILATION_Y,
                           _217_PADCONVDW_7_DILATION_X,
                           _217_PADCONVDW_7_KERNEL_HEIGHT,
                           _217_PADCONVDW_7_KERNEL_WIDTH,
                           _217_PADCONVDW_7_ACTIVATION,
                           _213_SPLIT_8_OUTPUT_1_MEM_CONT_OFFSET,
                           _213_SPLIT_8_OUTPUT_1_MEM_CONT_SIZE,
                           _213_SPLIT_8_OUTPUT_1_MEM_WRAP_OFFSET,
                           _213_SPLIT_8_OUTPUT_1_MEM_WRAP_SIZE,
                           _213_SPLIT_8_OUTPUT_1_MEM_STRIDE,
                           _217_PADCONVDW_7_OUTPUT_0_MEM_CONT_OFFSET,
                           _217_PADCONVDW_7_OUTPUT_0_MEM_CONT_SIZE,
                           _217_PADCONVDW_7_OUTPUT_0_MEM_WRAP_OFFSET,
                           _217_PADCONVDW_7_OUTPUT_0_MEM_WRAP_SIZE,
                           _217_PADCONVDW_7_OUTPUT_0_MEM_STRIDE>
                           (_213_Split_8_output_1, _217_PadConvDw_7_output_0, _217_PadConvDw_7_weights_output_0, _217_PadConvDw_7_biases_output_0, _217_PADCONVDW_7_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _217_PadConvDw_7\n");
    FILE* _217_PADCONVDW_7_OUTPUT_0_STREAM = fopen("data/export_outputs/_217_PadConvDw_7_output_0.txt", "w");
    export_cpp::saveOutputs<_217_PADCONVDW_7_OUTPUT_0_NB_OUTPUTS,
                            _217_PADCONVDW_7_OUTPUT_0_OUT_HEIGHT,
                            _217_PADCONVDW_7_OUTPUT_0_OUT_WIDTH,
                            _217_PADCONVDW_7_OUTPUT_0_MEM_CONT_OFFSET,
                            _217_PADCONVDW_7_OUTPUT_0_MEM_CONT_SIZE,
                            _217_PADCONVDW_7_OUTPUT_0_MEM_WRAP_OFFSET,
                            _217_PADCONVDW_7_OUTPUT_0_MEM_WRAP_SIZE,
                            _217_PADCONVDW_7_OUTPUT_0_DEV_FMT>
                            (_217_PadConvDw_7_output_0,
                            _217_PADCONVDW_7_OUTPUT_0_STREAM);
    fclose(_217_PADCONVDW_7_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _218_Sigmoid_56_output_0 = (float*) (mem_0 + _218_SIGMOID_56_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_218_SIGMOID_56_NB_ELTS>
                               (_217_PadConvDw_7_output_0, _218_Sigmoid_56_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _218_Sigmoid_56\n");
    FILE* _218_SIGMOID_56_OUTPUT_0_STREAM = fopen("data/export_outputs/_218_Sigmoid_56_output_0.txt", "w");
    export_cpp::saveOutputs<_218_SIGMOID_56_OUTPUT_0_NB_OUTPUTS,
                            _218_SIGMOID_56_OUTPUT_0_OUT_HEIGHT,
                            _218_SIGMOID_56_OUTPUT_0_OUT_WIDTH,
                            _218_SIGMOID_56_OUTPUT_0_MEM_CONT_OFFSET,
                            _218_SIGMOID_56_OUTPUT_0_MEM_CONT_SIZE,
                            _218_SIGMOID_56_OUTPUT_0_MEM_WRAP_OFFSET,
                            _218_SIGMOID_56_OUTPUT_0_MEM_WRAP_SIZE,
                            _218_SIGMOID_56_OUTPUT_0_DEV_FMT>
                            (_218_Sigmoid_56_output_0,
                            _218_SIGMOID_56_OUTPUT_0_STREAM);
    fclose(_218_SIGMOID_56_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _219_Sigmoid_57_output_0 = (float*) (mem_0 + _219_SIGMOID_57_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_219_SIGMOID_57_NB_ELTS>
                               (_216_Conv2D_30_output_0, _219_Sigmoid_57_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _219_Sigmoid_57\n");
    FILE* _219_SIGMOID_57_OUTPUT_0_STREAM = fopen("data/export_outputs/_219_Sigmoid_57_output_0.txt", "w");
    export_cpp::saveOutputs<_219_SIGMOID_57_OUTPUT_0_NB_OUTPUTS,
                            _219_SIGMOID_57_OUTPUT_0_OUT_HEIGHT,
                            _219_SIGMOID_57_OUTPUT_0_OUT_WIDTH,
                            _219_SIGMOID_57_OUTPUT_0_MEM_CONT_OFFSET,
                            _219_SIGMOID_57_OUTPUT_0_MEM_CONT_SIZE,
                            _219_SIGMOID_57_OUTPUT_0_MEM_WRAP_OFFSET,
                            _219_SIGMOID_57_OUTPUT_0_MEM_WRAP_SIZE,
                            _219_SIGMOID_57_OUTPUT_0_DEV_FMT>
                            (_219_Sigmoid_57_output_0,
                            _219_SIGMOID_57_OUTPUT_0_STREAM);
    fclose(_219_SIGMOID_57_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _220_Mul_56_output_0 = (float*) (mem_0 + _220_MUL_56_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_220_MUL_56_NB_MAT,
                                 _220_MUL_56_ELEM_OP,
                                 _220_MUL_56_INPUT1_CONT_SIZE,
                                 _220_MUL_56_INPUT2_CONT_SIZE,
                                 _220_MUL_56_OUTPUT_CONT_SIZE,
                                 _220_MUL_56_OFFSET_IN1,
                                 _220_MUL_56_OFFSET_IN2,
                                 _220_MUL_56_ACTIVATION,
                                 _217_PADCONVDW_7_OUTPUT_0_MEM_CONT_OFFSET,
                                 _217_PADCONVDW_7_OUTPUT_0_MEM_CONT_SIZE,
                                 _217_PADCONVDW_7_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _217_PADCONVDW_7_OUTPUT_0_MEM_WRAP_SIZE,
                                 _217_PADCONVDW_7_OUTPUT_0_MEM_STRIDE,
                                 _218_SIGMOID_56_OUTPUT_0_MEM_CONT_OFFSET,
                                 _218_SIGMOID_56_OUTPUT_0_MEM_CONT_SIZE,
                                 _218_SIGMOID_56_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _218_SIGMOID_56_OUTPUT_0_MEM_WRAP_SIZE,
                                 _218_SIGMOID_56_OUTPUT_0_MEM_STRIDE,
                                 _220_MUL_56_OUTPUT_0_MEM_CONT_OFFSET,
                                 _220_MUL_56_OUTPUT_0_MEM_CONT_SIZE,
                                 _220_MUL_56_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _220_MUL_56_OUTPUT_0_MEM_WRAP_SIZE,
                                 _220_MUL_56_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_220_Mul_56_output_0, _220_MUL_56_RESCALING, _217_PadConvDw_7_output_0, _218_Sigmoid_56_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _220_Mul_56\n");
    FILE* _220_MUL_56_OUTPUT_0_STREAM = fopen("data/export_outputs/_220_Mul_56_output_0.txt", "w");
    export_cpp::saveOutputs<_220_MUL_56_OUTPUT_0_NB_OUTPUTS,
                            _220_MUL_56_OUTPUT_0_OUT_HEIGHT,
                            _220_MUL_56_OUTPUT_0_OUT_WIDTH,
                            _220_MUL_56_OUTPUT_0_MEM_CONT_OFFSET,
                            _220_MUL_56_OUTPUT_0_MEM_CONT_SIZE,
                            _220_MUL_56_OUTPUT_0_MEM_WRAP_OFFSET,
                            _220_MUL_56_OUTPUT_0_MEM_WRAP_SIZE,
                            _220_MUL_56_OUTPUT_0_DEV_FMT>
                            (_220_Mul_56_output_0,
                            _220_MUL_56_OUTPUT_0_STREAM);
    fclose(_220_MUL_56_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _221_Mul_57_output_0 = (float*) (mem_0 + _221_MUL_57_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_221_MUL_57_NB_MAT,
                                 _221_MUL_57_ELEM_OP,
                                 _221_MUL_57_INPUT1_CONT_SIZE,
                                 _221_MUL_57_INPUT2_CONT_SIZE,
                                 _221_MUL_57_OUTPUT_CONT_SIZE,
                                 _221_MUL_57_OFFSET_IN1,
                                 _221_MUL_57_OFFSET_IN2,
                                 _221_MUL_57_ACTIVATION,
                                 _216_CONV2D_30_OUTPUT_0_MEM_CONT_OFFSET,
                                 _216_CONV2D_30_OUTPUT_0_MEM_CONT_SIZE,
                                 _216_CONV2D_30_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _216_CONV2D_30_OUTPUT_0_MEM_WRAP_SIZE,
                                 _216_CONV2D_30_OUTPUT_0_MEM_STRIDE,
                                 _219_SIGMOID_57_OUTPUT_0_MEM_CONT_OFFSET,
                                 _219_SIGMOID_57_OUTPUT_0_MEM_CONT_SIZE,
                                 _219_SIGMOID_57_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _219_SIGMOID_57_OUTPUT_0_MEM_WRAP_SIZE,
                                 _219_SIGMOID_57_OUTPUT_0_MEM_STRIDE,
                                 _221_MUL_57_OUTPUT_0_MEM_CONT_OFFSET,
                                 _221_MUL_57_OUTPUT_0_MEM_CONT_SIZE,
                                 _221_MUL_57_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _221_MUL_57_OUTPUT_0_MEM_WRAP_SIZE,
                                 _221_MUL_57_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_221_Mul_57_output_0, _221_MUL_57_RESCALING, _216_Conv2D_30_output_0, _219_Sigmoid_57_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _221_Mul_57\n");
    FILE* _221_MUL_57_OUTPUT_0_STREAM = fopen("data/export_outputs/_221_Mul_57_output_0.txt", "w");
    export_cpp::saveOutputs<_221_MUL_57_OUTPUT_0_NB_OUTPUTS,
                            _221_MUL_57_OUTPUT_0_OUT_HEIGHT,
                            _221_MUL_57_OUTPUT_0_OUT_WIDTH,
                            _221_MUL_57_OUTPUT_0_MEM_CONT_OFFSET,
                            _221_MUL_57_OUTPUT_0_MEM_CONT_SIZE,
                            _221_MUL_57_OUTPUT_0_MEM_WRAP_OFFSET,
                            _221_MUL_57_OUTPUT_0_MEM_WRAP_SIZE,
                            _221_MUL_57_OUTPUT_0_DEV_FMT>
                            (_221_Mul_57_output_0,
                            _221_MUL_57_OUTPUT_0_STREAM);
    fclose(_221_MUL_57_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _222_Conv2D_31_output_0 = (float*) (mem_0 + _222_CONV2D_31_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_220_MUL_56_OUTPUT_0_IN_BATCH,
                           _222_CONV2D_31_NB_CHANNELS,
                           _220_MUL_56_OUTPUT_0_IN_HEIGHT,
                           _220_MUL_56_OUTPUT_0_IN_WIDTH,
                           _222_CONV2D_31_NB_OUTPUTS,
                           _222_CONV2D_31_OUTPUT_0_OUT_HEIGHT,
                           _222_CONV2D_31_OUTPUT_0_OUT_WIDTH,
                           _222_CONV2D_31_PADDING_Y,
                           _222_CONV2D_31_PADDING_X,
                           _222_CONV2D_31_STRIDE_Y,
                           _222_CONV2D_31_STRIDE_X,
                           _222_CONV2D_31_DILATION_Y,
                           _222_CONV2D_31_DILATION_X,
                           _222_CONV2D_31_KERNEL_HEIGHT,
                           _222_CONV2D_31_KERNEL_WIDTH,
                           _222_CONV2D_31_ACTIVATION,
                           _220_MUL_56_OUTPUT_0_MEM_CONT_OFFSET,
                           _220_MUL_56_OUTPUT_0_MEM_CONT_SIZE,
                           _220_MUL_56_OUTPUT_0_MEM_WRAP_OFFSET,
                           _220_MUL_56_OUTPUT_0_MEM_WRAP_SIZE,
                           _220_MUL_56_OUTPUT_0_MEM_STRIDE,
                           _222_CONV2D_31_OUTPUT_0_MEM_CONT_OFFSET,
                           _222_CONV2D_31_OUTPUT_0_MEM_CONT_SIZE,
                           _222_CONV2D_31_OUTPUT_0_MEM_WRAP_OFFSET,
                           _222_CONV2D_31_OUTPUT_0_MEM_WRAP_SIZE,
                           _222_CONV2D_31_OUTPUT_0_MEM_STRIDE>
                           (_220_Mul_56_output_0, _222_Conv2D_31_output_0, _222_Conv2D_31_weights_output_0, _222_Conv2D_31_biases_output_0, _222_CONV2D_31_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _222_Conv2D_31\n");
    FILE* _222_CONV2D_31_OUTPUT_0_STREAM = fopen("data/export_outputs/_222_Conv2D_31_output_0.txt", "w");
    export_cpp::saveOutputs<_222_CONV2D_31_OUTPUT_0_NB_OUTPUTS,
                            _222_CONV2D_31_OUTPUT_0_OUT_HEIGHT,
                            _222_CONV2D_31_OUTPUT_0_OUT_WIDTH,
                            _222_CONV2D_31_OUTPUT_0_MEM_CONT_OFFSET,
                            _222_CONV2D_31_OUTPUT_0_MEM_CONT_SIZE,
                            _222_CONV2D_31_OUTPUT_0_MEM_WRAP_OFFSET,
                            _222_CONV2D_31_OUTPUT_0_MEM_WRAP_SIZE,
                            _222_CONV2D_31_OUTPUT_0_DEV_FMT>
                            (_222_Conv2D_31_output_0,
                            _222_CONV2D_31_OUTPUT_0_STREAM);
    fclose(_222_CONV2D_31_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _223_Conv2D_32_output_0 = (float*) (mem_0 + _223_CONV2D_32_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_221_MUL_57_OUTPUT_0_IN_BATCH,
                           _223_CONV2D_32_NB_CHANNELS,
                           _221_MUL_57_OUTPUT_0_IN_HEIGHT,
                           _221_MUL_57_OUTPUT_0_IN_WIDTH,
                           _223_CONV2D_32_NB_OUTPUTS,
                           _223_CONV2D_32_OUTPUT_0_OUT_HEIGHT,
                           _223_CONV2D_32_OUTPUT_0_OUT_WIDTH,
                           _223_CONV2D_32_PADDING_Y,
                           _223_CONV2D_32_PADDING_X,
                           _223_CONV2D_32_STRIDE_Y,
                           _223_CONV2D_32_STRIDE_X,
                           _223_CONV2D_32_DILATION_Y,
                           _223_CONV2D_32_DILATION_X,
                           _223_CONV2D_32_KERNEL_HEIGHT,
                           _223_CONV2D_32_KERNEL_WIDTH,
                           _223_CONV2D_32_ACTIVATION,
                           _221_MUL_57_OUTPUT_0_MEM_CONT_OFFSET,
                           _221_MUL_57_OUTPUT_0_MEM_CONT_SIZE,
                           _221_MUL_57_OUTPUT_0_MEM_WRAP_OFFSET,
                           _221_MUL_57_OUTPUT_0_MEM_WRAP_SIZE,
                           _221_MUL_57_OUTPUT_0_MEM_STRIDE,
                           _223_CONV2D_32_OUTPUT_0_MEM_CONT_OFFSET,
                           _223_CONV2D_32_OUTPUT_0_MEM_CONT_SIZE,
                           _223_CONV2D_32_OUTPUT_0_MEM_WRAP_OFFSET,
                           _223_CONV2D_32_OUTPUT_0_MEM_WRAP_SIZE,
                           _223_CONV2D_32_OUTPUT_0_MEM_STRIDE>
                           (_221_Mul_57_output_0, _223_Conv2D_32_output_0, _223_Conv2D_32_weights_output_0, _223_Conv2D_32_biases_output_0, _223_CONV2D_32_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _223_Conv2D_32\n");
    FILE* _223_CONV2D_32_OUTPUT_0_STREAM = fopen("data/export_outputs/_223_Conv2D_32_output_0.txt", "w");
    export_cpp::saveOutputs<_223_CONV2D_32_OUTPUT_0_NB_OUTPUTS,
                            _223_CONV2D_32_OUTPUT_0_OUT_HEIGHT,
                            _223_CONV2D_32_OUTPUT_0_OUT_WIDTH,
                            _223_CONV2D_32_OUTPUT_0_MEM_CONT_OFFSET,
                            _223_CONV2D_32_OUTPUT_0_MEM_CONT_SIZE,
                            _223_CONV2D_32_OUTPUT_0_MEM_WRAP_OFFSET,
                            _223_CONV2D_32_OUTPUT_0_MEM_WRAP_SIZE,
                            _223_CONV2D_32_OUTPUT_0_DEV_FMT>
                            (_223_Conv2D_32_output_0,
                            _223_CONV2D_32_OUTPUT_0_STREAM);
    fclose(_223_CONV2D_32_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _224_Sigmoid_58_output_0 = (float*) (mem_0 + _224_SIGMOID_58_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_224_SIGMOID_58_NB_ELTS>
                               (_222_Conv2D_31_output_0, _224_Sigmoid_58_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _224_Sigmoid_58\n");
    FILE* _224_SIGMOID_58_OUTPUT_0_STREAM = fopen("data/export_outputs/_224_Sigmoid_58_output_0.txt", "w");
    export_cpp::saveOutputs<_224_SIGMOID_58_OUTPUT_0_NB_OUTPUTS,
                            _224_SIGMOID_58_OUTPUT_0_OUT_HEIGHT,
                            _224_SIGMOID_58_OUTPUT_0_OUT_WIDTH,
                            _224_SIGMOID_58_OUTPUT_0_MEM_CONT_OFFSET,
                            _224_SIGMOID_58_OUTPUT_0_MEM_CONT_SIZE,
                            _224_SIGMOID_58_OUTPUT_0_MEM_WRAP_OFFSET,
                            _224_SIGMOID_58_OUTPUT_0_MEM_WRAP_SIZE,
                            _224_SIGMOID_58_OUTPUT_0_DEV_FMT>
                            (_224_Sigmoid_58_output_0,
                            _224_SIGMOID_58_OUTPUT_0_STREAM);
    fclose(_224_SIGMOID_58_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _225_Transpose_3_output_0 = (float*) (mem_0 + _225_TRANSPOSE_3_OUTPUT_0_MEM_OFFSET);

    export_cpp::transpose_ND_forward<float,
                                     _225_TRANSPOSE_3_NB_DIMS,
                                     _225_TRANSPOSE_3_NB_ELTS,
                                     _225_TRANSPOSE_3_PERMUTE,
                                     _225_TRANSPOSE_3_DIMS,
                                     _225_TRANSPOSE_3_OUT_STRIDES,
                                    _223_CONV2D_32_OUTPUT_0_MEM_CONT_OFFSET,
                                    _223_CONV2D_32_OUTPUT_0_MEM_CONT_SIZE,
                                    _223_CONV2D_32_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _223_CONV2D_32_OUTPUT_0_MEM_WRAP_SIZE,
                                    _223_CONV2D_32_OUTPUT_0_MEM_STRIDE,
                                    _225_TRANSPOSE_3_OUTPUT_0_MEM_CONT_OFFSET,
                                    _225_TRANSPOSE_3_OUTPUT_0_MEM_CONT_SIZE,
                                    _225_TRANSPOSE_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _225_TRANSPOSE_3_OUTPUT_0_MEM_WRAP_SIZE,
                                    _225_TRANSPOSE_3_OUTPUT_0_MEM_STRIDE>
                                     (_223_Conv2D_32_output_0,
                                     _225_Transpose_3_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _225_Transpose_3\n");
    FILE* _225_TRANSPOSE_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_225_Transpose_3_output_0.txt", "w");
    export_cpp::saveOutputs<_225_TRANSPOSE_3_OUTPUT_0_NB_OUTPUTS,
                            _225_TRANSPOSE_3_OUTPUT_0_OUT_HEIGHT,
                            _225_TRANSPOSE_3_OUTPUT_0_OUT_WIDTH,
                            _225_TRANSPOSE_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _225_TRANSPOSE_3_OUTPUT_0_MEM_CONT_SIZE,
                            _225_TRANSPOSE_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _225_TRANSPOSE_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _225_TRANSPOSE_3_OUTPUT_0_DEV_FMT>
                            (_225_Transpose_3_output_0,
                            _225_TRANSPOSE_3_OUTPUT_0_STREAM);
    fclose(_225_TRANSPOSE_3_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _226_Mul_58_output_0 = (float*) (mem_0 + _226_MUL_58_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_226_MUL_58_NB_MAT,
                                 _226_MUL_58_ELEM_OP,
                                 _226_MUL_58_INPUT1_CONT_SIZE,
                                 _226_MUL_58_INPUT2_CONT_SIZE,
                                 _226_MUL_58_OUTPUT_CONT_SIZE,
                                 _226_MUL_58_OFFSET_IN1,
                                 _226_MUL_58_OFFSET_IN2,
                                 _226_MUL_58_ACTIVATION,
                                 _222_CONV2D_31_OUTPUT_0_MEM_CONT_OFFSET,
                                 _222_CONV2D_31_OUTPUT_0_MEM_CONT_SIZE,
                                 _222_CONV2D_31_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _222_CONV2D_31_OUTPUT_0_MEM_WRAP_SIZE,
                                 _222_CONV2D_31_OUTPUT_0_MEM_STRIDE,
                                 _224_SIGMOID_58_OUTPUT_0_MEM_CONT_OFFSET,
                                 _224_SIGMOID_58_OUTPUT_0_MEM_CONT_SIZE,
                                 _224_SIGMOID_58_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _224_SIGMOID_58_OUTPUT_0_MEM_WRAP_SIZE,
                                 _224_SIGMOID_58_OUTPUT_0_MEM_STRIDE,
                                 _226_MUL_58_OUTPUT_0_MEM_CONT_OFFSET,
                                 _226_MUL_58_OUTPUT_0_MEM_CONT_SIZE,
                                 _226_MUL_58_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _226_MUL_58_OUTPUT_0_MEM_WRAP_SIZE,
                                 _226_MUL_58_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_226_Mul_58_output_0, _226_MUL_58_RESCALING, _222_Conv2D_31_output_0, _224_Sigmoid_58_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _226_Mul_58\n");
    FILE* _226_MUL_58_OUTPUT_0_STREAM = fopen("data/export_outputs/_226_Mul_58_output_0.txt", "w");
    export_cpp::saveOutputs<_226_MUL_58_OUTPUT_0_NB_OUTPUTS,
                            _226_MUL_58_OUTPUT_0_OUT_HEIGHT,
                            _226_MUL_58_OUTPUT_0_OUT_WIDTH,
                            _226_MUL_58_OUTPUT_0_MEM_CONT_OFFSET,
                            _226_MUL_58_OUTPUT_0_MEM_CONT_SIZE,
                            _226_MUL_58_OUTPUT_0_MEM_WRAP_OFFSET,
                            _226_MUL_58_OUTPUT_0_MEM_WRAP_SIZE,
                            _226_MUL_58_OUTPUT_0_DEV_FMT>
                            (_226_Mul_58_output_0,
                            _226_MUL_58_OUTPUT_0_STREAM);
    fclose(_226_MUL_58_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _227_Reshape_3_output_0 = (float*) (mem_0 + _227_RESHAPE_3_OUTPUT_0_MEM_OFFSET);


    export_cpp::identity_forward<_227_RESHAPE_3_NB_ELTS,
                                 _225_TRANSPOSE_3_OUTPUT_0_MEM_CONT_OFFSET,
                                 _225_TRANSPOSE_3_OUTPUT_0_MEM_CONT_SIZE,
                                 _225_TRANSPOSE_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _225_TRANSPOSE_3_OUTPUT_0_MEM_WRAP_SIZE,
                                 _225_TRANSPOSE_3_OUTPUT_0_MEM_STRIDE,
                                 _227_RESHAPE_3_OUTPUT_0_MEM_CONT_OFFSET,
                                 _227_RESHAPE_3_OUTPUT_0_MEM_CONT_SIZE,
                                 _227_RESHAPE_3_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _227_RESHAPE_3_OUTPUT_0_MEM_WRAP_SIZE,
                                 _227_RESHAPE_3_OUTPUT_0_MEM_STRIDE>
                                (_225_Transpose_3_output_0, _227_Reshape_3_output_0);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _227_Reshape_3\n");
    FILE* _227_RESHAPE_3_OUTPUT_0_STREAM = fopen("data/export_outputs/_227_Reshape_3_output_0.txt", "w");
    export_cpp::saveOutputs<_227_RESHAPE_3_OUTPUT_0_NB_OUTPUTS,
                            _227_RESHAPE_3_OUTPUT_0_OUT_HEIGHT,
                            _227_RESHAPE_3_OUTPUT_0_OUT_WIDTH,
                            _227_RESHAPE_3_OUTPUT_0_MEM_CONT_OFFSET,
                            _227_RESHAPE_3_OUTPUT_0_MEM_CONT_SIZE,
                            _227_RESHAPE_3_OUTPUT_0_MEM_WRAP_OFFSET,
                            _227_RESHAPE_3_OUTPUT_0_MEM_WRAP_SIZE,
                            _227_RESHAPE_3_OUTPUT_0_DEV_FMT>
                            (_227_Reshape_3_output_0,
                            _227_RESHAPE_3_OUTPUT_0_STREAM);
    fclose(_227_RESHAPE_3_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _228_PadConvDw_8_output_0 = (float*) (mem_0 + _228_PADCONVDW_8_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_226_MUL_58_OUTPUT_0_IN_BATCH,
                           _228_PADCONVDW_8_NB_CHANNELS,
                           _226_MUL_58_OUTPUT_0_IN_HEIGHT,
                           _226_MUL_58_OUTPUT_0_IN_WIDTH,
                           _228_PADCONVDW_8_NB_OUTPUTS,
                           _228_PADCONVDW_8_OUTPUT_0_OUT_HEIGHT,
                           _228_PADCONVDW_8_OUTPUT_0_OUT_WIDTH,
                           _228_PADCONVDW_8_PADDING_Y,
                           _228_PADCONVDW_8_PADDING_X,
                           _228_PADCONVDW_8_STRIDE_Y,
                           _228_PADCONVDW_8_STRIDE_X,
                           _228_PADCONVDW_8_DILATION_Y,
                           _228_PADCONVDW_8_DILATION_X,
                           _228_PADCONVDW_8_KERNEL_HEIGHT,
                           _228_PADCONVDW_8_KERNEL_WIDTH,
                           _228_PADCONVDW_8_ACTIVATION,
                           _226_MUL_58_OUTPUT_0_MEM_CONT_OFFSET,
                           _226_MUL_58_OUTPUT_0_MEM_CONT_SIZE,
                           _226_MUL_58_OUTPUT_0_MEM_WRAP_OFFSET,
                           _226_MUL_58_OUTPUT_0_MEM_WRAP_SIZE,
                           _226_MUL_58_OUTPUT_0_MEM_STRIDE,
                           _228_PADCONVDW_8_OUTPUT_0_MEM_CONT_OFFSET,
                           _228_PADCONVDW_8_OUTPUT_0_MEM_CONT_SIZE,
                           _228_PADCONVDW_8_OUTPUT_0_MEM_WRAP_OFFSET,
                           _228_PADCONVDW_8_OUTPUT_0_MEM_WRAP_SIZE,
                           _228_PADCONVDW_8_OUTPUT_0_MEM_STRIDE>
                           (_226_Mul_58_output_0, _228_PadConvDw_8_output_0, _228_PadConvDw_8_weights_output_0, _228_PadConvDw_8_biases_output_0, _228_PADCONVDW_8_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _228_PadConvDw_8\n");
    FILE* _228_PADCONVDW_8_OUTPUT_0_STREAM = fopen("data/export_outputs/_228_PadConvDw_8_output_0.txt", "w");
    export_cpp::saveOutputs<_228_PADCONVDW_8_OUTPUT_0_NB_OUTPUTS,
                            _228_PADCONVDW_8_OUTPUT_0_OUT_HEIGHT,
                            _228_PADCONVDW_8_OUTPUT_0_OUT_WIDTH,
                            _228_PADCONVDW_8_OUTPUT_0_MEM_CONT_OFFSET,
                            _228_PADCONVDW_8_OUTPUT_0_MEM_CONT_SIZE,
                            _228_PADCONVDW_8_OUTPUT_0_MEM_WRAP_OFFSET,
                            _228_PADCONVDW_8_OUTPUT_0_MEM_WRAP_SIZE,
                            _228_PADCONVDW_8_OUTPUT_0_DEV_FMT>
                            (_228_PadConvDw_8_output_0,
                            _228_PADCONVDW_8_OUTPUT_0_STREAM);
    fclose(_228_PADCONVDW_8_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _229_Sigmoid_59_output_0 = (float*) (mem_0 + _229_SIGMOID_59_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_229_SIGMOID_59_NB_ELTS>
                               (_228_PadConvDw_8_output_0, _229_Sigmoid_59_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _229_Sigmoid_59\n");
    FILE* _229_SIGMOID_59_OUTPUT_0_STREAM = fopen("data/export_outputs/_229_Sigmoid_59_output_0.txt", "w");
    export_cpp::saveOutputs<_229_SIGMOID_59_OUTPUT_0_NB_OUTPUTS,
                            _229_SIGMOID_59_OUTPUT_0_OUT_HEIGHT,
                            _229_SIGMOID_59_OUTPUT_0_OUT_WIDTH,
                            _229_SIGMOID_59_OUTPUT_0_MEM_CONT_OFFSET,
                            _229_SIGMOID_59_OUTPUT_0_MEM_CONT_SIZE,
                            _229_SIGMOID_59_OUTPUT_0_MEM_WRAP_OFFSET,
                            _229_SIGMOID_59_OUTPUT_0_MEM_WRAP_SIZE,
                            _229_SIGMOID_59_OUTPUT_0_DEV_FMT>
                            (_229_Sigmoid_59_output_0,
                            _229_SIGMOID_59_OUTPUT_0_STREAM);
    fclose(_229_SIGMOID_59_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _230_Mul_59_output_0 = (float*) (mem_0 + _230_MUL_59_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_230_MUL_59_NB_MAT,
                                 _230_MUL_59_ELEM_OP,
                                 _230_MUL_59_INPUT1_CONT_SIZE,
                                 _230_MUL_59_INPUT2_CONT_SIZE,
                                 _230_MUL_59_OUTPUT_CONT_SIZE,
                                 _230_MUL_59_OFFSET_IN1,
                                 _230_MUL_59_OFFSET_IN2,
                                 _230_MUL_59_ACTIVATION,
                                 _228_PADCONVDW_8_OUTPUT_0_MEM_CONT_OFFSET,
                                 _228_PADCONVDW_8_OUTPUT_0_MEM_CONT_SIZE,
                                 _228_PADCONVDW_8_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _228_PADCONVDW_8_OUTPUT_0_MEM_WRAP_SIZE,
                                 _228_PADCONVDW_8_OUTPUT_0_MEM_STRIDE,
                                 _229_SIGMOID_59_OUTPUT_0_MEM_CONT_OFFSET,
                                 _229_SIGMOID_59_OUTPUT_0_MEM_CONT_SIZE,
                                 _229_SIGMOID_59_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _229_SIGMOID_59_OUTPUT_0_MEM_WRAP_SIZE,
                                 _229_SIGMOID_59_OUTPUT_0_MEM_STRIDE,
                                 _230_MUL_59_OUTPUT_0_MEM_CONT_OFFSET,
                                 _230_MUL_59_OUTPUT_0_MEM_CONT_SIZE,
                                 _230_MUL_59_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _230_MUL_59_OUTPUT_0_MEM_WRAP_SIZE,
                                 _230_MUL_59_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_230_Mul_59_output_0, _230_MUL_59_RESCALING, _228_PadConvDw_8_output_0, _229_Sigmoid_59_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _230_Mul_59\n");
    FILE* _230_MUL_59_OUTPUT_0_STREAM = fopen("data/export_outputs/_230_Mul_59_output_0.txt", "w");
    export_cpp::saveOutputs<_230_MUL_59_OUTPUT_0_NB_OUTPUTS,
                            _230_MUL_59_OUTPUT_0_OUT_HEIGHT,
                            _230_MUL_59_OUTPUT_0_OUT_WIDTH,
                            _230_MUL_59_OUTPUT_0_MEM_CONT_OFFSET,
                            _230_MUL_59_OUTPUT_0_MEM_CONT_SIZE,
                            _230_MUL_59_OUTPUT_0_MEM_WRAP_OFFSET,
                            _230_MUL_59_OUTPUT_0_MEM_WRAP_SIZE,
                            _230_MUL_59_OUTPUT_0_DEV_FMT>
                            (_230_Mul_59_output_0,
                            _230_MUL_59_OUTPUT_0_STREAM);
    fclose(_230_MUL_59_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _231_Conv2D_33_output_0 = (float*) (mem_0 + _231_CONV2D_33_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_230_MUL_59_OUTPUT_0_IN_BATCH,
                           _231_CONV2D_33_NB_CHANNELS,
                           _230_MUL_59_OUTPUT_0_IN_HEIGHT,
                           _230_MUL_59_OUTPUT_0_IN_WIDTH,
                           _231_CONV2D_33_NB_OUTPUTS,
                           _231_CONV2D_33_OUTPUT_0_OUT_HEIGHT,
                           _231_CONV2D_33_OUTPUT_0_OUT_WIDTH,
                           _231_CONV2D_33_PADDING_Y,
                           _231_CONV2D_33_PADDING_X,
                           _231_CONV2D_33_STRIDE_Y,
                           _231_CONV2D_33_STRIDE_X,
                           _231_CONV2D_33_DILATION_Y,
                           _231_CONV2D_33_DILATION_X,
                           _231_CONV2D_33_KERNEL_HEIGHT,
                           _231_CONV2D_33_KERNEL_WIDTH,
                           _231_CONV2D_33_ACTIVATION,
                           _230_MUL_59_OUTPUT_0_MEM_CONT_OFFSET,
                           _230_MUL_59_OUTPUT_0_MEM_CONT_SIZE,
                           _230_MUL_59_OUTPUT_0_MEM_WRAP_OFFSET,
                           _230_MUL_59_OUTPUT_0_MEM_WRAP_SIZE,
                           _230_MUL_59_OUTPUT_0_MEM_STRIDE,
                           _231_CONV2D_33_OUTPUT_0_MEM_CONT_OFFSET,
                           _231_CONV2D_33_OUTPUT_0_MEM_CONT_SIZE,
                           _231_CONV2D_33_OUTPUT_0_MEM_WRAP_OFFSET,
                           _231_CONV2D_33_OUTPUT_0_MEM_WRAP_SIZE,
                           _231_CONV2D_33_OUTPUT_0_MEM_STRIDE>
                           (_230_Mul_59_output_0, _231_Conv2D_33_output_0, _231_Conv2D_33_weights_output_0, _231_Conv2D_33_biases_output_0, _231_CONV2D_33_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _231_Conv2D_33\n");
    FILE* _231_CONV2D_33_OUTPUT_0_STREAM = fopen("data/export_outputs/_231_Conv2D_33_output_0.txt", "w");
    export_cpp::saveOutputs<_231_CONV2D_33_OUTPUT_0_NB_OUTPUTS,
                            _231_CONV2D_33_OUTPUT_0_OUT_HEIGHT,
                            _231_CONV2D_33_OUTPUT_0_OUT_WIDTH,
                            _231_CONV2D_33_OUTPUT_0_MEM_CONT_OFFSET,
                            _231_CONV2D_33_OUTPUT_0_MEM_CONT_SIZE,
                            _231_CONV2D_33_OUTPUT_0_MEM_WRAP_OFFSET,
                            _231_CONV2D_33_OUTPUT_0_MEM_WRAP_SIZE,
                            _231_CONV2D_33_OUTPUT_0_DEV_FMT>
                            (_231_Conv2D_33_output_0,
                            _231_CONV2D_33_OUTPUT_0_STREAM);
    fclose(_231_CONV2D_33_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _232_Sigmoid_60_output_0 = (float*) (mem_0 + _232_SIGMOID_60_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_232_SIGMOID_60_NB_ELTS>
                               (_231_Conv2D_33_output_0, _232_Sigmoid_60_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _232_Sigmoid_60\n");
    FILE* _232_SIGMOID_60_OUTPUT_0_STREAM = fopen("data/export_outputs/_232_Sigmoid_60_output_0.txt", "w");
    export_cpp::saveOutputs<_232_SIGMOID_60_OUTPUT_0_NB_OUTPUTS,
                            _232_SIGMOID_60_OUTPUT_0_OUT_HEIGHT,
                            _232_SIGMOID_60_OUTPUT_0_OUT_WIDTH,
                            _232_SIGMOID_60_OUTPUT_0_MEM_CONT_OFFSET,
                            _232_SIGMOID_60_OUTPUT_0_MEM_CONT_SIZE,
                            _232_SIGMOID_60_OUTPUT_0_MEM_WRAP_OFFSET,
                            _232_SIGMOID_60_OUTPUT_0_MEM_WRAP_SIZE,
                            _232_SIGMOID_60_OUTPUT_0_DEV_FMT>
                            (_232_Sigmoid_60_output_0,
                            _232_SIGMOID_60_OUTPUT_0_STREAM);
    fclose(_232_SIGMOID_60_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _233_Mul_60_output_0 = (float*) (mem_0 + _233_MUL_60_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_233_MUL_60_NB_MAT,
                                 _233_MUL_60_ELEM_OP,
                                 _233_MUL_60_INPUT1_CONT_SIZE,
                                 _233_MUL_60_INPUT2_CONT_SIZE,
                                 _233_MUL_60_OUTPUT_CONT_SIZE,
                                 _233_MUL_60_OFFSET_IN1,
                                 _233_MUL_60_OFFSET_IN2,
                                 _233_MUL_60_ACTIVATION,
                                 _231_CONV2D_33_OUTPUT_0_MEM_CONT_OFFSET,
                                 _231_CONV2D_33_OUTPUT_0_MEM_CONT_SIZE,
                                 _231_CONV2D_33_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _231_CONV2D_33_OUTPUT_0_MEM_WRAP_SIZE,
                                 _231_CONV2D_33_OUTPUT_0_MEM_STRIDE,
                                 _232_SIGMOID_60_OUTPUT_0_MEM_CONT_OFFSET,
                                 _232_SIGMOID_60_OUTPUT_0_MEM_CONT_SIZE,
                                 _232_SIGMOID_60_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _232_SIGMOID_60_OUTPUT_0_MEM_WRAP_SIZE,
                                 _232_SIGMOID_60_OUTPUT_0_MEM_STRIDE,
                                 _233_MUL_60_OUTPUT_0_MEM_CONT_OFFSET,
                                 _233_MUL_60_OUTPUT_0_MEM_CONT_SIZE,
                                 _233_MUL_60_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _233_MUL_60_OUTPUT_0_MEM_WRAP_SIZE,
                                 _233_MUL_60_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_233_Mul_60_output_0, _233_MUL_60_RESCALING, _231_Conv2D_33_output_0, _232_Sigmoid_60_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _233_Mul_60\n");
    FILE* _233_MUL_60_OUTPUT_0_STREAM = fopen("data/export_outputs/_233_Mul_60_output_0.txt", "w");
    export_cpp::saveOutputs<_233_MUL_60_OUTPUT_0_NB_OUTPUTS,
                            _233_MUL_60_OUTPUT_0_OUT_HEIGHT,
                            _233_MUL_60_OUTPUT_0_OUT_WIDTH,
                            _233_MUL_60_OUTPUT_0_MEM_CONT_OFFSET,
                            _233_MUL_60_OUTPUT_0_MEM_CONT_SIZE,
                            _233_MUL_60_OUTPUT_0_MEM_WRAP_OFFSET,
                            _233_MUL_60_OUTPUT_0_MEM_WRAP_SIZE,
                            _233_MUL_60_OUTPUT_0_DEV_FMT>
                            (_233_Mul_60_output_0,
                            _233_MUL_60_OUTPUT_0_STREAM);
    fclose(_233_MUL_60_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _234_PadConvDw_9_output_0 = (float*) (mem_0 + _234_PADCONVDW_9_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_233_MUL_60_OUTPUT_0_IN_BATCH,
                           _234_PADCONVDW_9_NB_CHANNELS,
                           _233_MUL_60_OUTPUT_0_IN_HEIGHT,
                           _233_MUL_60_OUTPUT_0_IN_WIDTH,
                           _234_PADCONVDW_9_NB_OUTPUTS,
                           _234_PADCONVDW_9_OUTPUT_0_OUT_HEIGHT,
                           _234_PADCONVDW_9_OUTPUT_0_OUT_WIDTH,
                           _234_PADCONVDW_9_PADDING_Y,
                           _234_PADCONVDW_9_PADDING_X,
                           _234_PADCONVDW_9_STRIDE_Y,
                           _234_PADCONVDW_9_STRIDE_X,
                           _234_PADCONVDW_9_DILATION_Y,
                           _234_PADCONVDW_9_DILATION_X,
                           _234_PADCONVDW_9_KERNEL_HEIGHT,
                           _234_PADCONVDW_9_KERNEL_WIDTH,
                           _234_PADCONVDW_9_ACTIVATION,
                           _233_MUL_60_OUTPUT_0_MEM_CONT_OFFSET,
                           _233_MUL_60_OUTPUT_0_MEM_CONT_SIZE,
                           _233_MUL_60_OUTPUT_0_MEM_WRAP_OFFSET,
                           _233_MUL_60_OUTPUT_0_MEM_WRAP_SIZE,
                           _233_MUL_60_OUTPUT_0_MEM_STRIDE,
                           _234_PADCONVDW_9_OUTPUT_0_MEM_CONT_OFFSET,
                           _234_PADCONVDW_9_OUTPUT_0_MEM_CONT_SIZE,
                           _234_PADCONVDW_9_OUTPUT_0_MEM_WRAP_OFFSET,
                           _234_PADCONVDW_9_OUTPUT_0_MEM_WRAP_SIZE,
                           _234_PADCONVDW_9_OUTPUT_0_MEM_STRIDE>
                           (_233_Mul_60_output_0, _234_PadConvDw_9_output_0, _234_PadConvDw_9_weights_output_0, _234_PadConvDw_9_biases_output_0, _234_PADCONVDW_9_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _234_PadConvDw_9\n");
    FILE* _234_PADCONVDW_9_OUTPUT_0_STREAM = fopen("data/export_outputs/_234_PadConvDw_9_output_0.txt", "w");
    export_cpp::saveOutputs<_234_PADCONVDW_9_OUTPUT_0_NB_OUTPUTS,
                            _234_PADCONVDW_9_OUTPUT_0_OUT_HEIGHT,
                            _234_PADCONVDW_9_OUTPUT_0_OUT_WIDTH,
                            _234_PADCONVDW_9_OUTPUT_0_MEM_CONT_OFFSET,
                            _234_PADCONVDW_9_OUTPUT_0_MEM_CONT_SIZE,
                            _234_PADCONVDW_9_OUTPUT_0_MEM_WRAP_OFFSET,
                            _234_PADCONVDW_9_OUTPUT_0_MEM_WRAP_SIZE,
                            _234_PADCONVDW_9_OUTPUT_0_DEV_FMT>
                            (_234_PadConvDw_9_output_0,
                            _234_PADCONVDW_9_OUTPUT_0_STREAM);
    fclose(_234_PADCONVDW_9_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _235_Sigmoid_61_output_0 = (float*) (mem_0 + _235_SIGMOID_61_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_235_SIGMOID_61_NB_ELTS>
                               (_234_PadConvDw_9_output_0, _235_Sigmoid_61_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _235_Sigmoid_61\n");
    FILE* _235_SIGMOID_61_OUTPUT_0_STREAM = fopen("data/export_outputs/_235_Sigmoid_61_output_0.txt", "w");
    export_cpp::saveOutputs<_235_SIGMOID_61_OUTPUT_0_NB_OUTPUTS,
                            _235_SIGMOID_61_OUTPUT_0_OUT_HEIGHT,
                            _235_SIGMOID_61_OUTPUT_0_OUT_WIDTH,
                            _235_SIGMOID_61_OUTPUT_0_MEM_CONT_OFFSET,
                            _235_SIGMOID_61_OUTPUT_0_MEM_CONT_SIZE,
                            _235_SIGMOID_61_OUTPUT_0_MEM_WRAP_OFFSET,
                            _235_SIGMOID_61_OUTPUT_0_MEM_WRAP_SIZE,
                            _235_SIGMOID_61_OUTPUT_0_DEV_FMT>
                            (_235_Sigmoid_61_output_0,
                            _235_SIGMOID_61_OUTPUT_0_STREAM);
    fclose(_235_SIGMOID_61_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _236_Mul_61_output_0 = (float*) (mem_0 + _236_MUL_61_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_236_MUL_61_NB_MAT,
                                 _236_MUL_61_ELEM_OP,
                                 _236_MUL_61_INPUT1_CONT_SIZE,
                                 _236_MUL_61_INPUT2_CONT_SIZE,
                                 _236_MUL_61_OUTPUT_CONT_SIZE,
                                 _236_MUL_61_OFFSET_IN1,
                                 _236_MUL_61_OFFSET_IN2,
                                 _236_MUL_61_ACTIVATION,
                                 _234_PADCONVDW_9_OUTPUT_0_MEM_CONT_OFFSET,
                                 _234_PADCONVDW_9_OUTPUT_0_MEM_CONT_SIZE,
                                 _234_PADCONVDW_9_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _234_PADCONVDW_9_OUTPUT_0_MEM_WRAP_SIZE,
                                 _234_PADCONVDW_9_OUTPUT_0_MEM_STRIDE,
                                 _235_SIGMOID_61_OUTPUT_0_MEM_CONT_OFFSET,
                                 _235_SIGMOID_61_OUTPUT_0_MEM_CONT_SIZE,
                                 _235_SIGMOID_61_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _235_SIGMOID_61_OUTPUT_0_MEM_WRAP_SIZE,
                                 _235_SIGMOID_61_OUTPUT_0_MEM_STRIDE,
                                 _236_MUL_61_OUTPUT_0_MEM_CONT_OFFSET,
                                 _236_MUL_61_OUTPUT_0_MEM_CONT_SIZE,
                                 _236_MUL_61_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _236_MUL_61_OUTPUT_0_MEM_WRAP_SIZE,
                                 _236_MUL_61_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_236_Mul_61_output_0, _236_MUL_61_RESCALING, _234_PadConvDw_9_output_0, _235_Sigmoid_61_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _236_Mul_61\n");
    FILE* _236_MUL_61_OUTPUT_0_STREAM = fopen("data/export_outputs/_236_Mul_61_output_0.txt", "w");
    export_cpp::saveOutputs<_236_MUL_61_OUTPUT_0_NB_OUTPUTS,
                            _236_MUL_61_OUTPUT_0_OUT_HEIGHT,
                            _236_MUL_61_OUTPUT_0_OUT_WIDTH,
                            _236_MUL_61_OUTPUT_0_MEM_CONT_OFFSET,
                            _236_MUL_61_OUTPUT_0_MEM_CONT_SIZE,
                            _236_MUL_61_OUTPUT_0_MEM_WRAP_OFFSET,
                            _236_MUL_61_OUTPUT_0_MEM_WRAP_SIZE,
                            _236_MUL_61_OUTPUT_0_DEV_FMT>
                            (_236_Mul_61_output_0,
                            _236_MUL_61_OUTPUT_0_STREAM);
    fclose(_236_MUL_61_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _237_Add_8_output_0 = (float*) (mem_0 + _237_ADD_8_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_237_ADD_8_NB_MAT,
                                 _237_ADD_8_ELEM_OP,
                                 _237_ADD_8_INPUT1_CONT_SIZE,
                                 _237_ADD_8_INPUT2_CONT_SIZE,
                                 _237_ADD_8_OUTPUT_CONT_SIZE,
                                 _237_ADD_8_OFFSET_IN1,
                                 _237_ADD_8_OFFSET_IN2,
                                 _237_ADD_8_ACTIVATION,
                                 _213_SPLIT_8_OUTPUT_1_MEM_CONT_OFFSET,
                                 _213_SPLIT_8_OUTPUT_1_MEM_CONT_SIZE,
                                 _213_SPLIT_8_OUTPUT_1_MEM_WRAP_OFFSET,
                                 _213_SPLIT_8_OUTPUT_1_MEM_WRAP_SIZE,
                                 _213_SPLIT_8_OUTPUT_1_MEM_STRIDE,
                                 _236_MUL_61_OUTPUT_0_MEM_CONT_OFFSET,
                                 _236_MUL_61_OUTPUT_0_MEM_CONT_SIZE,
                                 _236_MUL_61_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _236_MUL_61_OUTPUT_0_MEM_WRAP_SIZE,
                                 _236_MUL_61_OUTPUT_0_MEM_STRIDE,
                                 _237_ADD_8_OUTPUT_0_MEM_CONT_OFFSET,
                                 _237_ADD_8_OUTPUT_0_MEM_CONT_SIZE,
                                 _237_ADD_8_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _237_ADD_8_OUTPUT_0_MEM_WRAP_SIZE,
                                 _237_ADD_8_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_237_Add_8_output_0, _237_ADD_8_RESCALING, _213_Split_8_output_1, _236_Mul_61_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _237_Add_8\n");
    FILE* _237_ADD_8_OUTPUT_0_STREAM = fopen("data/export_outputs/_237_Add_8_output_0.txt", "w");
    export_cpp::saveOutputs<_237_ADD_8_OUTPUT_0_NB_OUTPUTS,
                            _237_ADD_8_OUTPUT_0_OUT_HEIGHT,
                            _237_ADD_8_OUTPUT_0_OUT_WIDTH,
                            _237_ADD_8_OUTPUT_0_MEM_CONT_OFFSET,
                            _237_ADD_8_OUTPUT_0_MEM_CONT_SIZE,
                            _237_ADD_8_OUTPUT_0_MEM_WRAP_OFFSET,
                            _237_ADD_8_OUTPUT_0_MEM_WRAP_SIZE,
                            _237_ADD_8_OUTPUT_0_DEV_FMT>
                            (_237_Add_8_output_0,
                            _237_ADD_8_OUTPUT_0_STREAM);
    fclose(_237_ADD_8_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _238_Concat_13_output_0 = (float*) (mem_0 + _238_CONCAT_13_OUTPUT_0_MEM_OFFSET);

    const float* _238_Concat_13_inputs[] = { _213_Split_8_output_0, _213_Split_8_output_1, _237_Add_8_output_0 };
    export_cpp::concat_forward<_238_CONCAT_13_AXIS_SIZE_POST,
                               _238_CONCAT_13_AXIS_SIZE_PRE,
                               _238_CONCAT_13_AXIS_SIZE,
                               _238_CONCAT_13_TOTAL_AXIS_SIZE,
                               3> 
                              (_238_Concat_13_inputs,
                               _238_Concat_13_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _238_Concat_13\n");
    FILE* _238_CONCAT_13_OUTPUT_0_STREAM = fopen("data/export_outputs/_238_Concat_13_output_0.txt", "w");
    export_cpp::saveOutputs<_238_CONCAT_13_OUTPUT_0_NB_OUTPUTS,
                            _238_CONCAT_13_OUTPUT_0_OUT_HEIGHT,
                            _238_CONCAT_13_OUTPUT_0_OUT_WIDTH,
                            _238_CONCAT_13_OUTPUT_0_MEM_CONT_OFFSET,
                            _238_CONCAT_13_OUTPUT_0_MEM_CONT_SIZE,
                            _238_CONCAT_13_OUTPUT_0_MEM_WRAP_OFFSET,
                            _238_CONCAT_13_OUTPUT_0_MEM_WRAP_SIZE,
                            _238_CONCAT_13_OUTPUT_0_DEV_FMT>
                            (_238_Concat_13_output_0,
                            _238_CONCAT_13_OUTPUT_0_STREAM);
    fclose(_238_CONCAT_13_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _239_Conv2D_34_output_0 = (float*) (mem_0 + _239_CONV2D_34_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_238_CONCAT_13_OUTPUT_0_IN_BATCH,
                           _239_CONV2D_34_NB_CHANNELS,
                           _238_CONCAT_13_OUTPUT_0_IN_HEIGHT,
                           _238_CONCAT_13_OUTPUT_0_IN_WIDTH,
                           _239_CONV2D_34_NB_OUTPUTS,
                           _239_CONV2D_34_OUTPUT_0_OUT_HEIGHT,
                           _239_CONV2D_34_OUTPUT_0_OUT_WIDTH,
                           _239_CONV2D_34_PADDING_Y,
                           _239_CONV2D_34_PADDING_X,
                           _239_CONV2D_34_STRIDE_Y,
                           _239_CONV2D_34_STRIDE_X,
                           _239_CONV2D_34_DILATION_Y,
                           _239_CONV2D_34_DILATION_X,
                           _239_CONV2D_34_KERNEL_HEIGHT,
                           _239_CONV2D_34_KERNEL_WIDTH,
                           _239_CONV2D_34_ACTIVATION,
                           _238_CONCAT_13_OUTPUT_0_MEM_CONT_OFFSET,
                           _238_CONCAT_13_OUTPUT_0_MEM_CONT_SIZE,
                           _238_CONCAT_13_OUTPUT_0_MEM_WRAP_OFFSET,
                           _238_CONCAT_13_OUTPUT_0_MEM_WRAP_SIZE,
                           _238_CONCAT_13_OUTPUT_0_MEM_STRIDE,
                           _239_CONV2D_34_OUTPUT_0_MEM_CONT_OFFSET,
                           _239_CONV2D_34_OUTPUT_0_MEM_CONT_SIZE,
                           _239_CONV2D_34_OUTPUT_0_MEM_WRAP_OFFSET,
                           _239_CONV2D_34_OUTPUT_0_MEM_WRAP_SIZE,
                           _239_CONV2D_34_OUTPUT_0_MEM_STRIDE>
                           (_238_Concat_13_output_0, _239_Conv2D_34_output_0, _239_Conv2D_34_weights_output_0, _239_Conv2D_34_biases_output_0, _239_CONV2D_34_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _239_Conv2D_34\n");
    FILE* _239_CONV2D_34_OUTPUT_0_STREAM = fopen("data/export_outputs/_239_Conv2D_34_output_0.txt", "w");
    export_cpp::saveOutputs<_239_CONV2D_34_OUTPUT_0_NB_OUTPUTS,
                            _239_CONV2D_34_OUTPUT_0_OUT_HEIGHT,
                            _239_CONV2D_34_OUTPUT_0_OUT_WIDTH,
                            _239_CONV2D_34_OUTPUT_0_MEM_CONT_OFFSET,
                            _239_CONV2D_34_OUTPUT_0_MEM_CONT_SIZE,
                            _239_CONV2D_34_OUTPUT_0_MEM_WRAP_OFFSET,
                            _239_CONV2D_34_OUTPUT_0_MEM_WRAP_SIZE,
                            _239_CONV2D_34_OUTPUT_0_DEV_FMT>
                            (_239_Conv2D_34_output_0,
                            _239_CONV2D_34_OUTPUT_0_STREAM);
    fclose(_239_CONV2D_34_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _240_Sigmoid_62_output_0 = (float*) (mem_0 + _240_SIGMOID_62_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_240_SIGMOID_62_NB_ELTS>
                               (_239_Conv2D_34_output_0, _240_Sigmoid_62_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _240_Sigmoid_62\n");
    FILE* _240_SIGMOID_62_OUTPUT_0_STREAM = fopen("data/export_outputs/_240_Sigmoid_62_output_0.txt", "w");
    export_cpp::saveOutputs<_240_SIGMOID_62_OUTPUT_0_NB_OUTPUTS,
                            _240_SIGMOID_62_OUTPUT_0_OUT_HEIGHT,
                            _240_SIGMOID_62_OUTPUT_0_OUT_WIDTH,
                            _240_SIGMOID_62_OUTPUT_0_MEM_CONT_OFFSET,
                            _240_SIGMOID_62_OUTPUT_0_MEM_CONT_SIZE,
                            _240_SIGMOID_62_OUTPUT_0_MEM_WRAP_OFFSET,
                            _240_SIGMOID_62_OUTPUT_0_MEM_WRAP_SIZE,
                            _240_SIGMOID_62_OUTPUT_0_DEV_FMT>
                            (_240_Sigmoid_62_output_0,
                            _240_SIGMOID_62_OUTPUT_0_STREAM);
    fclose(_240_SIGMOID_62_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _241_Mul_62_output_0 = (float*) (mem_0 + _241_MUL_62_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_241_MUL_62_NB_MAT,
                                 _241_MUL_62_ELEM_OP,
                                 _241_MUL_62_INPUT1_CONT_SIZE,
                                 _241_MUL_62_INPUT2_CONT_SIZE,
                                 _241_MUL_62_OUTPUT_CONT_SIZE,
                                 _241_MUL_62_OFFSET_IN1,
                                 _241_MUL_62_OFFSET_IN2,
                                 _241_MUL_62_ACTIVATION,
                                 _239_CONV2D_34_OUTPUT_0_MEM_CONT_OFFSET,
                                 _239_CONV2D_34_OUTPUT_0_MEM_CONT_SIZE,
                                 _239_CONV2D_34_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _239_CONV2D_34_OUTPUT_0_MEM_WRAP_SIZE,
                                 _239_CONV2D_34_OUTPUT_0_MEM_STRIDE,
                                 _240_SIGMOID_62_OUTPUT_0_MEM_CONT_OFFSET,
                                 _240_SIGMOID_62_OUTPUT_0_MEM_CONT_SIZE,
                                 _240_SIGMOID_62_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _240_SIGMOID_62_OUTPUT_0_MEM_WRAP_SIZE,
                                 _240_SIGMOID_62_OUTPUT_0_MEM_STRIDE,
                                 _241_MUL_62_OUTPUT_0_MEM_CONT_OFFSET,
                                 _241_MUL_62_OUTPUT_0_MEM_CONT_SIZE,
                                 _241_MUL_62_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _241_MUL_62_OUTPUT_0_MEM_WRAP_SIZE,
                                 _241_MUL_62_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_241_Mul_62_output_0, _241_MUL_62_RESCALING, _239_Conv2D_34_output_0, _240_Sigmoid_62_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _241_Mul_62\n");
    FILE* _241_MUL_62_OUTPUT_0_STREAM = fopen("data/export_outputs/_241_Mul_62_output_0.txt", "w");
    export_cpp::saveOutputs<_241_MUL_62_OUTPUT_0_NB_OUTPUTS,
                            _241_MUL_62_OUTPUT_0_OUT_HEIGHT,
                            _241_MUL_62_OUTPUT_0_OUT_WIDTH,
                            _241_MUL_62_OUTPUT_0_MEM_CONT_OFFSET,
                            _241_MUL_62_OUTPUT_0_MEM_CONT_SIZE,
                            _241_MUL_62_OUTPUT_0_MEM_WRAP_OFFSET,
                            _241_MUL_62_OUTPUT_0_MEM_WRAP_SIZE,
                            _241_MUL_62_OUTPUT_0_DEV_FMT>
                            (_241_Mul_62_output_0,
                            _241_MUL_62_OUTPUT_0_STREAM);
    fclose(_241_MUL_62_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _242_PadConv_26_output_0 = (float*) (mem_0 + _242_PADCONV_26_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_241_MUL_62_OUTPUT_0_IN_BATCH,
                           _242_PADCONV_26_NB_CHANNELS,
                           _241_MUL_62_OUTPUT_0_IN_HEIGHT,
                           _241_MUL_62_OUTPUT_0_IN_WIDTH,
                           _242_PADCONV_26_NB_OUTPUTS,
                           _242_PADCONV_26_OUTPUT_0_OUT_HEIGHT,
                           _242_PADCONV_26_OUTPUT_0_OUT_WIDTH,
                           _242_PADCONV_26_PADDING_Y,
                           _242_PADCONV_26_PADDING_X,
                           _242_PADCONV_26_STRIDE_Y,
                           _242_PADCONV_26_STRIDE_X,
                           _242_PADCONV_26_DILATION_Y,
                           _242_PADCONV_26_DILATION_X,
                           _242_PADCONV_26_KERNEL_HEIGHT,
                           _242_PADCONV_26_KERNEL_WIDTH,
                           _242_PADCONV_26_ACTIVATION,
                           _241_MUL_62_OUTPUT_0_MEM_CONT_OFFSET,
                           _241_MUL_62_OUTPUT_0_MEM_CONT_SIZE,
                           _241_MUL_62_OUTPUT_0_MEM_WRAP_OFFSET,
                           _241_MUL_62_OUTPUT_0_MEM_WRAP_SIZE,
                           _241_MUL_62_OUTPUT_0_MEM_STRIDE,
                           _242_PADCONV_26_OUTPUT_0_MEM_CONT_OFFSET,
                           _242_PADCONV_26_OUTPUT_0_MEM_CONT_SIZE,
                           _242_PADCONV_26_OUTPUT_0_MEM_WRAP_OFFSET,
                           _242_PADCONV_26_OUTPUT_0_MEM_WRAP_SIZE,
                           _242_PADCONV_26_OUTPUT_0_MEM_STRIDE>
                           (_241_Mul_62_output_0, _242_PadConv_26_output_0, _242_PadConv_26_weights_output_0, _242_PadConv_26_biases_output_0, _242_PADCONV_26_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _242_PadConv_26\n");
    FILE* _242_PADCONV_26_OUTPUT_0_STREAM = fopen("data/export_outputs/_242_PadConv_26_output_0.txt", "w");
    export_cpp::saveOutputs<_242_PADCONV_26_OUTPUT_0_NB_OUTPUTS,
                            _242_PADCONV_26_OUTPUT_0_OUT_HEIGHT,
                            _242_PADCONV_26_OUTPUT_0_OUT_WIDTH,
                            _242_PADCONV_26_OUTPUT_0_MEM_CONT_OFFSET,
                            _242_PADCONV_26_OUTPUT_0_MEM_CONT_SIZE,
                            _242_PADCONV_26_OUTPUT_0_MEM_WRAP_OFFSET,
                            _242_PADCONV_26_OUTPUT_0_MEM_WRAP_SIZE,
                            _242_PADCONV_26_OUTPUT_0_DEV_FMT>
                            (_242_PadConv_26_output_0,
                            _242_PADCONV_26_OUTPUT_0_STREAM);
    fclose(_242_PADCONV_26_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _243_PadConvDw_10_output_0 = (float*) (mem_0 + _243_PADCONVDW_10_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_241_MUL_62_OUTPUT_0_IN_BATCH,
                           _243_PADCONVDW_10_NB_CHANNELS,
                           _241_MUL_62_OUTPUT_0_IN_HEIGHT,
                           _241_MUL_62_OUTPUT_0_IN_WIDTH,
                           _243_PADCONVDW_10_NB_OUTPUTS,
                           _243_PADCONVDW_10_OUTPUT_0_OUT_HEIGHT,
                           _243_PADCONVDW_10_OUTPUT_0_OUT_WIDTH,
                           _243_PADCONVDW_10_PADDING_Y,
                           _243_PADCONVDW_10_PADDING_X,
                           _243_PADCONVDW_10_STRIDE_Y,
                           _243_PADCONVDW_10_STRIDE_X,
                           _243_PADCONVDW_10_DILATION_Y,
                           _243_PADCONVDW_10_DILATION_X,
                           _243_PADCONVDW_10_KERNEL_HEIGHT,
                           _243_PADCONVDW_10_KERNEL_WIDTH,
                           _243_PADCONVDW_10_ACTIVATION,
                           _241_MUL_62_OUTPUT_0_MEM_CONT_OFFSET,
                           _241_MUL_62_OUTPUT_0_MEM_CONT_SIZE,
                           _241_MUL_62_OUTPUT_0_MEM_WRAP_OFFSET,
                           _241_MUL_62_OUTPUT_0_MEM_WRAP_SIZE,
                           _241_MUL_62_OUTPUT_0_MEM_STRIDE,
                           _243_PADCONVDW_10_OUTPUT_0_MEM_CONT_OFFSET,
                           _243_PADCONVDW_10_OUTPUT_0_MEM_CONT_SIZE,
                           _243_PADCONVDW_10_OUTPUT_0_MEM_WRAP_OFFSET,
                           _243_PADCONVDW_10_OUTPUT_0_MEM_WRAP_SIZE,
                           _243_PADCONVDW_10_OUTPUT_0_MEM_STRIDE>
                           (_241_Mul_62_output_0, _243_PadConvDw_10_output_0, _243_PadConvDw_10_weights_output_0, _243_PadConvDw_10_biases_output_0, _243_PADCONVDW_10_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _243_PadConvDw_10\n");
    FILE* _243_PADCONVDW_10_OUTPUT_0_STREAM = fopen("data/export_outputs/_243_PadConvDw_10_output_0.txt", "w");
    export_cpp::saveOutputs<_243_PADCONVDW_10_OUTPUT_0_NB_OUTPUTS,
                            _243_PADCONVDW_10_OUTPUT_0_OUT_HEIGHT,
                            _243_PADCONVDW_10_OUTPUT_0_OUT_WIDTH,
                            _243_PADCONVDW_10_OUTPUT_0_MEM_CONT_OFFSET,
                            _243_PADCONVDW_10_OUTPUT_0_MEM_CONT_SIZE,
                            _243_PADCONVDW_10_OUTPUT_0_MEM_WRAP_OFFSET,
                            _243_PADCONVDW_10_OUTPUT_0_MEM_WRAP_SIZE,
                            _243_PADCONVDW_10_OUTPUT_0_DEV_FMT>
                            (_243_PadConvDw_10_output_0,
                            _243_PADCONVDW_10_OUTPUT_0_STREAM);
    fclose(_243_PADCONVDW_10_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _244_Sigmoid_63_output_0 = (float*) (mem_0 + _244_SIGMOID_63_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_244_SIGMOID_63_NB_ELTS>
                               (_242_PadConv_26_output_0, _244_Sigmoid_63_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _244_Sigmoid_63\n");
    FILE* _244_SIGMOID_63_OUTPUT_0_STREAM = fopen("data/export_outputs/_244_Sigmoid_63_output_0.txt", "w");
    export_cpp::saveOutputs<_244_SIGMOID_63_OUTPUT_0_NB_OUTPUTS,
                            _244_SIGMOID_63_OUTPUT_0_OUT_HEIGHT,
                            _244_SIGMOID_63_OUTPUT_0_OUT_WIDTH,
                            _244_SIGMOID_63_OUTPUT_0_MEM_CONT_OFFSET,
                            _244_SIGMOID_63_OUTPUT_0_MEM_CONT_SIZE,
                            _244_SIGMOID_63_OUTPUT_0_MEM_WRAP_OFFSET,
                            _244_SIGMOID_63_OUTPUT_0_MEM_WRAP_SIZE,
                            _244_SIGMOID_63_OUTPUT_0_DEV_FMT>
                            (_244_Sigmoid_63_output_0,
                            _244_SIGMOID_63_OUTPUT_0_STREAM);
    fclose(_244_SIGMOID_63_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _245_Sigmoid_64_output_0 = (float*) (mem_0 + _245_SIGMOID_64_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_245_SIGMOID_64_NB_ELTS>
                               (_243_PadConvDw_10_output_0, _245_Sigmoid_64_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _245_Sigmoid_64\n");
    FILE* _245_SIGMOID_64_OUTPUT_0_STREAM = fopen("data/export_outputs/_245_Sigmoid_64_output_0.txt", "w");
    export_cpp::saveOutputs<_245_SIGMOID_64_OUTPUT_0_NB_OUTPUTS,
                            _245_SIGMOID_64_OUTPUT_0_OUT_HEIGHT,
                            _245_SIGMOID_64_OUTPUT_0_OUT_WIDTH,
                            _245_SIGMOID_64_OUTPUT_0_MEM_CONT_OFFSET,
                            _245_SIGMOID_64_OUTPUT_0_MEM_CONT_SIZE,
                            _245_SIGMOID_64_OUTPUT_0_MEM_WRAP_OFFSET,
                            _245_SIGMOID_64_OUTPUT_0_MEM_WRAP_SIZE,
                            _245_SIGMOID_64_OUTPUT_0_DEV_FMT>
                            (_245_Sigmoid_64_output_0,
                            _245_SIGMOID_64_OUTPUT_0_STREAM);
    fclose(_245_SIGMOID_64_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _246_Mul_63_output_0 = (float*) (mem_0 + _246_MUL_63_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_246_MUL_63_NB_MAT,
                                 _246_MUL_63_ELEM_OP,
                                 _246_MUL_63_INPUT1_CONT_SIZE,
                                 _246_MUL_63_INPUT2_CONT_SIZE,
                                 _246_MUL_63_OUTPUT_CONT_SIZE,
                                 _246_MUL_63_OFFSET_IN1,
                                 _246_MUL_63_OFFSET_IN2,
                                 _246_MUL_63_ACTIVATION,
                                 _242_PADCONV_26_OUTPUT_0_MEM_CONT_OFFSET,
                                 _242_PADCONV_26_OUTPUT_0_MEM_CONT_SIZE,
                                 _242_PADCONV_26_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _242_PADCONV_26_OUTPUT_0_MEM_WRAP_SIZE,
                                 _242_PADCONV_26_OUTPUT_0_MEM_STRIDE,
                                 _244_SIGMOID_63_OUTPUT_0_MEM_CONT_OFFSET,
                                 _244_SIGMOID_63_OUTPUT_0_MEM_CONT_SIZE,
                                 _244_SIGMOID_63_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _244_SIGMOID_63_OUTPUT_0_MEM_WRAP_SIZE,
                                 _244_SIGMOID_63_OUTPUT_0_MEM_STRIDE,
                                 _246_MUL_63_OUTPUT_0_MEM_CONT_OFFSET,
                                 _246_MUL_63_OUTPUT_0_MEM_CONT_SIZE,
                                 _246_MUL_63_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _246_MUL_63_OUTPUT_0_MEM_WRAP_SIZE,
                                 _246_MUL_63_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_246_Mul_63_output_0, _246_MUL_63_RESCALING, _242_PadConv_26_output_0, _244_Sigmoid_63_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _246_Mul_63\n");
    FILE* _246_MUL_63_OUTPUT_0_STREAM = fopen("data/export_outputs/_246_Mul_63_output_0.txt", "w");
    export_cpp::saveOutputs<_246_MUL_63_OUTPUT_0_NB_OUTPUTS,
                            _246_MUL_63_OUTPUT_0_OUT_HEIGHT,
                            _246_MUL_63_OUTPUT_0_OUT_WIDTH,
                            _246_MUL_63_OUTPUT_0_MEM_CONT_OFFSET,
                            _246_MUL_63_OUTPUT_0_MEM_CONT_SIZE,
                            _246_MUL_63_OUTPUT_0_MEM_WRAP_OFFSET,
                            _246_MUL_63_OUTPUT_0_MEM_WRAP_SIZE,
                            _246_MUL_63_OUTPUT_0_DEV_FMT>
                            (_246_Mul_63_output_0,
                            _246_MUL_63_OUTPUT_0_STREAM);
    fclose(_246_MUL_63_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _247_Mul_64_output_0 = (float*) (mem_0 + _247_MUL_64_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_247_MUL_64_NB_MAT,
                                 _247_MUL_64_ELEM_OP,
                                 _247_MUL_64_INPUT1_CONT_SIZE,
                                 _247_MUL_64_INPUT2_CONT_SIZE,
                                 _247_MUL_64_OUTPUT_CONT_SIZE,
                                 _247_MUL_64_OFFSET_IN1,
                                 _247_MUL_64_OFFSET_IN2,
                                 _247_MUL_64_ACTIVATION,
                                 _243_PADCONVDW_10_OUTPUT_0_MEM_CONT_OFFSET,
                                 _243_PADCONVDW_10_OUTPUT_0_MEM_CONT_SIZE,
                                 _243_PADCONVDW_10_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _243_PADCONVDW_10_OUTPUT_0_MEM_WRAP_SIZE,
                                 _243_PADCONVDW_10_OUTPUT_0_MEM_STRIDE,
                                 _245_SIGMOID_64_OUTPUT_0_MEM_CONT_OFFSET,
                                 _245_SIGMOID_64_OUTPUT_0_MEM_CONT_SIZE,
                                 _245_SIGMOID_64_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _245_SIGMOID_64_OUTPUT_0_MEM_WRAP_SIZE,
                                 _245_SIGMOID_64_OUTPUT_0_MEM_STRIDE,
                                 _247_MUL_64_OUTPUT_0_MEM_CONT_OFFSET,
                                 _247_MUL_64_OUTPUT_0_MEM_CONT_SIZE,
                                 _247_MUL_64_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _247_MUL_64_OUTPUT_0_MEM_WRAP_SIZE,
                                 _247_MUL_64_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_247_Mul_64_output_0, _247_MUL_64_RESCALING, _243_PadConvDw_10_output_0, _245_Sigmoid_64_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _247_Mul_64\n");
    FILE* _247_MUL_64_OUTPUT_0_STREAM = fopen("data/export_outputs/_247_Mul_64_output_0.txt", "w");
    export_cpp::saveOutputs<_247_MUL_64_OUTPUT_0_NB_OUTPUTS,
                            _247_MUL_64_OUTPUT_0_OUT_HEIGHT,
                            _247_MUL_64_OUTPUT_0_OUT_WIDTH,
                            _247_MUL_64_OUTPUT_0_MEM_CONT_OFFSET,
                            _247_MUL_64_OUTPUT_0_MEM_CONT_SIZE,
                            _247_MUL_64_OUTPUT_0_MEM_WRAP_OFFSET,
                            _247_MUL_64_OUTPUT_0_MEM_WRAP_SIZE,
                            _247_MUL_64_OUTPUT_0_DEV_FMT>
                            (_247_Mul_64_output_0,
                            _247_MUL_64_OUTPUT_0_STREAM);
    fclose(_247_MUL_64_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _248_Conv2D_35_output_0 = (float*) (mem_0 + _248_CONV2D_35_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_247_MUL_64_OUTPUT_0_IN_BATCH,
                           _248_CONV2D_35_NB_CHANNELS,
                           _247_MUL_64_OUTPUT_0_IN_HEIGHT,
                           _247_MUL_64_OUTPUT_0_IN_WIDTH,
                           _248_CONV2D_35_NB_OUTPUTS,
                           _248_CONV2D_35_OUTPUT_0_OUT_HEIGHT,
                           _248_CONV2D_35_OUTPUT_0_OUT_WIDTH,
                           _248_CONV2D_35_PADDING_Y,
                           _248_CONV2D_35_PADDING_X,
                           _248_CONV2D_35_STRIDE_Y,
                           _248_CONV2D_35_STRIDE_X,
                           _248_CONV2D_35_DILATION_Y,
                           _248_CONV2D_35_DILATION_X,
                           _248_CONV2D_35_KERNEL_HEIGHT,
                           _248_CONV2D_35_KERNEL_WIDTH,
                           _248_CONV2D_35_ACTIVATION,
                           _247_MUL_64_OUTPUT_0_MEM_CONT_OFFSET,
                           _247_MUL_64_OUTPUT_0_MEM_CONT_SIZE,
                           _247_MUL_64_OUTPUT_0_MEM_WRAP_OFFSET,
                           _247_MUL_64_OUTPUT_0_MEM_WRAP_SIZE,
                           _247_MUL_64_OUTPUT_0_MEM_STRIDE,
                           _248_CONV2D_35_OUTPUT_0_MEM_CONT_OFFSET,
                           _248_CONV2D_35_OUTPUT_0_MEM_CONT_SIZE,
                           _248_CONV2D_35_OUTPUT_0_MEM_WRAP_OFFSET,
                           _248_CONV2D_35_OUTPUT_0_MEM_WRAP_SIZE,
                           _248_CONV2D_35_OUTPUT_0_MEM_STRIDE>
                           (_247_Mul_64_output_0, _248_Conv2D_35_output_0, _248_Conv2D_35_weights_output_0, _248_Conv2D_35_biases_output_0, _248_CONV2D_35_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _248_Conv2D_35\n");
    FILE* _248_CONV2D_35_OUTPUT_0_STREAM = fopen("data/export_outputs/_248_Conv2D_35_output_0.txt", "w");
    export_cpp::saveOutputs<_248_CONV2D_35_OUTPUT_0_NB_OUTPUTS,
                            _248_CONV2D_35_OUTPUT_0_OUT_HEIGHT,
                            _248_CONV2D_35_OUTPUT_0_OUT_WIDTH,
                            _248_CONV2D_35_OUTPUT_0_MEM_CONT_OFFSET,
                            _248_CONV2D_35_OUTPUT_0_MEM_CONT_SIZE,
                            _248_CONV2D_35_OUTPUT_0_MEM_WRAP_OFFSET,
                            _248_CONV2D_35_OUTPUT_0_MEM_WRAP_SIZE,
                            _248_CONV2D_35_OUTPUT_0_DEV_FMT>
                            (_248_Conv2D_35_output_0,
                            _248_CONV2D_35_OUTPUT_0_STREAM);
    fclose(_248_CONV2D_35_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _249_PadConv_27_output_0 = (float*) (mem_0 + _249_PADCONV_27_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_246_MUL_63_OUTPUT_0_IN_BATCH,
                           _249_PADCONV_27_NB_CHANNELS,
                           _246_MUL_63_OUTPUT_0_IN_HEIGHT,
                           _246_MUL_63_OUTPUT_0_IN_WIDTH,
                           _249_PADCONV_27_NB_OUTPUTS,
                           _249_PADCONV_27_OUTPUT_0_OUT_HEIGHT,
                           _249_PADCONV_27_OUTPUT_0_OUT_WIDTH,
                           _249_PADCONV_27_PADDING_Y,
                           _249_PADCONV_27_PADDING_X,
                           _249_PADCONV_27_STRIDE_Y,
                           _249_PADCONV_27_STRIDE_X,
                           _249_PADCONV_27_DILATION_Y,
                           _249_PADCONV_27_DILATION_X,
                           _249_PADCONV_27_KERNEL_HEIGHT,
                           _249_PADCONV_27_KERNEL_WIDTH,
                           _249_PADCONV_27_ACTIVATION,
                           _246_MUL_63_OUTPUT_0_MEM_CONT_OFFSET,
                           _246_MUL_63_OUTPUT_0_MEM_CONT_SIZE,
                           _246_MUL_63_OUTPUT_0_MEM_WRAP_OFFSET,
                           _246_MUL_63_OUTPUT_0_MEM_WRAP_SIZE,
                           _246_MUL_63_OUTPUT_0_MEM_STRIDE,
                           _249_PADCONV_27_OUTPUT_0_MEM_CONT_OFFSET,
                           _249_PADCONV_27_OUTPUT_0_MEM_CONT_SIZE,
                           _249_PADCONV_27_OUTPUT_0_MEM_WRAP_OFFSET,
                           _249_PADCONV_27_OUTPUT_0_MEM_WRAP_SIZE,
                           _249_PADCONV_27_OUTPUT_0_MEM_STRIDE>
                           (_246_Mul_63_output_0, _249_PadConv_27_output_0, _249_PadConv_27_weights_output_0, _249_PadConv_27_biases_output_0, _249_PADCONV_27_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _249_PadConv_27\n");
    FILE* _249_PADCONV_27_OUTPUT_0_STREAM = fopen("data/export_outputs/_249_PadConv_27_output_0.txt", "w");
    export_cpp::saveOutputs<_249_PADCONV_27_OUTPUT_0_NB_OUTPUTS,
                            _249_PADCONV_27_OUTPUT_0_OUT_HEIGHT,
                            _249_PADCONV_27_OUTPUT_0_OUT_WIDTH,
                            _249_PADCONV_27_OUTPUT_0_MEM_CONT_OFFSET,
                            _249_PADCONV_27_OUTPUT_0_MEM_CONT_SIZE,
                            _249_PADCONV_27_OUTPUT_0_MEM_WRAP_OFFSET,
                            _249_PADCONV_27_OUTPUT_0_MEM_WRAP_SIZE,
                            _249_PADCONV_27_OUTPUT_0_DEV_FMT>
                            (_249_PadConv_27_output_0,
                            _249_PADCONV_27_OUTPUT_0_STREAM);
    fclose(_249_PADCONV_27_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _250_Sigmoid_65_output_0 = (float*) (mem_0 + _250_SIGMOID_65_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_250_SIGMOID_65_NB_ELTS>
                               (_249_PadConv_27_output_0, _250_Sigmoid_65_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _250_Sigmoid_65\n");
    FILE* _250_SIGMOID_65_OUTPUT_0_STREAM = fopen("data/export_outputs/_250_Sigmoid_65_output_0.txt", "w");
    export_cpp::saveOutputs<_250_SIGMOID_65_OUTPUT_0_NB_OUTPUTS,
                            _250_SIGMOID_65_OUTPUT_0_OUT_HEIGHT,
                            _250_SIGMOID_65_OUTPUT_0_OUT_WIDTH,
                            _250_SIGMOID_65_OUTPUT_0_MEM_CONT_OFFSET,
                            _250_SIGMOID_65_OUTPUT_0_MEM_CONT_SIZE,
                            _250_SIGMOID_65_OUTPUT_0_MEM_WRAP_OFFSET,
                            _250_SIGMOID_65_OUTPUT_0_MEM_WRAP_SIZE,
                            _250_SIGMOID_65_OUTPUT_0_DEV_FMT>
                            (_250_Sigmoid_65_output_0,
                            _250_SIGMOID_65_OUTPUT_0_STREAM);
    fclose(_250_SIGMOID_65_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _251_Sigmoid_66_output_0 = (float*) (mem_0 + _251_SIGMOID_66_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_251_SIGMOID_66_NB_ELTS>
                               (_248_Conv2D_35_output_0, _251_Sigmoid_66_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _251_Sigmoid_66\n");
    FILE* _251_SIGMOID_66_OUTPUT_0_STREAM = fopen("data/export_outputs/_251_Sigmoid_66_output_0.txt", "w");
    export_cpp::saveOutputs<_251_SIGMOID_66_OUTPUT_0_NB_OUTPUTS,
                            _251_SIGMOID_66_OUTPUT_0_OUT_HEIGHT,
                            _251_SIGMOID_66_OUTPUT_0_OUT_WIDTH,
                            _251_SIGMOID_66_OUTPUT_0_MEM_CONT_OFFSET,
                            _251_SIGMOID_66_OUTPUT_0_MEM_CONT_SIZE,
                            _251_SIGMOID_66_OUTPUT_0_MEM_WRAP_OFFSET,
                            _251_SIGMOID_66_OUTPUT_0_MEM_WRAP_SIZE,
                            _251_SIGMOID_66_OUTPUT_0_DEV_FMT>
                            (_251_Sigmoid_66_output_0,
                            _251_SIGMOID_66_OUTPUT_0_STREAM);
    fclose(_251_SIGMOID_66_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _252_Mul_65_output_0 = (float*) (mem_0 + _252_MUL_65_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_252_MUL_65_NB_MAT,
                                 _252_MUL_65_ELEM_OP,
                                 _252_MUL_65_INPUT1_CONT_SIZE,
                                 _252_MUL_65_INPUT2_CONT_SIZE,
                                 _252_MUL_65_OUTPUT_CONT_SIZE,
                                 _252_MUL_65_OFFSET_IN1,
                                 _252_MUL_65_OFFSET_IN2,
                                 _252_MUL_65_ACTIVATION,
                                 _249_PADCONV_27_OUTPUT_0_MEM_CONT_OFFSET,
                                 _249_PADCONV_27_OUTPUT_0_MEM_CONT_SIZE,
                                 _249_PADCONV_27_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _249_PADCONV_27_OUTPUT_0_MEM_WRAP_SIZE,
                                 _249_PADCONV_27_OUTPUT_0_MEM_STRIDE,
                                 _250_SIGMOID_65_OUTPUT_0_MEM_CONT_OFFSET,
                                 _250_SIGMOID_65_OUTPUT_0_MEM_CONT_SIZE,
                                 _250_SIGMOID_65_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _250_SIGMOID_65_OUTPUT_0_MEM_WRAP_SIZE,
                                 _250_SIGMOID_65_OUTPUT_0_MEM_STRIDE,
                                 _252_MUL_65_OUTPUT_0_MEM_CONT_OFFSET,
                                 _252_MUL_65_OUTPUT_0_MEM_CONT_SIZE,
                                 _252_MUL_65_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _252_MUL_65_OUTPUT_0_MEM_WRAP_SIZE,
                                 _252_MUL_65_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_252_Mul_65_output_0, _252_MUL_65_RESCALING, _249_PadConv_27_output_0, _250_Sigmoid_65_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _252_Mul_65\n");
    FILE* _252_MUL_65_OUTPUT_0_STREAM = fopen("data/export_outputs/_252_Mul_65_output_0.txt", "w");
    export_cpp::saveOutputs<_252_MUL_65_OUTPUT_0_NB_OUTPUTS,
                            _252_MUL_65_OUTPUT_0_OUT_HEIGHT,
                            _252_MUL_65_OUTPUT_0_OUT_WIDTH,
                            _252_MUL_65_OUTPUT_0_MEM_CONT_OFFSET,
                            _252_MUL_65_OUTPUT_0_MEM_CONT_SIZE,
                            _252_MUL_65_OUTPUT_0_MEM_WRAP_OFFSET,
                            _252_MUL_65_OUTPUT_0_MEM_WRAP_SIZE,
                            _252_MUL_65_OUTPUT_0_DEV_FMT>
                            (_252_Mul_65_output_0,
                            _252_MUL_65_OUTPUT_0_STREAM);
    fclose(_252_MUL_65_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _253_Mul_66_output_0 = (float*) (mem_0 + _253_MUL_66_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_253_MUL_66_NB_MAT,
                                 _253_MUL_66_ELEM_OP,
                                 _253_MUL_66_INPUT1_CONT_SIZE,
                                 _253_MUL_66_INPUT2_CONT_SIZE,
                                 _253_MUL_66_OUTPUT_CONT_SIZE,
                                 _253_MUL_66_OFFSET_IN1,
                                 _253_MUL_66_OFFSET_IN2,
                                 _253_MUL_66_ACTIVATION,
                                 _248_CONV2D_35_OUTPUT_0_MEM_CONT_OFFSET,
                                 _248_CONV2D_35_OUTPUT_0_MEM_CONT_SIZE,
                                 _248_CONV2D_35_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _248_CONV2D_35_OUTPUT_0_MEM_WRAP_SIZE,
                                 _248_CONV2D_35_OUTPUT_0_MEM_STRIDE,
                                 _251_SIGMOID_66_OUTPUT_0_MEM_CONT_OFFSET,
                                 _251_SIGMOID_66_OUTPUT_0_MEM_CONT_SIZE,
                                 _251_SIGMOID_66_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _251_SIGMOID_66_OUTPUT_0_MEM_WRAP_SIZE,
                                 _251_SIGMOID_66_OUTPUT_0_MEM_STRIDE,
                                 _253_MUL_66_OUTPUT_0_MEM_CONT_OFFSET,
                                 _253_MUL_66_OUTPUT_0_MEM_CONT_SIZE,
                                 _253_MUL_66_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _253_MUL_66_OUTPUT_0_MEM_WRAP_SIZE,
                                 _253_MUL_66_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_253_Mul_66_output_0, _253_MUL_66_RESCALING, _248_Conv2D_35_output_0, _251_Sigmoid_66_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _253_Mul_66\n");
    FILE* _253_MUL_66_OUTPUT_0_STREAM = fopen("data/export_outputs/_253_Mul_66_output_0.txt", "w");
    export_cpp::saveOutputs<_253_MUL_66_OUTPUT_0_NB_OUTPUTS,
                            _253_MUL_66_OUTPUT_0_OUT_HEIGHT,
                            _253_MUL_66_OUTPUT_0_OUT_WIDTH,
                            _253_MUL_66_OUTPUT_0_MEM_CONT_OFFSET,
                            _253_MUL_66_OUTPUT_0_MEM_CONT_SIZE,
                            _253_MUL_66_OUTPUT_0_MEM_WRAP_OFFSET,
                            _253_MUL_66_OUTPUT_0_MEM_WRAP_SIZE,
                            _253_MUL_66_OUTPUT_0_DEV_FMT>
                            (_253_Mul_66_output_0,
                            _253_MUL_66_OUTPUT_0_STREAM);
    fclose(_253_MUL_66_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _254_Conv2D_36_output_0 = (float*) (mem_0 + _254_CONV2D_36_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_252_MUL_65_OUTPUT_0_IN_BATCH,
                           _254_CONV2D_36_NB_CHANNELS,
                           _252_MUL_65_OUTPUT_0_IN_HEIGHT,
                           _252_MUL_65_OUTPUT_0_IN_WIDTH,
                           _254_CONV2D_36_NB_OUTPUTS,
                           _254_CONV2D_36_OUTPUT_0_OUT_HEIGHT,
                           _254_CONV2D_36_OUTPUT_0_OUT_WIDTH,
                           _254_CONV2D_36_PADDING_Y,
                           _254_CONV2D_36_PADDING_X,
                           _254_CONV2D_36_STRIDE_Y,
                           _254_CONV2D_36_STRIDE_X,
                           _254_CONV2D_36_DILATION_Y,
                           _254_CONV2D_36_DILATION_X,
                           _254_CONV2D_36_KERNEL_HEIGHT,
                           _254_CONV2D_36_KERNEL_WIDTH,
                           _254_CONV2D_36_ACTIVATION,
                           _252_MUL_65_OUTPUT_0_MEM_CONT_OFFSET,
                           _252_MUL_65_OUTPUT_0_MEM_CONT_SIZE,
                           _252_MUL_65_OUTPUT_0_MEM_WRAP_OFFSET,
                           _252_MUL_65_OUTPUT_0_MEM_WRAP_SIZE,
                           _252_MUL_65_OUTPUT_0_MEM_STRIDE,
                           _254_CONV2D_36_OUTPUT_0_MEM_CONT_OFFSET,
                           _254_CONV2D_36_OUTPUT_0_MEM_CONT_SIZE,
                           _254_CONV2D_36_OUTPUT_0_MEM_WRAP_OFFSET,
                           _254_CONV2D_36_OUTPUT_0_MEM_WRAP_SIZE,
                           _254_CONV2D_36_OUTPUT_0_MEM_STRIDE>
                           (_252_Mul_65_output_0, _254_Conv2D_36_output_0, _254_Conv2D_36_weights_output_0, _254_Conv2D_36_biases_output_0, _254_CONV2D_36_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _254_Conv2D_36\n");
    FILE* _254_CONV2D_36_OUTPUT_0_STREAM = fopen("data/export_outputs/_254_Conv2D_36_output_0.txt", "w");
    export_cpp::saveOutputs<_254_CONV2D_36_OUTPUT_0_NB_OUTPUTS,
                            _254_CONV2D_36_OUTPUT_0_OUT_HEIGHT,
                            _254_CONV2D_36_OUTPUT_0_OUT_WIDTH,
                            _254_CONV2D_36_OUTPUT_0_MEM_CONT_OFFSET,
                            _254_CONV2D_36_OUTPUT_0_MEM_CONT_SIZE,
                            _254_CONV2D_36_OUTPUT_0_MEM_WRAP_OFFSET,
                            _254_CONV2D_36_OUTPUT_0_MEM_WRAP_SIZE,
                            _254_CONV2D_36_OUTPUT_0_DEV_FMT>
                            (_254_Conv2D_36_output_0,
                            _254_CONV2D_36_OUTPUT_0_STREAM);
    fclose(_254_CONV2D_36_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _255_PadConvDw_11_output_0 = (float*) (mem_0 + _255_PADCONVDW_11_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_depthwise_forward<_253_MUL_66_OUTPUT_0_IN_BATCH,
                           _255_PADCONVDW_11_NB_CHANNELS,
                           _253_MUL_66_OUTPUT_0_IN_HEIGHT,
                           _253_MUL_66_OUTPUT_0_IN_WIDTH,
                           _255_PADCONVDW_11_NB_OUTPUTS,
                           _255_PADCONVDW_11_OUTPUT_0_OUT_HEIGHT,
                           _255_PADCONVDW_11_OUTPUT_0_OUT_WIDTH,
                           _255_PADCONVDW_11_PADDING_Y,
                           _255_PADCONVDW_11_PADDING_X,
                           _255_PADCONVDW_11_STRIDE_Y,
                           _255_PADCONVDW_11_STRIDE_X,
                           _255_PADCONVDW_11_DILATION_Y,
                           _255_PADCONVDW_11_DILATION_X,
                           _255_PADCONVDW_11_KERNEL_HEIGHT,
                           _255_PADCONVDW_11_KERNEL_WIDTH,
                           _255_PADCONVDW_11_ACTIVATION,
                           _253_MUL_66_OUTPUT_0_MEM_CONT_OFFSET,
                           _253_MUL_66_OUTPUT_0_MEM_CONT_SIZE,
                           _253_MUL_66_OUTPUT_0_MEM_WRAP_OFFSET,
                           _253_MUL_66_OUTPUT_0_MEM_WRAP_SIZE,
                           _253_MUL_66_OUTPUT_0_MEM_STRIDE,
                           _255_PADCONVDW_11_OUTPUT_0_MEM_CONT_OFFSET,
                           _255_PADCONVDW_11_OUTPUT_0_MEM_CONT_SIZE,
                           _255_PADCONVDW_11_OUTPUT_0_MEM_WRAP_OFFSET,
                           _255_PADCONVDW_11_OUTPUT_0_MEM_WRAP_SIZE,
                           _255_PADCONVDW_11_OUTPUT_0_MEM_STRIDE>
                           (_253_Mul_66_output_0, _255_PadConvDw_11_output_0, _255_PadConvDw_11_weights_output_0, _255_PadConvDw_11_biases_output_0, _255_PADCONVDW_11_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _255_PadConvDw_11\n");
    FILE* _255_PADCONVDW_11_OUTPUT_0_STREAM = fopen("data/export_outputs/_255_PadConvDw_11_output_0.txt", "w");
    export_cpp::saveOutputs<_255_PADCONVDW_11_OUTPUT_0_NB_OUTPUTS,
                            _255_PADCONVDW_11_OUTPUT_0_OUT_HEIGHT,
                            _255_PADCONVDW_11_OUTPUT_0_OUT_WIDTH,
                            _255_PADCONVDW_11_OUTPUT_0_MEM_CONT_OFFSET,
                            _255_PADCONVDW_11_OUTPUT_0_MEM_CONT_SIZE,
                            _255_PADCONVDW_11_OUTPUT_0_MEM_WRAP_OFFSET,
                            _255_PADCONVDW_11_OUTPUT_0_MEM_WRAP_SIZE,
                            _255_PADCONVDW_11_OUTPUT_0_DEV_FMT>
                            (_255_PadConvDw_11_output_0,
                            _255_PADCONVDW_11_OUTPUT_0_STREAM);
    fclose(_255_PADCONVDW_11_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _256_Sigmoid_67_output_0 = (float*) (mem_0 + _256_SIGMOID_67_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_256_SIGMOID_67_NB_ELTS>
                               (_255_PadConvDw_11_output_0, _256_Sigmoid_67_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _256_Sigmoid_67\n");
    FILE* _256_SIGMOID_67_OUTPUT_0_STREAM = fopen("data/export_outputs/_256_Sigmoid_67_output_0.txt", "w");
    export_cpp::saveOutputs<_256_SIGMOID_67_OUTPUT_0_NB_OUTPUTS,
                            _256_SIGMOID_67_OUTPUT_0_OUT_HEIGHT,
                            _256_SIGMOID_67_OUTPUT_0_OUT_WIDTH,
                            _256_SIGMOID_67_OUTPUT_0_MEM_CONT_OFFSET,
                            _256_SIGMOID_67_OUTPUT_0_MEM_CONT_SIZE,
                            _256_SIGMOID_67_OUTPUT_0_MEM_WRAP_OFFSET,
                            _256_SIGMOID_67_OUTPUT_0_MEM_WRAP_SIZE,
                            _256_SIGMOID_67_OUTPUT_0_DEV_FMT>
                            (_256_Sigmoid_67_output_0,
                            _256_SIGMOID_67_OUTPUT_0_STREAM);
    fclose(_256_SIGMOID_67_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _257_Transpose_4_output_0 = (float*) (mem_0 + _257_TRANSPOSE_4_OUTPUT_0_MEM_OFFSET);

    export_cpp::transpose_ND_forward<float,
                                     _257_TRANSPOSE_4_NB_DIMS,
                                     _257_TRANSPOSE_4_NB_ELTS,
                                     _257_TRANSPOSE_4_PERMUTE,
                                     _257_TRANSPOSE_4_DIMS,
                                     _257_TRANSPOSE_4_OUT_STRIDES,
                                    _254_CONV2D_36_OUTPUT_0_MEM_CONT_OFFSET,
                                    _254_CONV2D_36_OUTPUT_0_MEM_CONT_SIZE,
                                    _254_CONV2D_36_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _254_CONV2D_36_OUTPUT_0_MEM_WRAP_SIZE,
                                    _254_CONV2D_36_OUTPUT_0_MEM_STRIDE,
                                    _257_TRANSPOSE_4_OUTPUT_0_MEM_CONT_OFFSET,
                                    _257_TRANSPOSE_4_OUTPUT_0_MEM_CONT_SIZE,
                                    _257_TRANSPOSE_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _257_TRANSPOSE_4_OUTPUT_0_MEM_WRAP_SIZE,
                                    _257_TRANSPOSE_4_OUTPUT_0_MEM_STRIDE>
                                     (_254_Conv2D_36_output_0,
                                     _257_Transpose_4_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _257_Transpose_4\n");
    FILE* _257_TRANSPOSE_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_257_Transpose_4_output_0.txt", "w");
    export_cpp::saveOutputs<_257_TRANSPOSE_4_OUTPUT_0_NB_OUTPUTS,
                            _257_TRANSPOSE_4_OUTPUT_0_OUT_HEIGHT,
                            _257_TRANSPOSE_4_OUTPUT_0_OUT_WIDTH,
                            _257_TRANSPOSE_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _257_TRANSPOSE_4_OUTPUT_0_MEM_CONT_SIZE,
                            _257_TRANSPOSE_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _257_TRANSPOSE_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _257_TRANSPOSE_4_OUTPUT_0_DEV_FMT>
                            (_257_Transpose_4_output_0,
                            _257_TRANSPOSE_4_OUTPUT_0_STREAM);
    fclose(_257_TRANSPOSE_4_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _258_Mul_67_output_0 = (float*) (mem_0 + _258_MUL_67_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_258_MUL_67_NB_MAT,
                                 _258_MUL_67_ELEM_OP,
                                 _258_MUL_67_INPUT1_CONT_SIZE,
                                 _258_MUL_67_INPUT2_CONT_SIZE,
                                 _258_MUL_67_OUTPUT_CONT_SIZE,
                                 _258_MUL_67_OFFSET_IN1,
                                 _258_MUL_67_OFFSET_IN2,
                                 _258_MUL_67_ACTIVATION,
                                 _255_PADCONVDW_11_OUTPUT_0_MEM_CONT_OFFSET,
                                 _255_PADCONVDW_11_OUTPUT_0_MEM_CONT_SIZE,
                                 _255_PADCONVDW_11_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _255_PADCONVDW_11_OUTPUT_0_MEM_WRAP_SIZE,
                                 _255_PADCONVDW_11_OUTPUT_0_MEM_STRIDE,
                                 _256_SIGMOID_67_OUTPUT_0_MEM_CONT_OFFSET,
                                 _256_SIGMOID_67_OUTPUT_0_MEM_CONT_SIZE,
                                 _256_SIGMOID_67_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _256_SIGMOID_67_OUTPUT_0_MEM_WRAP_SIZE,
                                 _256_SIGMOID_67_OUTPUT_0_MEM_STRIDE,
                                 _258_MUL_67_OUTPUT_0_MEM_CONT_OFFSET,
                                 _258_MUL_67_OUTPUT_0_MEM_CONT_SIZE,
                                 _258_MUL_67_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _258_MUL_67_OUTPUT_0_MEM_WRAP_SIZE,
                                 _258_MUL_67_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_258_Mul_67_output_0, _258_MUL_67_RESCALING, _255_PadConvDw_11_output_0, _256_Sigmoid_67_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _258_Mul_67\n");
    FILE* _258_MUL_67_OUTPUT_0_STREAM = fopen("data/export_outputs/_258_Mul_67_output_0.txt", "w");
    export_cpp::saveOutputs<_258_MUL_67_OUTPUT_0_NB_OUTPUTS,
                            _258_MUL_67_OUTPUT_0_OUT_HEIGHT,
                            _258_MUL_67_OUTPUT_0_OUT_WIDTH,
                            _258_MUL_67_OUTPUT_0_MEM_CONT_OFFSET,
                            _258_MUL_67_OUTPUT_0_MEM_CONT_SIZE,
                            _258_MUL_67_OUTPUT_0_MEM_WRAP_OFFSET,
                            _258_MUL_67_OUTPUT_0_MEM_WRAP_SIZE,
                            _258_MUL_67_OUTPUT_0_DEV_FMT>
                            (_258_Mul_67_output_0,
                            _258_MUL_67_OUTPUT_0_STREAM);
    fclose(_258_MUL_67_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _259_Reshape_4_output_0 = (float*) (mem_0 + _259_RESHAPE_4_OUTPUT_0_MEM_OFFSET);


    export_cpp::identity_forward<_259_RESHAPE_4_NB_ELTS,
                                 _257_TRANSPOSE_4_OUTPUT_0_MEM_CONT_OFFSET,
                                 _257_TRANSPOSE_4_OUTPUT_0_MEM_CONT_SIZE,
                                 _257_TRANSPOSE_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _257_TRANSPOSE_4_OUTPUT_0_MEM_WRAP_SIZE,
                                 _257_TRANSPOSE_4_OUTPUT_0_MEM_STRIDE,
                                 _259_RESHAPE_4_OUTPUT_0_MEM_CONT_OFFSET,
                                 _259_RESHAPE_4_OUTPUT_0_MEM_CONT_SIZE,
                                 _259_RESHAPE_4_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _259_RESHAPE_4_OUTPUT_0_MEM_WRAP_SIZE,
                                 _259_RESHAPE_4_OUTPUT_0_MEM_STRIDE>
                                (_257_Transpose_4_output_0, _259_Reshape_4_output_0);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _259_Reshape_4\n");
    FILE* _259_RESHAPE_4_OUTPUT_0_STREAM = fopen("data/export_outputs/_259_Reshape_4_output_0.txt", "w");
    export_cpp::saveOutputs<_259_RESHAPE_4_OUTPUT_0_NB_OUTPUTS,
                            _259_RESHAPE_4_OUTPUT_0_OUT_HEIGHT,
                            _259_RESHAPE_4_OUTPUT_0_OUT_WIDTH,
                            _259_RESHAPE_4_OUTPUT_0_MEM_CONT_OFFSET,
                            _259_RESHAPE_4_OUTPUT_0_MEM_CONT_SIZE,
                            _259_RESHAPE_4_OUTPUT_0_MEM_WRAP_OFFSET,
                            _259_RESHAPE_4_OUTPUT_0_MEM_WRAP_SIZE,
                            _259_RESHAPE_4_OUTPUT_0_DEV_FMT>
                            (_259_Reshape_4_output_0,
                            _259_RESHAPE_4_OUTPUT_0_STREAM);
    fclose(_259_RESHAPE_4_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _260_Concat_14_output_0 = (float*) (mem_0 + _260_CONCAT_14_OUTPUT_0_MEM_OFFSET);

    const float* _260_Concat_14_inputs[] = { _173_Reshape_0_output_0, _215_Reshape_2_output_0, _259_Reshape_4_output_0 };
    export_cpp::concat_forward<_260_CONCAT_14_AXIS_SIZE_POST,
                               _260_CONCAT_14_AXIS_SIZE_PRE,
                               _260_CONCAT_14_AXIS_SIZE,
                               _260_CONCAT_14_TOTAL_AXIS_SIZE,
                               3> 
                              (_260_Concat_14_inputs,
                               _260_Concat_14_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _260_Concat_14\n");
    FILE* _260_CONCAT_14_OUTPUT_0_STREAM = fopen("data/export_outputs/_260_Concat_14_output_0.txt", "w");
    export_cpp::saveOutputs<_260_CONCAT_14_OUTPUT_0_NB_OUTPUTS,
                            _260_CONCAT_14_OUTPUT_0_OUT_HEIGHT,
                            _260_CONCAT_14_OUTPUT_0_OUT_WIDTH,
                            _260_CONCAT_14_OUTPUT_0_MEM_CONT_OFFSET,
                            _260_CONCAT_14_OUTPUT_0_MEM_CONT_SIZE,
                            _260_CONCAT_14_OUTPUT_0_MEM_WRAP_OFFSET,
                            _260_CONCAT_14_OUTPUT_0_MEM_WRAP_SIZE,
                            _260_CONCAT_14_OUTPUT_0_DEV_FMT>
                            (_260_Concat_14_output_0,
                            _260_CONCAT_14_OUTPUT_0_STREAM);
    fclose(_260_CONCAT_14_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _261_Conv2D_37_output_0 = (float*) (mem_0 + _261_CONV2D_37_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_258_MUL_67_OUTPUT_0_IN_BATCH,
                           _261_CONV2D_37_NB_CHANNELS,
                           _258_MUL_67_OUTPUT_0_IN_HEIGHT,
                           _258_MUL_67_OUTPUT_0_IN_WIDTH,
                           _261_CONV2D_37_NB_OUTPUTS,
                           _261_CONV2D_37_OUTPUT_0_OUT_HEIGHT,
                           _261_CONV2D_37_OUTPUT_0_OUT_WIDTH,
                           _261_CONV2D_37_PADDING_Y,
                           _261_CONV2D_37_PADDING_X,
                           _261_CONV2D_37_STRIDE_Y,
                           _261_CONV2D_37_STRIDE_X,
                           _261_CONV2D_37_DILATION_Y,
                           _261_CONV2D_37_DILATION_X,
                           _261_CONV2D_37_KERNEL_HEIGHT,
                           _261_CONV2D_37_KERNEL_WIDTH,
                           _261_CONV2D_37_ACTIVATION,
                           _258_MUL_67_OUTPUT_0_MEM_CONT_OFFSET,
                           _258_MUL_67_OUTPUT_0_MEM_CONT_SIZE,
                           _258_MUL_67_OUTPUT_0_MEM_WRAP_OFFSET,
                           _258_MUL_67_OUTPUT_0_MEM_WRAP_SIZE,
                           _258_MUL_67_OUTPUT_0_MEM_STRIDE,
                           _261_CONV2D_37_OUTPUT_0_MEM_CONT_OFFSET,
                           _261_CONV2D_37_OUTPUT_0_MEM_CONT_SIZE,
                           _261_CONV2D_37_OUTPUT_0_MEM_WRAP_OFFSET,
                           _261_CONV2D_37_OUTPUT_0_MEM_WRAP_SIZE,
                           _261_CONV2D_37_OUTPUT_0_MEM_STRIDE>
                           (_258_Mul_67_output_0, _261_Conv2D_37_output_0, _261_Conv2D_37_weights_output_0, _261_Conv2D_37_biases_output_0, _261_CONV2D_37_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _261_Conv2D_37\n");
    FILE* _261_CONV2D_37_OUTPUT_0_STREAM = fopen("data/export_outputs/_261_Conv2D_37_output_0.txt", "w");
    export_cpp::saveOutputs<_261_CONV2D_37_OUTPUT_0_NB_OUTPUTS,
                            _261_CONV2D_37_OUTPUT_0_OUT_HEIGHT,
                            _261_CONV2D_37_OUTPUT_0_OUT_WIDTH,
                            _261_CONV2D_37_OUTPUT_0_MEM_CONT_OFFSET,
                            _261_CONV2D_37_OUTPUT_0_MEM_CONT_SIZE,
                            _261_CONV2D_37_OUTPUT_0_MEM_WRAP_OFFSET,
                            _261_CONV2D_37_OUTPUT_0_MEM_WRAP_SIZE,
                            _261_CONV2D_37_OUTPUT_0_DEV_FMT>
                            (_261_Conv2D_37_output_0,
                            _261_CONV2D_37_OUTPUT_0_STREAM);
    fclose(_261_CONV2D_37_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _262_Sigmoid_68_output_0 = (float*) (mem_0 + _262_SIGMOID_68_OUTPUT_0_MEM_OFFSET);

    export_cpp::sigmoid_forward<_262_SIGMOID_68_NB_ELTS>
                               (_261_Conv2D_37_output_0, _262_Sigmoid_68_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _262_Sigmoid_68\n");
    FILE* _262_SIGMOID_68_OUTPUT_0_STREAM = fopen("data/export_outputs/_262_Sigmoid_68_output_0.txt", "w");
    export_cpp::saveOutputs<_262_SIGMOID_68_OUTPUT_0_NB_OUTPUTS,
                            _262_SIGMOID_68_OUTPUT_0_OUT_HEIGHT,
                            _262_SIGMOID_68_OUTPUT_0_OUT_WIDTH,
                            _262_SIGMOID_68_OUTPUT_0_MEM_CONT_OFFSET,
                            _262_SIGMOID_68_OUTPUT_0_MEM_CONT_SIZE,
                            _262_SIGMOID_68_OUTPUT_0_MEM_WRAP_OFFSET,
                            _262_SIGMOID_68_OUTPUT_0_MEM_WRAP_SIZE,
                            _262_SIGMOID_68_OUTPUT_0_DEV_FMT>
                            (_262_Sigmoid_68_output_0,
                            _262_SIGMOID_68_OUTPUT_0_STREAM);
    fclose(_262_SIGMOID_68_OUTPUT_0_STREAM);
    #endif


    
    
    

    float* _263_Mul_68_output_0 = (float*) (mem_0 + _263_MUL_68_OUTPUT_0_MEM_OFFSET);

    export_cpp::elemwise_forward<_263_MUL_68_NB_MAT,
                                 _263_MUL_68_ELEM_OP,
                                 _263_MUL_68_INPUT1_CONT_SIZE,
                                 _263_MUL_68_INPUT2_CONT_SIZE,
                                 _263_MUL_68_OUTPUT_CONT_SIZE,
                                 _263_MUL_68_OFFSET_IN1,
                                 _263_MUL_68_OFFSET_IN2,
                                 _263_MUL_68_ACTIVATION,
                                 _261_CONV2D_37_OUTPUT_0_MEM_CONT_OFFSET,
                                 _261_CONV2D_37_OUTPUT_0_MEM_CONT_SIZE,
                                 _261_CONV2D_37_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _261_CONV2D_37_OUTPUT_0_MEM_WRAP_SIZE,
                                 _261_CONV2D_37_OUTPUT_0_MEM_STRIDE,
                                 _262_SIGMOID_68_OUTPUT_0_MEM_CONT_OFFSET,
                                 _262_SIGMOID_68_OUTPUT_0_MEM_CONT_SIZE,
                                 _262_SIGMOID_68_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _262_SIGMOID_68_OUTPUT_0_MEM_WRAP_SIZE,
                                 _262_SIGMOID_68_OUTPUT_0_MEM_STRIDE,
                                 _263_MUL_68_OUTPUT_0_MEM_CONT_OFFSET,
                                 _263_MUL_68_OUTPUT_0_MEM_CONT_SIZE,
                                 _263_MUL_68_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _263_MUL_68_OUTPUT_0_MEM_WRAP_SIZE,
                                 _263_MUL_68_OUTPUT_0_MEM_STRIDE,
                                 float>
                                 (_263_Mul_68_output_0, _263_MUL_68_RESCALING, _261_Conv2D_37_output_0, _262_Sigmoid_68_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _263_Mul_68\n");
    FILE* _263_MUL_68_OUTPUT_0_STREAM = fopen("data/export_outputs/_263_Mul_68_output_0.txt", "w");
    export_cpp::saveOutputs<_263_MUL_68_OUTPUT_0_NB_OUTPUTS,
                            _263_MUL_68_OUTPUT_0_OUT_HEIGHT,
                            _263_MUL_68_OUTPUT_0_OUT_WIDTH,
                            _263_MUL_68_OUTPUT_0_MEM_CONT_OFFSET,
                            _263_MUL_68_OUTPUT_0_MEM_CONT_SIZE,
                            _263_MUL_68_OUTPUT_0_MEM_WRAP_OFFSET,
                            _263_MUL_68_OUTPUT_0_MEM_WRAP_SIZE,
                            _263_MUL_68_OUTPUT_0_DEV_FMT>
                            (_263_Mul_68_output_0,
                            _263_MUL_68_OUTPUT_0_STREAM);
    fclose(_263_MUL_68_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _264_Conv2D_38_output_0 = (float*) (mem_0 + _264_CONV2D_38_OUTPUT_0_MEM_OFFSET);

    export_cpp::convolution_forward<_263_MUL_68_OUTPUT_0_IN_BATCH,
                           _264_CONV2D_38_NB_CHANNELS,
                           _263_MUL_68_OUTPUT_0_IN_HEIGHT,
                           _263_MUL_68_OUTPUT_0_IN_WIDTH,
                           _264_CONV2D_38_NB_OUTPUTS,
                           _264_CONV2D_38_OUTPUT_0_OUT_HEIGHT,
                           _264_CONV2D_38_OUTPUT_0_OUT_WIDTH,
                           _264_CONV2D_38_PADDING_Y,
                           _264_CONV2D_38_PADDING_X,
                           _264_CONV2D_38_STRIDE_Y,
                           _264_CONV2D_38_STRIDE_X,
                           _264_CONV2D_38_DILATION_Y,
                           _264_CONV2D_38_DILATION_X,
                           _264_CONV2D_38_KERNEL_HEIGHT,
                           _264_CONV2D_38_KERNEL_WIDTH,
                           _264_CONV2D_38_ACTIVATION,
                           _263_MUL_68_OUTPUT_0_MEM_CONT_OFFSET,
                           _263_MUL_68_OUTPUT_0_MEM_CONT_SIZE,
                           _263_MUL_68_OUTPUT_0_MEM_WRAP_OFFSET,
                           _263_MUL_68_OUTPUT_0_MEM_WRAP_SIZE,
                           _263_MUL_68_OUTPUT_0_MEM_STRIDE,
                           _264_CONV2D_38_OUTPUT_0_MEM_CONT_OFFSET,
                           _264_CONV2D_38_OUTPUT_0_MEM_CONT_SIZE,
                           _264_CONV2D_38_OUTPUT_0_MEM_WRAP_OFFSET,
                           _264_CONV2D_38_OUTPUT_0_MEM_WRAP_SIZE,
                           _264_CONV2D_38_OUTPUT_0_MEM_STRIDE>
                           (_263_Mul_68_output_0, _264_Conv2D_38_output_0, _264_Conv2D_38_weights_output_0, _264_Conv2D_38_biases_output_0, _264_CONV2D_38_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _264_Conv2D_38\n");
    FILE* _264_CONV2D_38_OUTPUT_0_STREAM = fopen("data/export_outputs/_264_Conv2D_38_output_0.txt", "w");
    export_cpp::saveOutputs<_264_CONV2D_38_OUTPUT_0_NB_OUTPUTS,
                            _264_CONV2D_38_OUTPUT_0_OUT_HEIGHT,
                            _264_CONV2D_38_OUTPUT_0_OUT_WIDTH,
                            _264_CONV2D_38_OUTPUT_0_MEM_CONT_OFFSET,
                            _264_CONV2D_38_OUTPUT_0_MEM_CONT_SIZE,
                            _264_CONV2D_38_OUTPUT_0_MEM_WRAP_OFFSET,
                            _264_CONV2D_38_OUTPUT_0_MEM_WRAP_SIZE,
                            _264_CONV2D_38_OUTPUT_0_DEV_FMT>
                            (_264_Conv2D_38_output_0,
                            _264_CONV2D_38_OUTPUT_0_STREAM);
    fclose(_264_CONV2D_38_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _265_Transpose_5_output_0 = (float*) (mem_0 + _265_TRANSPOSE_5_OUTPUT_0_MEM_OFFSET);

    export_cpp::transpose_ND_forward<float,
                                     _265_TRANSPOSE_5_NB_DIMS,
                                     _265_TRANSPOSE_5_NB_ELTS,
                                     _265_TRANSPOSE_5_PERMUTE,
                                     _265_TRANSPOSE_5_DIMS,
                                     _265_TRANSPOSE_5_OUT_STRIDES,
                                    _264_CONV2D_38_OUTPUT_0_MEM_CONT_OFFSET,
                                    _264_CONV2D_38_OUTPUT_0_MEM_CONT_SIZE,
                                    _264_CONV2D_38_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _264_CONV2D_38_OUTPUT_0_MEM_WRAP_SIZE,
                                    _264_CONV2D_38_OUTPUT_0_MEM_STRIDE,
                                    _265_TRANSPOSE_5_OUTPUT_0_MEM_CONT_OFFSET,
                                    _265_TRANSPOSE_5_OUTPUT_0_MEM_CONT_SIZE,
                                    _265_TRANSPOSE_5_OUTPUT_0_MEM_WRAP_OFFSET,
                                    _265_TRANSPOSE_5_OUTPUT_0_MEM_WRAP_SIZE,
                                    _265_TRANSPOSE_5_OUTPUT_0_MEM_STRIDE>
                                     (_264_Conv2D_38_output_0,
                                     _265_Transpose_5_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _265_Transpose_5\n");
    FILE* _265_TRANSPOSE_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_265_Transpose_5_output_0.txt", "w");
    export_cpp::saveOutputs<_265_TRANSPOSE_5_OUTPUT_0_NB_OUTPUTS,
                            _265_TRANSPOSE_5_OUTPUT_0_OUT_HEIGHT,
                            _265_TRANSPOSE_5_OUTPUT_0_OUT_WIDTH,
                            _265_TRANSPOSE_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _265_TRANSPOSE_5_OUTPUT_0_MEM_CONT_SIZE,
                            _265_TRANSPOSE_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _265_TRANSPOSE_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _265_TRANSPOSE_5_OUTPUT_0_DEV_FMT>
                            (_265_Transpose_5_output_0,
                            _265_TRANSPOSE_5_OUTPUT_0_STREAM);
    fclose(_265_TRANSPOSE_5_OUTPUT_0_STREAM);
    #endif



    
    
    

    float* _266_Reshape_5_output_0 = (float*) (mem_0 + _266_RESHAPE_5_OUTPUT_0_MEM_OFFSET);


    export_cpp::identity_forward<_266_RESHAPE_5_NB_ELTS,
                                 _265_TRANSPOSE_5_OUTPUT_0_MEM_CONT_OFFSET,
                                 _265_TRANSPOSE_5_OUTPUT_0_MEM_CONT_SIZE,
                                 _265_TRANSPOSE_5_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _265_TRANSPOSE_5_OUTPUT_0_MEM_WRAP_SIZE,
                                 _265_TRANSPOSE_5_OUTPUT_0_MEM_STRIDE,
                                 _266_RESHAPE_5_OUTPUT_0_MEM_CONT_OFFSET,
                                 _266_RESHAPE_5_OUTPUT_0_MEM_CONT_SIZE,
                                 _266_RESHAPE_5_OUTPUT_0_MEM_WRAP_OFFSET,
                                 _266_RESHAPE_5_OUTPUT_0_MEM_WRAP_SIZE,
                                 _266_RESHAPE_5_OUTPUT_0_MEM_STRIDE>
                                (_265_Transpose_5_output_0, _266_Reshape_5_output_0);


    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _266_Reshape_5\n");
    FILE* _266_RESHAPE_5_OUTPUT_0_STREAM = fopen("data/export_outputs/_266_Reshape_5_output_0.txt", "w");
    export_cpp::saveOutputs<_266_RESHAPE_5_OUTPUT_0_NB_OUTPUTS,
                            _266_RESHAPE_5_OUTPUT_0_OUT_HEIGHT,
                            _266_RESHAPE_5_OUTPUT_0_OUT_WIDTH,
                            _266_RESHAPE_5_OUTPUT_0_MEM_CONT_OFFSET,
                            _266_RESHAPE_5_OUTPUT_0_MEM_CONT_SIZE,
                            _266_RESHAPE_5_OUTPUT_0_MEM_WRAP_OFFSET,
                            _266_RESHAPE_5_OUTPUT_0_MEM_WRAP_SIZE,
                            _266_RESHAPE_5_OUTPUT_0_DEV_FMT>
                            (_266_Reshape_5_output_0,
                            _266_RESHAPE_5_OUTPUT_0_STREAM);
    fclose(_266_RESHAPE_5_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _267_Concat_15_output_0 = (float*) (mem_0 + _267_CONCAT_15_OUTPUT_0_MEM_OFFSET);

    const float* _267_Concat_15_inputs[] = { _185_Reshape_1_output_0, _227_Reshape_3_output_0, _266_Reshape_5_output_0 };
    export_cpp::concat_forward<_267_CONCAT_15_AXIS_SIZE_POST,
                               _267_CONCAT_15_AXIS_SIZE_PRE,
                               _267_CONCAT_15_AXIS_SIZE,
                               _267_CONCAT_15_TOTAL_AXIS_SIZE,
                               3> 
                              (_267_Concat_15_inputs,
                               _267_Concat_15_output_0);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _267_Concat_15\n");
    FILE* _267_CONCAT_15_OUTPUT_0_STREAM = fopen("data/export_outputs/_267_Concat_15_output_0.txt", "w");
    export_cpp::saveOutputs<_267_CONCAT_15_OUTPUT_0_NB_OUTPUTS,
                            _267_CONCAT_15_OUTPUT_0_OUT_HEIGHT,
                            _267_CONCAT_15_OUTPUT_0_OUT_WIDTH,
                            _267_CONCAT_15_OUTPUT_0_MEM_CONT_OFFSET,
                            _267_CONCAT_15_OUTPUT_0_MEM_CONT_SIZE,
                            _267_CONCAT_15_OUTPUT_0_MEM_WRAP_OFFSET,
                            _267_CONCAT_15_OUTPUT_0_MEM_WRAP_SIZE,
                            _267_CONCAT_15_OUTPUT_0_DEV_FMT>
                            (_267_Concat_15_output_0,
                            _267_CONCAT_15_OUTPUT_0_STREAM);
    fclose(_267_CONCAT_15_OUTPUT_0_STREAM);
    #endif



    
    *_260_Concat_14_output_0_ptr = _260_Concat_14_output_0;
    *_267_Concat_15_output_0_ptr = _267_Concat_15_output_0;
    *_94_PadMaxPool_0_output_1_ptr = _94_PadMaxPool_0_output_1;
    *_95_PadMaxPool_1_output_1_ptr = _95_PadMaxPool_1_output_1;
    *_96_PadMaxPool_2_output_1_ptr = _96_PadMaxPool_2_output_1;
}
