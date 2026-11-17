#ifndef __AIDGE_EXPORT_CPP_KERNELS_CONVOLUTION__
#define __AIDGE_EXPORT_CPP_KERNELS_CONVOLUTION__

#include "utils/cpp/activation_utils.hpp"
#include "utils/cpp/macs.hpp"
#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"
#include "utils/cpp/utils.hpp"
#include <cstddef>

#ifndef CONVOLUTION_INLINING
#define CONVOLUTION_INLINING INLINE_DEFAULT
#endif

namespace export_cpp {

/**
 * @brief Conv kernel with batch support. Input and output are in NHWC format.
 * Weight is in OHWI format where O and I are the number of output and input
 * channels respectively.
 * @note No WRAP
 * @note Any padding
 */
template <
    size_t NB_BATCHES,
    size_t NB_CHANNELS,
    size_t CHANNELS_HEIGHT,
    size_t CHANNELS_WIDTH,
    size_t NB_OUTPUTS,
    size_t OUTPUTS_HEIGHT,
    size_t OUTPUTS_WIDTH,
    size_t PADDING_Y,
    size_t PADDING_X,
    size_t STRIDE_Y,
    size_t STRIDE_X,
    size_t DILATION_Y,
    size_t DILATION_X,
    size_t KERNEL_HEIGHT,
    size_t KERNEL_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    size_t INPUT_MEM_CONT_OFFSET,
    size_t INPUT_MEM_CONT_SIZE,
    size_t INPUT_MEM_WRAP_OFFSET,
    size_t INPUT_MEM_WRAP_SIZE,
    size_t INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    size_t OUTPUT_MEM_CONT_OFFSET,
    size_t OUTPUT_MEM_CONT_SIZE,
    size_t OUTPUT_MEM_WRAP_OFFSET,
    size_t OUTPUT_MEM_WRAP_SIZE,
    size_t OUTPUT_MEM_STRIDE,
    typename Input_T,
    typename Output_T,
    typename Weight_T,
    typename Bias_T,
    typename Rescaling_T,
    std::enable_if_t<!(INPUT_MEM_WRAP_SIZE > 0 || OUTPUT_MEM_WRAP_SIZE > 0),
                     int> = 0,
    std::enable_if_t<!(PADDING_X == 0 && PADDING_Y == 0), int> = 0>
CONVOLUTION_INLINING void
convolution_forward(const Input_T *__restrict inputs,
                    Output_T *__restrict outputs,
                    const Weight_T *__restrict weights,
                    const Bias_T *__restrict biases,
                    const Rescaling_T &__restrict rescaling)
{
    constexpr size_t NB_I_CHANNELS_PACKED =
        (NB_CHANNELS + n_pack<Input_T>() - 1) / n_pack<Input_T>();
    constexpr size_t NB_W_CHANNELS_PACKED =
        (NB_CHANNELS + n_pack<Weight_T>() - 1) / n_pack<Weight_T>();
    constexpr size_t NB_OUTPUTS_PACKED =
        (NB_OUTPUTS + n_pack<Output_T>() - 1) / n_pack<Output_T>();
    constexpr size_t NB_OUTPUTS_REM = NB_OUTPUTS % n_pack<Output_T>();

    constexpr size_t DILATED_KERNEL_HEIGHT =
        KERNEL_HEIGHT + (DILATION_Y - 1) * (KERNEL_HEIGHT - 1);

    constexpr size_t DILATED_KERNEL_WIDTH =
        KERNEL_WIDTH + (DILATION_X - 1) * (KERNEL_WIDTH - 1);

    constexpr size_t OUTPUTS_HEIGHT_NOPAD =
        (CHANNELS_HEIGHT - DILATION_Y * (KERNEL_HEIGHT - 1) - 1 + STRIDE_Y) /
        STRIDE_Y;
    constexpr size_t OUTPUTS_WIDTH_NOPAD =
        (CHANNELS_WIDTH - DILATION_X * (KERNEL_WIDTH - 1) - 1 + STRIDE_X) /
        STRIDE_X;

    constexpr size_t INPUT_BATCH_STRIDE =
        CHANNELS_HEIGHT * CHANNELS_WIDTH * (INPUT_MEM_STRIDE / sizeof(Input_T));
    constexpr size_t OUTPUT_BATCH_STRIDE =
        OUTPUTS_HEIGHT * OUTPUTS_WIDTH * (OUTPUT_MEM_STRIDE / sizeof(Output_T));

#ifdef _OPENMP
#pragma omp parallel for if (NB_BATCHES > 1)
#endif
    for (size_t batch = 0; batch < NB_BATCHES; ++batch) {
        const size_t iOffsetBatch = batch * INPUT_BATCH_STRIDE;
        const size_t oOffsetBatch = batch * OUTPUT_BATCH_STRIDE;

        for (size_t oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
            const size_t syMin =
                (PADDING_Y == 0) ? 0 : max(PADDING_Y - (oy * STRIDE_Y), 0);
            const size_t syMax =
                (PADDING_Y == 0 && OUTPUTS_HEIGHT == OUTPUTS_HEIGHT_NOPAD)
                    ? DILATED_KERNEL_HEIGHT
                    : clamp(CHANNELS_HEIGHT + PADDING_Y - (oy * STRIDE_Y),
                            0,
                            DILATED_KERNEL_HEIGHT);
            const int iy =
                static_cast<int>(oy * STRIDE_Y) - static_cast<int>(PADDING_Y);

            // oy loop should not be parallelized to allow memory wrapping:
            // memory lines must be processed in order.
#ifdef _OPENMP
            // Wrapping cannot occur in the middle of a line, except if there is
            // only one line (1D)!
#pragma omp parallel for collapse(2) if (NB_BATCHES == 1 &&                    \
                                             !(OUTPUT_MEM_WRAP_SIZE > 0 &&     \
                                                   KERNEL_WIDTH > 1 &&         \
                                                   OUTPUTS_HEIGHT == 1))
#endif
            for (size_t ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
                for (size_t outputPack = 0; outputPack < NB_OUTPUTS_PACKED;
                     ++outputPack)
                {
                    Output_T packedOutputVal;

                    const size_t oPos = (ox + OUTPUTS_WIDTH * oy);
                    int oOffset =
                        (OUTPUT_MEM_STRIDE / sizeof(Output_T)) * oPos +
                        oOffsetBatch;

                    for (size_t packElt = 0; packElt < n_pack<Output_T>();
                         ++packElt)
                    {
                        if (NB_OUTPUTS_REM != 0 && packElt == NB_OUTPUTS_REM) {
                            break;
                        }

                        const size_t output =
                            outputPack * n_pack<Output_T>() + packElt;

                        // moved to inner loop for collapsing -->
                        const size_t sxMin =
                            (PADDING_X == 0)
                                ? 0
                                : max(PADDING_X - (ox * STRIDE_X), 0);
                        const size_t sxMax =
                            (PADDING_X == 0 &&
                             OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                                ? DILATED_KERNEL_WIDTH
                                : clamp(CHANNELS_WIDTH + PADDING_X -
                                            (ox * STRIDE_X),
                                        0,
                                        DILATED_KERNEL_WIDTH);
                        const size_t ix = sxMin + ox * STRIDE_X - PADDING_X;

                        // <--
                        // Check if the biases are defined
                        Bias_T weightedSum =
                            biases ? biases[output] : Bias_T(0);

                        for (size_t sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                            if ((PADDING_Y != 0 ||
                                 OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD) &&
                                ((sy * DILATION_Y < syMin) ||
                                 (sy * DILATION_Y >= syMax)))
                            {
                                continue;
                            }

                            // sxMin must be included here, otherwise iPos could
                            // point to the previous line and wrapping not added
                            // correctly!
                            const size_t iPos =
                                ix + CHANNELS_WIDTH * (iy + sy * DILATION_Y);
                            int iOffset =
                                (INPUT_MEM_STRIDE / sizeof(Input_T)) * iPos +
                                iOffsetBatch;

                            const size_t wOffset =
                                (output * KERNEL_HEIGHT + sy) * KERNEL_WIDTH *
                                NB_W_CHANNELS_PACKED;

                            if (NB_I_CHANNELS_PACKED ==
                                    (INPUT_MEM_STRIDE / sizeof(Input_T)) &&
                                DILATION_X == 1 &&
                                ((PADDING_X == 0 &&
                                  OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD) ||
                                 sxMax - sxMin == KERNEL_WIDTH))
                            {
                                macsOnRange<KERNEL_WIDTH * NB_CHANNELS>(
                                    inputs + iOffset,
                                    weights + wOffset,
                                    weightedSum);
                            } else {
                                for (size_t sx = 0; sx < KERNEL_WIDTH; ++sx) {
                                    if ((PADDING_X != 0 ||
                                         OUTPUTS_WIDTH !=
                                             OUTPUTS_WIDTH_NOPAD) &&
                                        ((sx * DILATION_X < sxMin) ||
                                         (sx * DILATION_X >= sxMax)))
                                    {
                                        continue;
                                    }

                                    int iOffsetInRange =
                                        iOffset + (sx * DILATION_X - sxMin) *
                                                      (INPUT_MEM_STRIDE /
                                                       sizeof(Input_T));

                                    macsOnRange<NB_CHANNELS>(
                                        // same input line so no wrapping can
                                        // occur
                                        inputs + iOffsetInRange,
                                        weights + wOffset +
                                            sx * NB_W_CHANNELS_PACKED,
                                        weightedSum);
                                }
                            }
                        }

                        const auto outputVal =
                            activation_forward_value<Output_T>(
                                weightedSum, output, ACTIVATION, rescaling);
                        pack_rev_set(packedOutputVal, packElt, outputVal);
                    }

                    outputs[oOffset + outputPack] = packedOutputVal;
                }
            }
        }
    }
}

/**
 * @brief Conv kernel with batch support. Input and output are in NHWC format.
 * Weight is in OHWI format where O and I are the number of output and input
 * channels respectively.
 * @note No WRAP
 * @note No padding
 * @note Any kernel
 */
template <
    size_t NB_BATCHES,
    size_t NB_CHANNELS,
    size_t CHANNELS_HEIGHT,
    size_t CHANNELS_WIDTH,
    size_t NB_OUTPUTS,
    size_t OUTPUTS_HEIGHT,
    size_t OUTPUTS_WIDTH,
    size_t PADDING_Y,
    size_t PADDING_X,
    size_t STRIDE_Y,
    size_t STRIDE_X,
    size_t DILATION_Y,
    size_t DILATION_X,
    size_t KERNEL_HEIGHT,
    size_t KERNEL_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    size_t INPUT_MEM_CONT_OFFSET,
    size_t INPUT_MEM_CONT_SIZE,
    size_t INPUT_MEM_WRAP_OFFSET,
    size_t INPUT_MEM_WRAP_SIZE,
    size_t INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    size_t OUTPUT_MEM_CONT_OFFSET,
    size_t OUTPUT_MEM_CONT_SIZE,
    size_t OUTPUT_MEM_WRAP_OFFSET,
    size_t OUTPUT_MEM_WRAP_SIZE,
    size_t OUTPUT_MEM_STRIDE,
    typename Input_T,
    typename Output_T,
    typename Weight_T,
    typename Bias_T,
    typename Rescaling_T,
    std::enable_if_t<!(INPUT_MEM_WRAP_SIZE > 0 || OUTPUT_MEM_WRAP_SIZE > 0),
                     int> = 0,
    std::enable_if_t<(PADDING_X == 0 && PADDING_Y == 0), int> = 0,
    std::enable_if_t<!(KERNEL_HEIGHT == 1 && KERNEL_WIDTH == 1), int> = 0>
CONVOLUTION_INLINING void
convolution_forward(const Input_T *__restrict inputs,
                    Output_T *__restrict outputs,
                    const Weight_T *__restrict weights,
                    const Bias_T *__restrict biases,
                    const Rescaling_T &__restrict rescaling)
{
    constexpr size_t NB_I_CHANNELS_PACKED =
        (NB_CHANNELS + n_pack<Input_T>() - 1) / n_pack<Input_T>();
    constexpr size_t NB_W_CHANNELS_PACKED =
        (NB_CHANNELS + n_pack<Weight_T>() - 1) / n_pack<Weight_T>();
    constexpr size_t NB_OUTPUTS_PACKED =
        (NB_OUTPUTS + n_pack<Output_T>() - 1) / n_pack<Output_T>();
    constexpr size_t NB_OUTPUTS_REM = NB_OUTPUTS % n_pack<Output_T>();

    constexpr size_t DILATED_KERNEL_HEIGHT =
        KERNEL_HEIGHT + (DILATION_Y - 1) * (KERNEL_HEIGHT - 1);

    constexpr size_t DILATED_KERNEL_WIDTH =
        KERNEL_WIDTH + (DILATION_X - 1) * (KERNEL_WIDTH - 1);

    constexpr size_t OUTPUTS_HEIGHT_NOPAD =
        (CHANNELS_HEIGHT - DILATION_Y * (KERNEL_HEIGHT - 1) - 1 + STRIDE_Y) /
        STRIDE_Y;
    constexpr size_t OUTPUTS_WIDTH_NOPAD =
        (CHANNELS_WIDTH - DILATION_X * (KERNEL_WIDTH - 1) - 1 + STRIDE_X) /
        STRIDE_X;

    constexpr size_t INPUT_BATCH_STRIDE =
        CHANNELS_HEIGHT * CHANNELS_WIDTH * (INPUT_MEM_STRIDE / sizeof(Input_T));
    constexpr size_t OUTPUT_BATCH_STRIDE =
        OUTPUTS_HEIGHT * OUTPUTS_WIDTH * (OUTPUT_MEM_STRIDE / sizeof(Output_T));

#ifdef _OPENMP
#pragma omp parallel for if (NB_BATCHES > 1)
#endif
    for (size_t batch = 0; batch < NB_BATCHES; ++batch) {
        const size_t iOffsetBatch = batch * INPUT_BATCH_STRIDE;
        const size_t oOffsetBatch = batch * OUTPUT_BATCH_STRIDE;

        for (size_t oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
            const size_t syMin =
                (PADDING_Y == 0) ? 0 : max(PADDING_Y - (oy * STRIDE_Y), 0);
            const size_t syMax =
                (PADDING_Y == 0 && OUTPUTS_HEIGHT == OUTPUTS_HEIGHT_NOPAD)
                    ? DILATED_KERNEL_HEIGHT
                    : clamp(CHANNELS_HEIGHT + PADDING_Y - (oy * STRIDE_Y),
                            0,
                            DILATED_KERNEL_HEIGHT);
            const int iy =
                static_cast<int>(oy * STRIDE_Y) - static_cast<int>(PADDING_Y);

            // oy loop should not be parallelized to allow memory wrapping:
            // memory lines must be processed in order.
#ifdef _OPENMP
            // Wrapping cannot occur in the middle of a line, except if there is
            // only one line (1D)!
#pragma omp parallel for collapse(2) if (NB_BATCHES == 1 &&                    \
                                             !(OUTPUT_MEM_WRAP_SIZE > 0 &&     \
                                                   KERNEL_WIDTH > 1 &&         \
                                                   OUTPUTS_HEIGHT == 1))
#endif
            for (size_t ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
                for (size_t outputPack = 0; outputPack < NB_OUTPUTS_PACKED;
                     ++outputPack)
                {
                    Output_T packedOutputVal;

                    const size_t oPos = (ox + OUTPUTS_WIDTH * oy);
                    int oOffset =
                        (OUTPUT_MEM_STRIDE / sizeof(Output_T)) * oPos +
                        oOffsetBatch;

                    for (size_t packElt = 0; packElt < n_pack<Output_T>();
                         ++packElt)
                    {
                        if (NB_OUTPUTS_REM != 0 && packElt == NB_OUTPUTS_REM) {
                            break;
                        }

                        const size_t output =
                            outputPack * n_pack<Output_T>() + packElt;

                        // moved to inner loop for collapsing -->
                        const size_t sxMin =
                            (PADDING_X == 0)
                                ? 0
                                : max(PADDING_X - (ox * STRIDE_X), 0);
                        const size_t sxMax =
                            (PADDING_X == 0 &&
                             OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                                ? DILATED_KERNEL_WIDTH
                                : clamp(CHANNELS_WIDTH + PADDING_X -
                                            (ox * STRIDE_X),
                                        0,
                                        DILATED_KERNEL_WIDTH);
                        const int ix = static_cast<int>(ox * STRIDE_X) -
                                       static_cast<int>(PADDING_X);

                        // <--
                        // Check if the biases are defined
                        Bias_T weightedSum =
                            biases ? biases[output] : Bias_T(0);

                        for (size_t sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                            if ((PADDING_Y != 0 ||
                                 OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD) &&
                                ((sy * DILATION_Y < syMin) ||
                                 (sy * DILATION_Y >= syMax)))
                            {
                                continue;
                            }

                            // sxMin must be included here, otherwise iPos could
                            // point to the previous line and wrapping not added
                            // correctly!
                            const size_t iPos =
                                (ix + sxMin) +
                                CHANNELS_WIDTH * (iy + sy * DILATION_Y);
                            int iOffset =
                                (INPUT_MEM_STRIDE / sizeof(Input_T)) * iPos +
                                iOffsetBatch;

                            const size_t wOffset =
                                (output * KERNEL_HEIGHT + sy) * KERNEL_WIDTH *
                                NB_W_CHANNELS_PACKED;

                            if (NB_I_CHANNELS_PACKED ==
                                    (INPUT_MEM_STRIDE / sizeof(Input_T)) &&
                                DILATION_X == 1 &&
                                ((PADDING_X == 0 &&
                                  OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD) ||
                                 sxMax - sxMin == KERNEL_WIDTH))
                            {
                                macsOnRange<KERNEL_WIDTH * NB_CHANNELS>(
                                    inputs + iOffset,
                                    weights + wOffset,
                                    weightedSum);
                            } else {
                                for (size_t sx = 0; sx < KERNEL_WIDTH; ++sx) {
                                    if ((PADDING_X != 0 ||
                                         OUTPUTS_WIDTH !=
                                             OUTPUTS_WIDTH_NOPAD) &&
                                        ((sx * DILATION_X < sxMin) ||
                                         (sx * DILATION_X >= sxMax)))
                                    {
                                        continue;
                                    }

                                    int iOffsetInRange =
                                        iOffset + (sx * DILATION_X - sxMin) *
                                                      (INPUT_MEM_STRIDE /
                                                       sizeof(Input_T));

                                    macsOnRange<NB_CHANNELS>(
                                        // same input line so no wrapping can
                                        // occur
                                        inputs + iOffsetInRange,
                                        weights + wOffset +
                                            sx * NB_W_CHANNELS_PACKED,
                                        weightedSum);
                                }
                            }
                        }

                        const auto outputVal =
                            activation_forward_value<Output_T>(
                                weightedSum, output, ACTIVATION, rescaling);
                        pack_rev_set(packedOutputVal, packElt, outputVal);
                    }

                    outputs[oOffset + outputPack] = packedOutputVal;
                }
            }
        }
    }
}

/**
 * @brief Conv kernel with batch support. Input and output are in NHWC format.
 * Weight is in OHWI format where O and I are the number of output and input
 * channels respectively.
 * @note No WRAP
 * @note No padding
 * @note Pointwise
 */
template <
    size_t NB_BATCHES,
    size_t NB_CHANNELS,
    size_t CHANNELS_HEIGHT,
    size_t CHANNELS_WIDTH,
    size_t NB_OUTPUTS,
    size_t OUTPUTS_HEIGHT,
    size_t OUTPUTS_WIDTH,
    size_t PADDING_Y,
    size_t PADDING_X,
    size_t STRIDE_Y,
    size_t STRIDE_X,
    size_t DILATION_Y,
    size_t DILATION_X,
    size_t KERNEL_HEIGHT,
    size_t KERNEL_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    size_t INPUT_MEM_CONT_OFFSET,
    size_t INPUT_MEM_CONT_SIZE,
    size_t INPUT_MEM_WRAP_OFFSET,
    size_t INPUT_MEM_WRAP_SIZE,
    size_t INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    size_t OUTPUT_MEM_CONT_OFFSET,
    size_t OUTPUT_MEM_CONT_SIZE,
    size_t OUTPUT_MEM_WRAP_OFFSET,
    size_t OUTPUT_MEM_WRAP_SIZE,
    size_t OUTPUT_MEM_STRIDE,
    typename Input_T,
    typename Output_T,
    typename Weight_T,
    typename Bias_T,
    typename Rescaling_T,
    std::enable_if_t<!(INPUT_MEM_WRAP_SIZE > 0 || OUTPUT_MEM_WRAP_SIZE > 0),
                     int> = 0,
    std::enable_if_t<(PADDING_X == 0 && PADDING_Y == 0), int> = 0,
    std::enable_if_t<(KERNEL_HEIGHT == 1 && KERNEL_WIDTH == 1), int> = 0>
CONVOLUTION_INLINING void
convolution_forward(const Input_T *__restrict inputs,
                    Output_T *__restrict outputs,
                    const Weight_T *__restrict weights,
                    const Bias_T *__restrict biases,
                    const Rescaling_T &__restrict rescaling)
{
    // constexpr size_t NB_I_CHANNELS_PACKED =
    // (NB_CHANNELS + n_pack<Input_T>() - 1) / n_pack<Input_T>();
    constexpr size_t NB_W_CHANNELS_PACKED =
        (NB_CHANNELS + n_pack<Weight_T>() - 1) / n_pack<Weight_T>();
    constexpr size_t NB_OUTPUTS_PACKED =
        (NB_OUTPUTS + n_pack<Output_T>() - 1) / n_pack<Output_T>();
    constexpr size_t NB_OUTPUTS_REM = NB_OUTPUTS % n_pack<Output_T>();

    constexpr size_t INPUT_C_S = (INPUT_MEM_STRIDE / sizeof(Input_T));
    constexpr size_t INPUT_WC_S = CHANNELS_WIDTH * INPUT_C_S;
    constexpr size_t INPUT_HWC_S = CHANNELS_HEIGHT * INPUT_WC_S;

    constexpr size_t OUTPUT_C_S = (OUTPUT_MEM_STRIDE / sizeof(Output_T));
    constexpr size_t OUTPUT_WC_S = OUTPUTS_WIDTH * OUTPUT_C_S;
    constexpr size_t OUTPUT_HWC_S = OUTPUTS_HEIGHT * OUTPUT_WC_S;

#ifdef _OPENMP
#pragma omp parallel for if (NB_BATCHES > 1)
#endif
    for (size_t batch = 0; batch < NB_BATCHES; ++batch) {
        const size_t iOffsetBatch = batch * INPUT_HWC_S;
        const size_t oOffsetBatch = batch * OUTPUT_HWC_S;

        for (size_t oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
            const size_t iy = oy * STRIDE_Y;

            // oy loop should not be parallelized to allow memory wrapping:
            // memory lines must be processed in order.
#ifdef _OPENMP
            // Wrapping cannot occur in the middle of a line, except if there is
            // only one line (1D)!
#pragma omp parallel for collapse(2) if (NB_BATCHES == 1 &&                    \
                                             !(OUTPUTS_HEIGHT == 1))
#endif
            for (size_t ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
                for (size_t outputPack = 0; outputPack < NB_OUTPUTS_PACKED;
                     ++outputPack)
                {
                    Output_T packedOutputVal;

                    size_t oOffset =
                        ox * OUTPUT_C_S + oy * OUTPUT_WC_S + oOffsetBatch;

                    for (size_t packElt = 0; packElt < n_pack<Output_T>();
                         ++packElt)
                    {
                        if (NB_OUTPUTS_REM != 0 && packElt == NB_OUTPUTS_REM) {
                            break;
                        }

                        size_t output =
                            outputPack * n_pack<Output_T>() + packElt;

                        // moved to inner loop for collapsing -->
                        size_t ix = ox * STRIDE_X;

                        // <--
                        // Check if the biases are defined
                        Bias_T weightedSum =
                            biases ? biases[output] : Bias_T(0);

                        size_t iOffset =
                            ix * INPUT_C_S + iy * INPUT_WC_S + iOffsetBatch;

                        const size_t wOffset = output * KERNEL_HEIGHT *
                                               KERNEL_WIDTH *
                                               NB_W_CHANNELS_PACKED;
                        // shouldn't NB_I_CHANNELS_PACKED be used?

                        macsOnRange<KERNEL_WIDTH * NB_CHANNELS>(
                            inputs + iOffset, weights + wOffset, weightedSum);

                        const auto outputVal =
                            activation_forward_value<Output_T>(
                                weightedSum, output, ACTIVATION, rescaling);
                        pack_rev_set(packedOutputVal, packElt, outputVal);
                    }

                    outputs[oOffset + outputPack] = packedOutputVal;
                }
            }
        }
    }
}
// {
//     constexpr size_t INPUT_C_S = (INPUT_MEM_STRIDE / sizeof(Input_T));
//     constexpr size_t INPUT_WC_S = CHANNELS_WIDTH * INPUT_C_S;

//     constexpr size_t OUTPUT_C_S = (OUTPUT_MEM_STRIDE / sizeof(Output_T));
//     constexpr size_t OUTPUT_WC_S = OUTPUTS_WIDTH * OUTPUT_C_S;

//     constexpr size_t WEIGHT_HWI_S = KERNEL_HEIGHT * KERNEL_WIDTH *
//     NB_CHANNELS;

//     for (size_t oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
//         size_t iOffset = oy * STRIDE_Y * INPUT_WC_S;
//         size_t oOffset = oy * OUTPUT_WC_S;
//         for (size_t ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
//             size_t wOffset = 0;
//             if (NB_OUTPUTS % 4 == 0) {
//                 for (size_t oc = 0; oc < NB_OUTPUTS; oc += 4) {
//                     // Check if the biases are defined
//                     Bias_T weightedSum0 = biases ? biases[oc] : Bias_T(0);
//                     Bias_T weightedSum1 = biases ? biases[oc + 1] :
//                     Bias_T(0); Bias_T weightedSum2 = biases ? biases[oc + 2]
//                     : Bias_T(0); Bias_T weightedSum3 = biases ? biases[oc +
//                     3] : Bias_T(0);

//                     macsOnRange<WEIGHT_HWI_S>(
//                         inputs + iOffset, weights + wOffset, weightedSum0);
//                     macsOnRange<WEIGHT_HWI_S>(inputs + iOffset,
//                                               weights + wOffset +
//                                               WEIGHT_HWI_S, weightedSum1);
//                     macsOnRange<WEIGHT_HWI_S>(inputs + iOffset,
//                                               weights + wOffset +
//                                                   2 * WEIGHT_HWI_S,
//                                               weightedSum2);
//                     macsOnRange<WEIGHT_HWI_S>(inputs + iOffset,
//                                               weights + wOffset +
//                                                   3 * WEIGHT_HWI_S,
//                                               weightedSum3);

//                     outputs[oOffset + oc] =
//                     activation_forward_value<Output_T>(
//                         weightedSum0, oc, ACTIVATION, rescaling);
//                     outputs[oOffset + oc + 1] =
//                         activation_forward_value<Output_T>(
//                             weightedSum1, oc + 1, ACTIVATION, rescaling);
//                     outputs[oOffset + oc + 2] =
//                         activation_forward_value<Output_T>(
//                             weightedSum2, oc + 2, ACTIVATION, rescaling);
//                     outputs[oOffset + oc + 3] =
//                         activation_forward_value<Output_T>(
//                             weightedSum3, oc + 3, ACTIVATION, rescaling);

//                     wOffset += 4 * WEIGHT_HWI_S;
//                 }
//             } else {
//                 for (size_t oc = 0; oc < NB_OUTPUTS; ++oc) {
//                     // Check if the biases are defined
//                     Bias_T weightedSum = biases ? biases[oc] : Bias_T(0);

//                     macsOnRange<WEIGHT_HWI_S>(
//                         inputs + iOffset, weights + wOffset, weightedSum);

//                     outputs[oOffset + oc] =
//                     activation_forward_value<Output_T>(
//                         weightedSum, oc, ACTIVATION, rescaling);
//                     wOffset += WEIGHT_HWI_S;
//                 }
//             }
//             oOffset += OUTPUT_C_S;
//             iOffset += INPUT_C_S * STRIDE_X;
//         }
//     }
// }

/**
 * @brief Conv kernel with batch support. Input and output are in NHWC format.
 * Weight is in OHWI format where O and I are the number of output and input
 * channels respectively.
 * @note WRAP used.
 * @note Any padding.
 * @note Any kernel.
 */
template <
    size_t NB_BATCHES,
    size_t NB_CHANNELS,
    size_t CHANNELS_HEIGHT,
    size_t CHANNELS_WIDTH,
    size_t NB_OUTPUTS,
    size_t OUTPUTS_HEIGHT,
    size_t OUTPUTS_WIDTH,
    size_t PADDING_Y,
    size_t PADDING_X,
    size_t STRIDE_Y,
    size_t STRIDE_X,
    size_t DILATION_Y,
    size_t DILATION_X,
    size_t KERNEL_HEIGHT,
    size_t KERNEL_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE,
    typename Input_T,
    typename Output_T,
    typename Weight_T,
    typename Bias_T,
    typename Rescaling_T,
    std::enable_if_t<(INPUT_MEM_WRAP_SIZE > 0 || OUTPUT_MEM_WRAP_SIZE > 0),
                     int> = 0>
CONVOLUTION_INLINING void
convolution_forward(const Input_T *__restrict inputs,
                    Output_T *__restrict outputs,
                    const Weight_T *__restrict weights,
                    const Bias_T *__restrict biases,
                    const Rescaling_T &__restrict rescaling)
{
    constexpr size_t NB_I_CHANNELS_PACKED =
        (NB_CHANNELS + n_pack<Input_T>() - 1) / n_pack<Input_T>();
    constexpr size_t NB_W_CHANNELS_PACKED =
        (NB_CHANNELS + n_pack<Weight_T>() - 1) / n_pack<Weight_T>();
    constexpr size_t NB_OUTPUTS_PACKED =
        (NB_OUTPUTS + n_pack<Output_T>() - 1) / n_pack<Output_T>();
    constexpr size_t NB_OUTPUTS_REM = NB_OUTPUTS % n_pack<Output_T>();

    constexpr int DILATED_KERNEL_HEIGHT =
        KERNEL_HEIGHT + (DILATION_Y - 1) * (KERNEL_HEIGHT - 1);

    constexpr int DILATED_KERNEL_WIDTH =
        KERNEL_WIDTH + (DILATION_X - 1) * (KERNEL_WIDTH - 1);

    constexpr size_t OUTPUTS_HEIGHT_NOPAD =
        (CHANNELS_HEIGHT - DILATION_Y * (KERNEL_HEIGHT - 1) - 1 + STRIDE_Y) /
        STRIDE_Y;
    constexpr size_t OUTPUTS_WIDTH_NOPAD =
        (CHANNELS_WIDTH - DILATION_X * (KERNEL_WIDTH - 1) - 1 + STRIDE_X) /
        STRIDE_X;

    constexpr int INPUT_SPATIAL_SIZE =
        CHANNELS_HEIGHT * CHANNELS_WIDTH * (INPUT_MEM_STRIDE / sizeof(Input_T));
    constexpr int OUTPUT_SPATIAL_SIZE =
        OUTPUTS_HEIGHT * OUTPUTS_WIDTH * (OUTPUT_MEM_STRIDE / sizeof(Output_T));

#ifdef _OPENMP
#pragma omp parallel for if (NB_BATCHES > 1)
#endif
    for (size_t batch = 0; batch < NB_BATCHES; ++batch) {
        const int iOffsetBatch = batch * INPUT_SPATIAL_SIZE;
        const int oOffsetBatch = batch * OUTPUT_SPATIAL_SIZE;

        for (size_t oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
            const size_t syMin =
                (PADDING_Y == 0) ? 0 : max(PADDING_Y - (oy * STRIDE_Y), 0);
            const size_t syMax =
                (PADDING_Y == 0 && OUTPUTS_HEIGHT == OUTPUTS_HEIGHT_NOPAD)
                    ? DILATED_KERNEL_HEIGHT
                    : clamp(CHANNELS_HEIGHT + PADDING_Y - (oy * STRIDE_Y),
                            0,
                            DILATED_KERNEL_HEIGHT);
            const int iy =
                static_cast<int>(oy * STRIDE_Y) - static_cast<int>(PADDING_Y);

            // oy loop should not be parallelized to allow memory wrapping:
            // memory lines must be processed in order.
#ifdef _OPENMP
            // Wrapping cannot occur in the middle of a line, except if there is
            // only one line (1D)!
#pragma omp parallel for collapse(2) if (NB_BATCHES == 1 &&                    \
                                             !(OUTPUT_MEM_WRAP_SIZE > 0 &&     \
                                                   KERNEL_WIDTH > 1 &&         \
                                                   OUTPUTS_HEIGHT == 1))
#endif
            for (size_t ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
                for (size_t outputPack = 0; outputPack < NB_OUTPUTS_PACKED;
                     ++outputPack)
                {
                    Output_T packedOutputVal;

                    const size_t oPos = (ox + OUTPUTS_WIDTH * oy);
                    int oOffset =
                        (OUTPUT_MEM_STRIDE / sizeof(Output_T)) * oPos +
                        oOffsetBatch;

                    if (OUTPUT_MEM_WRAP_SIZE > 0 &&
                        oOffset >= OUTPUT_MEM_CONT_SIZE /
                                       static_cast<int>(sizeof(Output_T)))
                    {
                        oOffset +=
                            (OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET -
                             OUTPUT_MEM_CONT_SIZE) /
                            static_cast<int>(sizeof(Output_T));
                    }

                    for (size_t packElt = 0; packElt < n_pack<Output_T>();
                         ++packElt)
                    {
                        if (NB_OUTPUTS_REM != 0 && packElt == NB_OUTPUTS_REM) {
                            break;
                        }

                        const size_t output =
                            outputPack * n_pack<Output_T>() + packElt;

                        // moved to inner loop for collapsing -->
                        const size_t sxMin =
                            (PADDING_X == 0)
                                ? 0
                                : max(PADDING_X - (ox * STRIDE_X), 0);
                        const size_t sxMax =
                            (PADDING_X == 0 &&
                             OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                                ? DILATED_KERNEL_WIDTH
                                : clamp(CHANNELS_WIDTH + PADDING_X -
                                            (ox * STRIDE_X),
                                        0,
                                        DILATED_KERNEL_WIDTH);
                        const int ix = static_cast<int>(ox * STRIDE_X) -
                                       static_cast<int>(PADDING_X);

                        // <--
                        // Check if the biases are defined
                        Bias_T weightedSum =
                            biases ? biases[output] : Bias_T(0);

                        for (size_t sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                            if ((PADDING_Y != 0 ||
                                 OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD) &&
                                ((sy * DILATION_Y < syMin) ||
                                 (sy * DILATION_Y >= syMax)))
                            {
                                continue;
                            }

                            // sxMin must be included here, otherwise iPos could
                            // point to the previous line and wrapping not added
                            // correctly!
                            const size_t iPos =
                                (ix + sxMin) +
                                CHANNELS_WIDTH * (iy + sy * DILATION_Y);
                            int iOffset =
                                (INPUT_MEM_STRIDE / sizeof(Input_T)) * iPos +
                                iOffsetBatch;

                            // Wrapping cannot occur in the middle of a line,
                            // except if there is only one line (1D)!
                            bool wrapInRange = false;

                            if (INPUT_MEM_WRAP_SIZE > 0 &&
                                iOffset >=
                                    INPUT_MEM_CONT_SIZE /
                                        static_cast<int>(sizeof(Input_T)))
                            {
                                iOffset += (INPUT_MEM_WRAP_OFFSET -
                                            INPUT_MEM_CONT_OFFSET -
                                            INPUT_MEM_CONT_SIZE) /
                                           static_cast<int>(sizeof(Input_T));
                            } else if (INPUT_MEM_WRAP_SIZE > 0 &&
                                       KERNEL_WIDTH > 1 &&
                                       CHANNELS_HEIGHT ==
                                           1 && // single line (1D)!
                                       iOffset - iOffsetBatch +
                                               KERNEL_WIDTH *
                                                   NB_I_CHANNELS_PACKED >
                                           (INPUT_MEM_CONT_SIZE /
                                            static_cast<int>(sizeof(Input_T))))
                            {
                                wrapInRange = true;
                            }

                            const size_t wOffset =
                                (output * KERNEL_HEIGHT + sy) * KERNEL_WIDTH *
                                NB_W_CHANNELS_PACKED;

                            if (!wrapInRange &&
                                NB_I_CHANNELS_PACKED ==
                                    (INPUT_MEM_STRIDE / sizeof(Input_T)) &&
                                DILATION_X == 1 &&
                                ((PADDING_X == 0 &&
                                  OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD) ||
                                 sxMax - sxMin == KERNEL_WIDTH))
                            {
                                macsOnRange<KERNEL_WIDTH * NB_CHANNELS>(
                                    inputs + iOffset,
                                    weights + wOffset,
                                    weightedSum);
                            } else {
                                for (size_t sx = 0; sx < KERNEL_WIDTH; ++sx) {
                                    if ((PADDING_X != 0 ||
                                         OUTPUTS_WIDTH !=
                                             OUTPUTS_WIDTH_NOPAD) &&
                                        ((sx * DILATION_X < sxMin) ||
                                         (sx * DILATION_X >= sxMax)))
                                    {
                                        continue;
                                    }

                                    int iOffsetInRange =
                                        iOffset + (sx * DILATION_X - sxMin) *
                                                      (INPUT_MEM_STRIDE /
                                                       sizeof(Input_T));

                                    if (wrapInRange &&
                                        iOffsetInRange - iOffsetBatch >=
                                            INPUT_MEM_CONT_SIZE /
                                                static_cast<int>(
                                                    sizeof(Input_T)))
                                    {
                                        iOffsetInRange +=
                                            (INPUT_MEM_WRAP_OFFSET -
                                             INPUT_MEM_CONT_OFFSET -
                                             INPUT_MEM_CONT_SIZE) /
                                            static_cast<int>(sizeof(Input_T));
                                    }

                                    macsOnRange<NB_CHANNELS>(
                                        // same input line so no wrapping can
                                        // occur
                                        inputs + iOffsetInRange,
                                        weights + wOffset +
                                            sx * NB_W_CHANNELS_PACKED,
                                        weightedSum);
                                }
                            }
                        }

                        const auto outputVal =
                            activation_forward_value<Output_T>(
                                weightedSum, output, ACTIVATION, rescaling);
                        pack_rev_set(packedOutputVal, packElt, outputVal);
                    }

                    outputs[oOffset + outputPack] = packedOutputVal;
                }
            }
        }
    }
}

// Template overloading when biases are not given to the convolution
template <size_t NB_BATCHES,
          size_t NB_CHANNELS,
          size_t CHANNELS_HEIGHT,
          size_t CHANNELS_WIDTH,
          size_t NB_OUTPUTS,
          size_t OUTPUTS_HEIGHT,
          size_t OUTPUTS_WIDTH,
          size_t PADDING_Y,
          size_t PADDING_X,
          size_t STRIDE_Y,
          size_t STRIDE_X,
          size_t DILATION_Y,
          size_t DILATION_X,
          size_t KERNEL_HEIGHT,
          size_t KERNEL_WIDTH,
          ActivationFunction_T ACTIVATION,
          // Memory mapping: inputs
          size_t INPUT_MEM_CONT_OFFSET,
          size_t INPUT_MEM_CONT_SIZE,
          size_t INPUT_MEM_WRAP_OFFSET,
          size_t INPUT_MEM_WRAP_SIZE,
          size_t INPUT_MEM_STRIDE,
          // Memory mapping: outputs
          size_t OUTPUT_MEM_CONT_OFFSET,
          size_t OUTPUT_MEM_CONT_SIZE,
          size_t OUTPUT_MEM_WRAP_OFFSET,
          size_t OUTPUT_MEM_WRAP_SIZE,
          size_t OUTPUT_MEM_STRIDE,
          typename Input_T,
          typename Output_T,
          typename Weight_T,
          typename Rescaling_T>
CONVOLUTION_INLINING void
convolution_forward(const Input_T *__restrict inputs,
                    Output_T *__restrict outputs,
                    const Weight_T *__restrict weights,
                    std::nullptr_t,
                    const Rescaling_T &__restrict rescaling)
{
    const float *b = nullptr;

    convolution_forward<NB_BATCHES,
                        NB_CHANNELS,
                        CHANNELS_HEIGHT,
                        CHANNELS_WIDTH,
                        NB_OUTPUTS,
                        OUTPUTS_HEIGHT,
                        OUTPUTS_WIDTH,
                        PADDING_Y,
                        PADDING_X,
                        STRIDE_Y,
                        STRIDE_X,
                        DILATION_Y,
                        DILATION_X,
                        KERNEL_HEIGHT,
                        KERNEL_WIDTH,
                        ACTIVATION,
                        // Memory mapping: inputs
                        INPUT_MEM_CONT_OFFSET,
                        INPUT_MEM_CONT_SIZE,
                        INPUT_MEM_WRAP_OFFSET,
                        INPUT_MEM_WRAP_SIZE,
                        INPUT_MEM_STRIDE,
                        // Memory mapping: outputs
                        OUTPUT_MEM_CONT_OFFSET,
                        OUTPUT_MEM_CONT_SIZE,
                        OUTPUT_MEM_WRAP_OFFSET,
                        OUTPUT_MEM_WRAP_SIZE,
                        OUTPUT_MEM_STRIDE>(
        inputs, outputs, weights, b, rescaling);
}

} // namespace export_cpp

#endif // __AIDGE_EXPORT_CPP_KERNELS_CONVOLUTION__
