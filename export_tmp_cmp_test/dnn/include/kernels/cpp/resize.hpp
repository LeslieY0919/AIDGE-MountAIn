#ifndef __AIDGE_EXPORT_CPP_KERNELS_RESIZE__
#define __AIDGE_EXPORT_CPP_KERNELS_RESIZE__

#include "utils/cpp/typedefs.hpp"
#include "utils/cpp/utils.hpp"
#include <algorithm>
#include <cmath>

#ifndef RESIZE_INLINING
#define RESIZE_INLINING INLINE_DEFAULT
#endif

namespace export_cpp {

template <typename T> inline T _clamp(T v, T lo, T hi)
{
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

template <int MEM_CONT_SIZE,
          int MEM_WRAP_OFFSET,
          int MEM_WRAP_SIZE,
          int MEM_CONT_OFFSET,
          int MEM_STRIDE,
          int CHANNELS_WIDTH,
          typename T>
RESIZE_INLINING static int offsetWrapLogic(int _x, int _y)
{
    int offset = MEM_CONT_OFFSET + MEM_STRIDE * (_x + CHANNELS_WIDTH * _y);
    if (MEM_WRAP_SIZE > 0 && offset >= MEM_CONT_OFFSET + MEM_CONT_SIZE) {
        offset += MEM_WRAP_OFFSET - MEM_CONT_OFFSET - MEM_CONT_SIZE;
    }
    return offset / static_cast<int>(sizeof(T));
}

template <CoordTrans_T COORD>
inline float map_coord(int dst_idx,
                       const int src_size,
                       const int dst_size,
                       const float scale)
{

    static_assert(COORD == CoordTrans_T::HalfPixel ||
                      COORD == CoordTrans_T::AlignCorners ||
                      COORD == CoordTrans_T::Asymmetric ||
                      COORD == CoordTrans_T::HalfPixelSymmetric ||
                      COORD == CoordTrans_T::PytorchHalfPixel ||
                      COORD == CoordTrans_T::TFHalfPixelForNN,
                  "Unhandled CoordTrans_T value");

    if (COORD == CoordTrans_T::HalfPixel) {
        return (static_cast<float>(dst_idx) + 0.5f) / scale - 0.5f;

    } else if (COORD == CoordTrans_T::AlignCorners) {
        return (dst_size == 1) ? 0.0f
                               : static_cast<float>(dst_idx) *
                                     static_cast<float>(src_size - 1) /
                                     static_cast<float>(dst_size - 1);

    } else if (COORD == CoordTrans_T::Asymmetric) {
        return static_cast<float>(dst_idx) / scale;

    } else if (COORD == CoordTrans_T::HalfPixelSymmetric) {
        return (static_cast<float>(dst_idx) + 0.5f) / scale - 0.5f;

    } else if (COORD == CoordTrans_T::PytorchHalfPixel) {
        return (dst_size > 1)
                   ? (static_cast<float>(dst_idx) + 0.5f) / scale - 0.5f
                   : 0.0f;

    } else if (COORD == CoordTrans_T::TFHalfPixelForNN) {
        return (static_cast<float>(dst_idx) + 0.5f) / scale;
    }
}

template <Interpolation_T INTERP> inline int nearest_neighbor(float v, int max)
{
    int rounded;

    if (INTERP == Interpolation_T::Floor)
        rounded = static_cast<int>(floorf(v));
    else if (INTERP == Interpolation_T::Ceil)
        rounded = static_cast<int>(ceilf(v));
    else if (INTERP == Interpolation_T::RoundPreferFloor)
        rounded = (v - floorf(v)) == 0.5f ? static_cast<int>(floorf(v))
                                          : static_cast<int>(roundf(v));
    else if (INTERP == Interpolation_T::RoundPreferCeil)
        rounded = (v - floorf(v)) == 0.5f ? static_cast<int>(ceilf(v))
                                          : static_cast<int>(roundf(v));

    return _clamp(rounded, 0, max);
}

template <int INPUT_CHANNELS_WIDTH,
          int INPUT_CHANNELS_HEIGHT,
          int OUTPUTS_WIDTH,
          int OUTPUTS_HEIGHT,
          int OUTPUTS_NB_OUTPUTS,
          bool ANTIALIAS,
          typename Input_T,
          typename Output_T,
          typename OffsetWrap_T>
RESIZE_INLINING void linearInterpolate(const Input_T *inputs,
                                       Output_T *outputs,
                                       float src_x,
                                       float src_y,
                                       int oOffset,
                                       OffsetWrap_T inputOffsetWrap,
                                       [[maybe_unused]] const float scale_x,
                                       [[maybe_unused]] const float scale_y)
{

    if (ANTIALIAS) {

        const float support_x = (scale_x < 1.f) ? (1.f / scale_x) : 1.f;
        const float support_y = (scale_y < 1.f) ? (1.f / scale_y) : 1.f;

        const int ix0 = static_cast<int>(ceilf(src_x - support_x));
        const int ix1 = static_cast<int>(floorf(src_x + support_x));
        const int iy0 = static_cast<int>(ceilf(src_y - support_y));
        const int iy1 = static_cast<int>(floorf(src_y + support_y));

        auto triangle = [](float t, float support) -> float {
            return std::max(0.f, 1.f - std::abs(t) / support);
        };

        float acc[OUTPUTS_NB_OUTPUTS] = {};
        float weight_sum = 0.f;

        for (int iy = iy0; iy <= iy1; ++iy) {
            const float wy =
                triangle(src_y - static_cast<float>(iy), support_y);
            if (wy == 0.f)
                continue;

            for (int ix = ix0; ix <= ix1; ++ix) {
                const float wx =
                    triangle(src_x - static_cast<float>(ix), support_x);
                if (wx == 0.f)
                    continue;

                const float w = wx * wy;

                const int cx = _clamp(ix, 0, INPUT_CHANNELS_WIDTH - 1);
                const int cy = _clamp(iy, 0, INPUT_CHANNELS_HEIGHT - 1);
                const int idx = inputOffsetWrap(cx, cy);

                for (int c = 0; c < OUTPUTS_NB_OUTPUTS; ++c)
                    acc[c] += static_cast<float>(inputs[idx + c]) * w;

                weight_sum += w;
            }
        }

        if (weight_sum > 0.f) {
            const float inv_ws = 1.f / weight_sum;
            for (int c = 0; c < OUTPUTS_NB_OUTPUTS; ++c)
                outputs[oOffset + c] = static_cast<Output_T>(acc[c] * inv_ws);
        }
    } else { // Anti-alias false

        const int ix0 = static_cast<int>(floorf(src_x));
        const int iy0 = static_cast<int>(floorf(src_y));

        const int ix1 = std::min(ix0 + 1, INPUT_CHANNELS_WIDTH - 1);
        const int iy1 = std::min(iy0 + 1, INPUT_CHANNELS_HEIGHT - 1);

        const float dx = src_x - floorf(src_x);
        const float dy = src_y - floorf(src_y);

        const int i00 = inputOffsetWrap(ix0, iy0);
        const int i01 = inputOffsetWrap(ix0, iy1);
        const int i10 = inputOffsetWrap(ix1, iy0);
        const int i11 = inputOffsetWrap(ix1, iy1);

        for (int c = 0; c < OUTPUTS_NB_OUTPUTS; ++c) {
            const float interp =
                static_cast<float>(inputs[i00 + c]) * (1.f - dy) * (1.f - dx) +
                static_cast<float>(inputs[i01 + c]) * dy * (1.f - dx) +
                static_cast<float>(inputs[i10 + c]) * (1.f - dy) * dx +
                static_cast<float>(inputs[i11 + c]) * dy * dx;

            outputs[oOffset + c] = static_cast<Output_T>(interp);
        }
    }
}

RESIZE_INLINING static float cubic_weight(float x, float a)
{
    x = fabsf(x);
    if (x < 1.0f)
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    if (x < 2.0f)
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    return 0.0f;
}

RESIZE_INLINING static float
cubic_weight_antialiased(float x, float scale, float a)
{
    const float s = std::min(scale, 1.0f);
    x = fabsf(x * s);
    const float x2 = x * x;
    const float x3 = x2 * x;

    if (x <= 1.0f)
        return (a + 2.0f) * x3 - (a + 3.0f) * x2 + 1.0f;
    if (x < 2.0f)
        return a * x3 - 5.0f * a * x2 + 8.0f * a * x - 4.0f * a;
    return 0.0f;
}

template <bool ANTIALIAS,
          bool EXCLUDE_OUTSIDE,
          int INPUT_CHANNELS_WIDTH,
          int INPUT_CHANNELS_HEIGHT,
          int OUTPUTS_NB_OUTPUTS,
          typename Input_T,
          typename Output_T,
          typename OffsetWrap_T>
RESIZE_INLINING static void cubicInterpolate(const Input_T *inputs,
                                             Output_T *outputs,
                                             float src_x,
                                             float src_y,
                                             int oOffset,
                                             OffsetWrap_T inputOffsetWrap,
                                             float cubic_a,
                                             [[maybe_unused]] const float s_x,
                                             [[maybe_unused]] const float s_y,
                                             const float support_x,
                                             const float support_y,
                                             const int window_x,
                                             const int window_y)
{
    const int ix_center = static_cast<int>(std::floor(src_x));
    const int iy_center = static_cast<int>(std::floor(src_y));

    const int ix_start = ix_center - static_cast<int>(std::ceil(support_x)) + 1;
    const int iy_start = iy_center - static_cast<int>(std::ceil(support_y)) + 1;

    constexpr int MAX_WINDOW = 16;

    int row_iy[MAX_WINDOW];
    int row_ciy[MAX_WINDOW];
    bool row_valid[MAX_WINDOW];
    float row_weights_y[MAX_WINDOW];

    float y_wsum = 0.f;

    for (int k = 0; k < window_y; ++k) {
        row_iy[k] = iy_start + k;
        row_ciy[k] = _clamp(row_iy[k], 0, INPUT_CHANNELS_HEIGHT - 1);

        if (EXCLUDE_OUTSIDE)
            row_valid[k] =
                (row_iy[k] >= 0 && row_iy[k] < INPUT_CHANNELS_HEIGHT);
        else
            row_valid[k] = true;

        const float dy = src_y - static_cast<float>(row_iy[k]);

        if (ANTIALIAS)
            row_weights_y[k] = cubic_weight_antialiased(dy, s_y, cubic_a);
        else
            row_weights_y[k] = cubic_weight(dy, cubic_a);

        if (row_valid[k])
            y_wsum += row_weights_y[k];
    }

    const float inv_y_wsum = (y_wsum > 0.f) ? 1.f / y_wsum : 0.f;

    int tap_ix[MAX_WINDOW];
    int tap_cix[MAX_WINDOW];
    bool tap_valid[MAX_WINDOW];
    float x_weights[MAX_WINDOW];

    float x_wsum = 0.f;

    for (int j = 0; j < window_x; ++j) {
        tap_ix[j] = ix_start + j;
        tap_cix[j] = _clamp(tap_ix[j], 0, INPUT_CHANNELS_WIDTH - 1);

        if (EXCLUDE_OUTSIDE)
            tap_valid[j] = (tap_ix[j] >= 0 && tap_ix[j] < INPUT_CHANNELS_WIDTH);
        else
            tap_valid[j] = true;

        const float dx = src_x - static_cast<float>(tap_ix[j]);

        if (ANTIALIAS)
            x_weights[j] = cubic_weight_antialiased(dx, s_x, cubic_a);
        else
            x_weights[j] = cubic_weight(dx, cubic_a);

        if (tap_valid[j])
            x_wsum += x_weights[j];
    }

    const float inv_x_wsum = (x_wsum > 0.f) ? 1.f / x_wsum : 0.f;

    int input_offsets[MAX_WINDOW][MAX_WINDOW];

    for (int k = 0; k < window_y; ++k) {
        if (!row_valid[k])
            continue;
        for (int j = 0; j < window_x; ++j) {
            if (!tap_valid[j])
                continue;
            input_offsets[k][j] = inputOffsetWrap(tap_cix[j], row_ciy[k]);
        }
    }

    for (int c = 0; c < OUTPUTS_NB_OUTPUTS; ++c) {

        float y_result = 0.f;

        for (int k = 0; k < window_y; ++k) {
            if (!row_valid[k])
                continue;

            // Pass 1: X interpolation for this row
            float x_result = 0.f;

            for (int j = 0; j < window_x; ++j) {
                if (!tap_valid[j])
                    continue;
                x_result +=
                    static_cast<float>(inputs[input_offsets[k][j] + c]) *
                    x_weights[j];
            }

            // Pass 2: accumulate into Y
            y_result += (x_result * inv_x_wsum) * row_weights_y[k];
        }

        outputs[oOffset + c] = static_cast<Output_T>(y_result * inv_y_wsum);
    }
}

template <int NB_BATCHES,
          int INPUT_NB_CHANNELS,
          int INPUT_CHANNELS_HEIGHT,
          int INPUT_CHANNELS_WIDTH,
          int OUTPUTS_NB_OUTPUTS,
          int OUTPUTS_HEIGHT,
          int OUTPUTS_WIDTH,
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
          const bool ANTIALIAS,
          const bool EXCLUDE_OUTSIDE,
          typename Input_T,
          typename Output_T,
          // Behavior
          Interpolation_T INTERP = Interpolation_T::RoundPreferFloor,
          CoordTrans_T COORD = CoordTrans_T::HalfPixel,
          std::enable_if_t<(NB_BATCHES == 1), int> = 0>
RESIZE_INLINING static void resize_forward(const Input_T *__restrict inputs,
                                           Output_T *__restrict outputs,
                                           const float cubic_a = -0.75f)

{
    auto inputOffsetWrap = [&](int ix, int iy) {
        return offsetWrapLogic<INPUT_MEM_CONT_SIZE,
                               INPUT_MEM_WRAP_OFFSET,
                               INPUT_MEM_WRAP_SIZE,
                               INPUT_MEM_CONT_OFFSET,
                               INPUT_MEM_STRIDE,
                               INPUT_CHANNELS_WIDTH,
                               Input_T>(ix, iy);
    };

    auto outputOffsetWrap = [&](int ox, int oy) {
        return offsetWrapLogic<OUTPUT_MEM_CONT_SIZE,
                               OUTPUT_MEM_WRAP_OFFSET,
                               OUTPUT_MEM_WRAP_SIZE,
                               OUTPUT_MEM_CONT_OFFSET,
                               OUTPUT_MEM_STRIDE,
                               OUTPUTS_WIDTH,
                               Output_T>(ox, oy);
    };

    const float scale_x =
        static_cast<float>(OUTPUTS_WIDTH) / INPUT_CHANNELS_WIDTH;
    const float scale_y =
        static_cast<float>(OUTPUTS_HEIGHT) / INPUT_CHANNELS_HEIGHT;

    /////////////////////////////////----------------------/////////////////////////////////
    // In the case of Cubic

    const float s_x = std::min(scale_x, 1.0f);
    const float s_y = std::min(scale_y, 1.0f);

    const float support_x = ANTIALIAS ? (2.0f / s_x) : 2.0f;
    const float support_y = ANTIALIAS ? (2.0f / s_y) : 2.0f;

    const int window_x = 2 * static_cast<int>(std::ceil(support_x));
    const int window_y = 2 * static_cast<int>(std::ceil(support_y));

    /////////////////////////////////----------------------/////////////////////////////////

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {

            int oOffset = outputOffsetWrap(ox, oy);
            const float src_x = map_coord<COORD>(
                ox, INPUT_CHANNELS_WIDTH, OUTPUTS_WIDTH, scale_x);
            const float src_y = map_coord<COORD>(
                oy, INPUT_CHANNELS_HEIGHT, OUTPUTS_HEIGHT, scale_y);

            if (INTERP == Interpolation_T::Floor ||
                INTERP == Interpolation_T::Ceil ||
                INTERP == Interpolation_T::RoundPreferFloor ||
                INTERP == Interpolation_T::RoundPreferCeil)
            {
                const int ix =
                    nearest_neighbor<INTERP>(src_x, INPUT_CHANNELS_WIDTH - 1);
                const int iy =
                    nearest_neighbor<INTERP>(src_y, INPUT_CHANNELS_HEIGHT - 1);

                int iOffset = inputOffsetWrap(ix, iy);

                // Write result to output
                for (int c = 0; c < OUTPUTS_NB_OUTPUTS; ++c)
                    outputs[oOffset + c] =
                        static_cast<Output_T>(inputs[iOffset + c]);

            } else if (INTERP == Interpolation_T::Linear) {

                const float clamped_src_x =
                    _clamp(src_x, 0.f, INPUT_CHANNELS_WIDTH - 1.f);
                const float clamped_src_y =
                    _clamp(src_y, 0.f, INPUT_CHANNELS_HEIGHT - 1.f);

                linearInterpolate<INPUT_CHANNELS_WIDTH,
                                  INPUT_CHANNELS_HEIGHT,
                                  OUTPUTS_WIDTH,
                                  OUTPUTS_HEIGHT,
                                  OUTPUTS_NB_OUTPUTS,
                                  ANTIALIAS,
                                  Input_T,
                                  Output_T,
                                  decltype(inputOffsetWrap)>(inputs,
                                                             outputs,
                                                             clamped_src_x,
                                                             clamped_src_y,
                                                             oOffset,
                                                             inputOffsetWrap,
                                                             scale_x,
                                                             scale_y);

            } else if (INTERP == Interpolation_T::Cubic) {

                cubicInterpolate<ANTIALIAS,
                                 EXCLUDE_OUTSIDE,
                                 INPUT_CHANNELS_WIDTH,
                                 INPUT_CHANNELS_HEIGHT,
                                 OUTPUTS_NB_OUTPUTS,
                                 Input_T,
                                 Output_T,
                                 decltype(inputOffsetWrap)>(inputs,
                                                            outputs,
                                                            src_x,
                                                            src_y,
                                                            oOffset,
                                                            inputOffsetWrap,
                                                            cubic_a,
                                                            s_x,
                                                            s_y,
                                                            support_x,
                                                            support_y,
                                                            window_x,
                                                            window_y);
            }
        }
    }
}

} // namespace export_cpp

#endif // __AIDGE_EXPORT_CPP_KERNELS_RESIZE__
