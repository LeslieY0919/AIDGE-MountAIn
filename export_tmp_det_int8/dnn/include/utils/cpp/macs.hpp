#ifndef __AIDGE_EXPORT_CPP_MACS_HPP__
#define __AIDGE_EXPORT_CPP_MACS_HPP__

#include "typedefs.hpp"
#include <cstdint>
#include <type_traits>

#if defined(__has_include)
#if __has_include(<bit>)
#include <bit>
#define HAS_STD_POPCOUNT_HEADER 1
#endif
#endif

#if defined(HAS_STD_POPCOUNT_HEADER) && defined(__cpp_lib_bitops) &&           \
    __cpp_lib_bitops >= 201907L

namespace bit_utils {

using std::popcount;

} // namespace bit_utils

#else

namespace bit_utils {

inline int popcount(std::uint8_t x)
{
    x = x - ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    return (x + (x >> 4)) & 0x0F;
}

} // namespace bit_utils

#endif
namespace export_cpp {

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T>
inline static typename std::enable_if_t<NB_ITERATIONS == 0>
macsOnRange(const Input_T *__restrict /*inputs*/,
            const Weight_T *__restrict /*weights*/,
            Sum_T &__restrict /*weightedSum*/)
{
}

// ----------------------------------------------------------------------------
// Packed <*> weights - Unpacked inputs
// ----------------------------------------------------------------------------

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T>
static typename std::enable_if_t<(is_packed<Weight_T>() &&
                                  n_pack<Weight_T>() < 8 &&
                                  NB_ITERATIONS == 1 && !is_packed<Input_T>())>
macsOnRange(const Input_T *__restrict inputs,
            const Weight_T *__restrict weights,
            Sum_T &__restrict weightedSum)
{
    weightedSum +=
        static_cast<Sum_T>(*inputs) * weights[0 * WEIGHTS_INC].rev_fields.op0;
}

// ----------------------------------------------------------------------------
// Packed <2> weights - Unpacked inputs
// ----------------------------------------------------------------------------

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<(n_pack<Weight_T>() == 2 &&
                                   !is_packed<Input_T>())>::type * = nullptr>
static void dualMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[1 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 2 && NB_ITERATIONS >= 2 &&
                             !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    dualMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 2, INPUTS_INC, WEIGHTS_INC>(
        inputs + 2 * INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Packed <2> weights - Packed <2> inputs
// ----------------------------------------------------------------------------

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<(is_packed<Weight_T>() &&
                                   n_pack<Weight_T>() == n_pack<Input_T>() &&
                                   NB_ITERATIONS == 1)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                   weights[0 * WEIGHTS_INC].rev_fields.op0;
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<(n_pack<Weight_T>() == 2 &&
                                   n_pack<Input_T>() == 2)>::type * = nullptr>
static void dualMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 2 && NB_ITERATIONS >= 2 &&
                             n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    dualMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 2, INPUTS_INC, WEIGHTS_INC>(
        inputs + INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Unpacked weights - Packed <*> inputs
// ----------------------------------------------------------------------------

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(!is_packed<Weight_T>() && NB_ITERATIONS == 1 &&
                             is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                   weights[0 * WEIGHTS_INC];
}

// ----------------------------------------------------------------------------
// Unpacked weights - Packed <2> inputs
// ----------------------------------------------------------------------------

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<(!is_packed<Weight_T>() &&
                                   n_pack<Input_T>() == 2)>::type * = nullptr>
static void dualMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC] +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) *
                       weights[1 * WEIGHTS_INC];
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(!is_packed<Weight_T>() && NB_ITERATIONS >= 2 &&
                             n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    dualMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 2, INPUTS_INC, WEIGHTS_INC>(
        inputs + INPUTS_INC, weights + 2 * WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Packed <4> weights - Unpacked inputs
// ----------------------------------------------------------------------------

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS == 2 &&
                             !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[1 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS == 3 &&
                             !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[1 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1 +
                   static_cast<Sum_T>(inputs[2 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op2;
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<(n_pack<Weight_T>() == 4 &&
                                   !is_packed<Input_T>())>::type * = nullptr>
static void quadMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[1 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1 +
                   static_cast<Sum_T>(inputs[2 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op2 +
                   static_cast<Sum_T>(inputs[3 * INPUTS_INC]) *
                       weights[0 * WEIGHTS_INC].rev_fields.op3;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS >= 4 &&
                             !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    quadMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 4, INPUTS_INC, WEIGHTS_INC>(
        inputs + 4 * INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Packed <4> weights - Packed <2> inputs
// ----------------------------------------------------------------------------

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS == 1 &&
                             n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                   weights[0 * WEIGHTS_INC].rev_fields.op0;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS == 2 &&
                             n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS == 3 &&
                             n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1 +
                   static_cast<Sum_T>(inputs[1 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC].rev_fields.op2;
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<(n_pack<Weight_T>() == 4 &&
                                   n_pack<Input_T>() == 2)>::type * = nullptr>
static void quadMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1 +
                   static_cast<Sum_T>(inputs[1 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC].rev_fields.op2 +
                   static_cast<Sum_T>(inputs[1 * INPUTS_INC].rev_fields.op1) *
                       weights[0 * WEIGHTS_INC].rev_fields.op3;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS >= 4 &&
                             n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    quadMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 4, INPUTS_INC, WEIGHTS_INC>(
        inputs + 2 * INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Packed <4> weights - Packed <4> inputs
// ----------------------------------------------------------------------------

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS == 1 &&
                             n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                   weights[0 * WEIGHTS_INC].rev_fields.op0;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS == 2 &&
                             n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS == 3 &&
                             n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op2) *
                       weights[0 * WEIGHTS_INC].rev_fields.op2;
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<(n_pack<Weight_T>() == 4 &&
                                   n_pack<Input_T>() == 4)>::type * = nullptr>
static void quadMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) *
                       weights[0 * WEIGHTS_INC].rev_fields.op0 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) *
                       weights[0 * WEIGHTS_INC].rev_fields.op1 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op2) *
                       weights[0 * WEIGHTS_INC].rev_fields.op2 +
                   static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op3) *
                       weights[0 * WEIGHTS_INC].rev_fields.op3;
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 4 && NB_ITERATIONS >= 4 &&
                             n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    quadMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 4, INPUTS_INC, WEIGHTS_INC>(
        inputs + INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Packed <8> weights - Unpacked inputs
// ----------------------------------------------------------------------------

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 1 && !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum +=
        ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0) ? (Sum_T)(-(*inputs))
                                                        : (Sum_T)(*inputs));
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 2 && !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC])
                        : (Sum_T)inputs[0 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC])
                        : (Sum_T)inputs[1 * INPUTS_INC]);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 3 && !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC])
                        : (Sum_T)inputs[0 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC])
                        : (Sum_T)inputs[1 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC])
                        : (Sum_T)inputs[2 * INPUTS_INC]);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 4 && !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC])
                        : (Sum_T)inputs[0 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC])
                        : (Sum_T)inputs[1 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC])
                        : (Sum_T)inputs[2 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[3 * INPUTS_INC])
                        : (Sum_T)inputs[3 * INPUTS_INC]);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 5 && !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC])
                        : (Sum_T)inputs[0 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC])
                        : (Sum_T)inputs[1 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC])
                        : (Sum_T)inputs[2 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[3 * INPUTS_INC])
                        : (Sum_T)inputs[3 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[4 * INPUTS_INC])
                        : (Sum_T)inputs[4 * INPUTS_INC]);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 6 && !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC])
                        : (Sum_T)inputs[0 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC])
                        : (Sum_T)inputs[1 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC])
                        : (Sum_T)inputs[2 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[3 * INPUTS_INC])
                        : (Sum_T)inputs[3 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[4 * INPUTS_INC])
                        : (Sum_T)inputs[4 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[5 * INPUTS_INC])
                        : (Sum_T)inputs[5 * INPUTS_INC]);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 7 && !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC])
                        : (Sum_T)inputs[0 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC])
                        : (Sum_T)inputs[1 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC])
                        : (Sum_T)inputs[2 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[3 * INPUTS_INC])
                        : (Sum_T)inputs[3 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[4 * INPUTS_INC])
                        : (Sum_T)inputs[4 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[5 * INPUTS_INC])
                        : (Sum_T)inputs[5 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op6 == 0)
                        ? (Sum_T)(-inputs[6 * INPUTS_INC])
                        : (Sum_T)inputs[6 * INPUTS_INC]);
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<(n_pack<Weight_T>() == 8 &&
                                   !std::is_unsigned<Weight_T>() &&
                                   !is_packed<Input_T>())>::type * = nullptr>
static void octoMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC])
                        : (Sum_T)inputs[0 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC])
                        : (Sum_T)inputs[1 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC])
                        : (Sum_T)inputs[2 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[3 * INPUTS_INC])
                        : (Sum_T)inputs[3 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[4 * INPUTS_INC])
                        : (Sum_T)inputs[4 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[5 * INPUTS_INC])
                        : (Sum_T)inputs[5 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op6 == 0)
                        ? (Sum_T)(-inputs[6 * INPUTS_INC])
                        : (Sum_T)inputs[6 * INPUTS_INC]) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op7 == 0)
                        ? (Sum_T)(-inputs[7 * INPUTS_INC])
                        : (Sum_T)inputs[7 * INPUTS_INC]);
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 8 && NB_ITERATIONS >= 8 &&
                             !is_packed<Input_T>())>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    octoMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 8, INPUTS_INC, WEIGHTS_INC>(
        inputs + 8 * INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Packed <8> weights - Packed <2> inputs
// ----------------------------------------------------------------------------

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 1 && n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 2 && n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 3 && n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 4 && n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op1);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 5 && n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[2 * INPUTS_INC].rev_fields.op0);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 6 && n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[2 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[2 * INPUTS_INC].rev_fields.op1);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 7 && n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[2 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[2 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op6 == 0)
                        ? (Sum_T)(-inputs[3 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[3 * INPUTS_INC].rev_fields.op0);
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<(n_pack<Weight_T>() == 8 &&
                                   !std::is_unsigned<Weight_T>() &&
                                   n_pack<Input_T>() == 2)>::type * = nullptr>
static void octoMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[2 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[2 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[2 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op6 == 0)
                        ? (Sum_T)(-inputs[3 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[3 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op7 == 0)
                        ? (Sum_T)(-inputs[3 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[3 * INPUTS_INC].rev_fields.op1);
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 8 && NB_ITERATIONS >= 8 &&
                             n_pack<Input_T>() == 2)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    octoMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 8, INPUTS_INC, WEIGHTS_INC>(
        inputs + 4 * INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Packed <8> weights - Packed <4> inputs
// ----------------------------------------------------------------------------

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 1 && n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 2 && n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 3 && n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 4 && n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op3);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 5 && n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op3) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 6 && n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op3) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op1);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 7 && n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op3) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op6 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op2);
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<(n_pack<Weight_T>() == 8 &&
                                   !std::is_unsigned<Weight_T>() &&
                                   n_pack<Input_T>() == 4)>::type * = nullptr>
static void octoMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op3) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op6 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op7 == 0)
                        ? (Sum_T)(-inputs[1 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[1 * INPUTS_INC].rev_fields.op3);
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 8 && NB_ITERATIONS >= 8 &&
                             n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    octoMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 8, INPUTS_INC, WEIGHTS_INC>(
        inputs + 2 * INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Packed <8> weights - Packed <8> inputs
// ----------------------------------------------------------------------------

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 1 && n_pack<Input_T>() == 8)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 2 && n_pack<Input_T>() == 8)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 3 && n_pack<Input_T>() == 8)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 4 && n_pack<Input_T>() == 8)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op3);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 5 && n_pack<Input_T>() == 8)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op3) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op4)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op4);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 6 && n_pack<Input_T>() == 8)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op3) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op4)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op4) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op5)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op5);
}

template <size_t NB_ITERATIONS,
          size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Weight_T,
          typename Sum_T,
          class Input_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               NB_ITERATIONS == 7 && n_pack<Input_T>() == 8)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum += ((weights[0 * WEIGHTS_INC].rev_fields.op0 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op0)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op0) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op1 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op1)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op1) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op2 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op2)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op2) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op3 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op3)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op3) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op4 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op4)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op4) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op5 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op5)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op5) +
                   ((weights[0 * WEIGHTS_INC].rev_fields.op6 == 0)
                        ? (Sum_T)(-inputs[0 * INPUTS_INC].rev_fields.op6)
                        : (Sum_T)inputs[0 * INPUTS_INC].rev_fields.op6);
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<(
              n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
              n_pack<Input_T>() == 8 && !std::is_unsigned<Input_T>())>::type * =
              nullptr>
static void octoMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    const int mismatches =
        bit_utils::popcount(weights[0 * WEIGHTS_INC] ^ inputs[0 * INPUTS_INC]);

    weightedSum += Sum_T(8) - 2 * mismatches;
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && !std::is_unsigned<Weight_T>() &&
               n_pack<Input_T>() == 8 && std::is_unsigned<Input_T>())>::type * =
              nullptr>
static void octoMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += 2 * bit_utils::popcount(weights[0 * WEIGHTS_INC] &
                                           inputs[0 * INPUTS_INC]) -
                   bit_utils::popcount(inputs[0 * INPUTS_INC]);
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<(
              n_pack<Weight_T>() == 8 && std::is_unsigned<Weight_T>() &&
              n_pack<Input_T>() == 8 && !std::is_unsigned<Input_T>())>::type * =
              nullptr>
static void octoMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum += 2 * bit_utils::popcount(weights[0 * WEIGHTS_INC] &
                                           inputs[0 * INPUTS_INC]) -
                   bit_utils::popcount(weights[0 * INPUTS_INC]);
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<
              (n_pack<Weight_T>() == 8 && std::is_unsigned<Weight_T>() &&
               n_pack<Input_T>() == 8 && std::is_unsigned<Input_T>())>::type * =
              nullptr>
static void octoMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum +=
        bit_utils::popcount(weights[0 * WEIGHTS_INC] & inputs[0 * INPUTS_INC]);
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(n_pack<Weight_T>() == 8 && NB_ITERATIONS >= 8 &&
                             n_pack<Input_T>() == 8)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    octoMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 8, INPUTS_INC, WEIGHTS_INC>(
        inputs + INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Unpacked weights - Packed <4> inputs
// ----------------------------------------------------------------------------

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(!is_packed<Weight_T>() && NB_ITERATIONS == 2 &&
                             n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum +=
        weights[0 * WEIGHTS_INC] *
            static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) +
        weights[1 * WEIGHTS_INC] *
            static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1);
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(!is_packed<Weight_T>() && NB_ITERATIONS == 3 &&
                             n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    weightedSum +=
        weights[0 * WEIGHTS_INC] *
            static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) +
        weights[1 * WEIGHTS_INC] *
            static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) +
        weights[2 * WEIGHTS_INC] *
            static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op2);
}

template <size_t INPUTS_INC = 1,
          size_t WEIGHTS_INC = 1,
          typename Input_T,
          typename Weight_T,
          typename Sum_T,
          typename std::enable_if<(!is_packed<Weight_T>() &&
                                   n_pack<Input_T>() == 4)>::type * = nullptr>
static void quadMac(const Input_T *__restrict inputs,
                    const Weight_T *__restrict weights,
                    Sum_T &__restrict weightedSum)
{
    weightedSum +=
        weights[0 * WEIGHTS_INC] *
            static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op0) +
        weights[1 * WEIGHTS_INC] *
            static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op1) +
        weights[2 * WEIGHTS_INC] *
            static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op2) +
        weights[3 * WEIGHTS_INC] *
            static_cast<Sum_T>(inputs[0 * INPUTS_INC].rev_fields.op3);
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(!is_packed<Weight_T>() && NB_ITERATIONS >= 4 &&
                             n_pack<Input_T>() == 4)>::type * = nullptr>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    quadMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
    macsOnRange<NB_ITERATIONS - 4, INPUTS_INC, WEIGHTS_INC>(
        inputs + INPUTS_INC, weights + 4 * WEIGHTS_INC, weightedSum);
}

// ----------------------------------------------------------------------------
// Unpacked weights - Unpacked inputs
// ----------------------------------------------------------------------------

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(NB_ITERATIONS > 0 && !is_packed<Weight_T>() &&
                             !is_packed<Input_T>())>::type * = nullptr,
    typename std::enable_if_t<(INPUTS_INC == 1 && WEIGHTS_INC == 1), int> = 0>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    for (size_t iter = 0; iter < NB_ITERATIONS; ++iter) {
        weightedSum += static_cast<Sum_T>(inputs[iter]) * weights[iter];
    }
}

template <
    size_t NB_ITERATIONS,
    size_t INPUTS_INC = 1,
    size_t WEIGHTS_INC = 1,
    typename Weight_T,
    typename Sum_T,
    class Input_T,
    typename std::enable_if<(NB_ITERATIONS > 0 && !is_packed<Weight_T>() &&
                             !is_packed<Input_T>())>::type * = nullptr,
    typename std::enable_if_t<!(INPUTS_INC == 1 && WEIGHTS_INC == 1), int> = 0>
static void macsOnRange(const Input_T *__restrict inputs,
                        const Weight_T *__restrict weights,
                        Sum_T &__restrict weightedSum)
{
    for (size_t iter = 0; iter < NB_ITERATIONS; ++iter) {
        weightedSum += static_cast<Sum_T>(inputs[iter * INPUTS_INC]) *
                       weights[iter * WEIGHTS_INC];
    }
}

}; // namespace export_cpp

#endif
