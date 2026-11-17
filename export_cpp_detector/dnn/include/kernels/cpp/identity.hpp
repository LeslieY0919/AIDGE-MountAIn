#ifndef __AIDGE_EXPORT_CPP_KERNELS_IDENTITY__
#define __AIDGE_EXPORT_CPP_KERNELS_IDENTITY__

#include "utils/cpp/typedefs.hpp"
#include "utils/cpp/utils.hpp"
#include <cstddef>
#include <cstring>

#ifndef IDENTITY_INLINING
#define IDENTITY_INLINING INLINE_DEFAULT
#endif

// Generic function for identity and activation

namespace export_cpp {

template <size_t NB_ELTS,
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
          typename Output_T>
IDENTITY_INLINING void identity_forward(const Input_T *__restrict inputs,
                                        Output_T *__restrict outputs)
{
    // If inputs and outputs pointers are the same, the memory manager has
    // already optimized this function so it is a no-op !
    if (static_cast<const void *>(inputs) == static_cast<const void *>(outputs))
        return;

    // NB_ELTS is the number of input elements!
    constexpr auto NB_BYTES = NB_ELTS * sizeof(Input_T);
    const auto *in = reinterpret_cast<const unsigned char *>(inputs);
    auto *out = reinterpret_cast<unsigned char *>(outputs);

    // A identity in c++ world should equal to a Noop
    // We only need to copy the input buffer to the output
    size_t copied = 0;
    while (copied < NB_BYTES) {
        size_t inOffset = copied;
        size_t outOffset = copied;

        size_t inRemaining =
            static_cast<size_t>(INPUT_MEM_CONT_SIZE) > copied
                ? static_cast<size_t>(INPUT_MEM_CONT_SIZE) - copied
                : 0;

        size_t outRemaining =
            static_cast<size_t>(OUTPUT_MEM_CONT_SIZE) > copied
                ? static_cast<size_t>(OUTPUT_MEM_CONT_SIZE) - copied
                : 0;

        // If we've crossed the contiguous region, apply the wrap offset.
        if (INPUT_MEM_WRAP_SIZE > 0 &&
            copied >= static_cast<size_t>(INPUT_MEM_CONT_SIZE))
        {
            inOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET -
                        INPUT_MEM_CONT_SIZE;
        }

        if (OUTPUT_MEM_WRAP_SIZE > 0 &&
            copied >= static_cast<size_t>(OUTPUT_MEM_CONT_SIZE))
        {
            outOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET -
                         OUTPUT_MEM_CONT_SIZE;
        }

        size_t chunk = NB_BYTES - copied;

        if (inRemaining != 0)
            chunk = std::min(chunk, inRemaining);

        if (outRemaining != 0)
            chunk = std::min(chunk, outRemaining);

        std::memcpy(out + outOffset, in + inOffset, chunk);

        copied += chunk;
    }
}

} // namespace export_cpp

#endif // __AIDGE_EXPORT_CPP_KERNELS_IDENTITY__
