#ifndef __AIDGE_EXPORT_CPP_KERNELS_SPLIT__
#define __AIDGE_EXPORT_CPP_KERNELS_SPLIT__

#include "utils/cpp/typedefs.hpp"
#include "utils/cpp/utils.hpp"
#include <cstddef>

#ifndef SPLIT_INLINING
#define SPLIT_INLINING INLINE_DEFAULT
#endif

// Generic function for split

namespace export_cpp {

template <size_t NB_OUTPUTS,
          size_t AXIS_DIM,
          size_t PRE_AXIS_ELEMS,
          size_t POST_AXIS_ELEMS,
          const size_t SPLITS[NB_OUTPUTS],
          typename T>
SPLIT_INLINING void split_forward(const T *__restrict inputs,
                                  T **__restrict outputs)
{

    size_t offset = 0;
    const size_t groupStride = AXIS_DIM * POST_AXIS_ELEMS;
    for (size_t i = 0; i < NB_OUTPUTS; ++i) { // for each output
        size_t outputIdx = 0; // to map between input and output tensors

        // Iterate over the 3D view, [PRE_AXIS, SPLITS[i], POS_AXIS]
        for (size_t j = 0; j < PRE_AXIS_ELEMS; ++j) {
            for (size_t k = 0; k < SPLITS[i]; ++k) {
                const size_t chunkStart =
                    j * groupStride + (offset + k) * POST_AXIS_ELEMS;
                for (size_t l = 0; l < POST_AXIS_ELEMS; ++l) {
                    // Write to each output[NB_OUTPUTS]
                    outputs[i][outputIdx++] = inputs[chunkStart + l];
                }
            }
        }
        offset += SPLITS[i];
    }
}

} // namespace export_cpp

#endif // __AIDGE_EXPORT_CPP_KERNELS_SPLIT__
