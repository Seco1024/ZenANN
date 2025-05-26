#pragma once
#include <cstddef>
#include <immintrin.h>

namespace zenann {
inline float l2_simd(const float* __restrict a,
                     const float* __restrict b,
                     size_t dim) {
#if defined(__AVX2__)
    const size_t step = 8;            // 8 × 32-bit floats
    __m256 acc       = _mm256_setzero_ps();
    size_t i         = 0;
    for (; i + step - 1 < dim; i += step) {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc         = _mm256_fmadd_ps(diff, diff, acc);   // acc += diff²
    }
    float buf[step];
    _mm256_storeu_ps(buf, acc);
    float d = 0.f;
    for (int j = 0; j < step; ++j) d += buf[j];

    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
#else
    float d = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
#endif
}

}  
