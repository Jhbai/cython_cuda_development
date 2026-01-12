#ifndef SIMD_CUH
#define SIMD_CUH

#ifdef __cplusplus
extern "C" {
#endif

void launch_cuda_add(float* a, float* b, float* out, int n);


#ifdef __cplusplus
}
#endif

#endif
