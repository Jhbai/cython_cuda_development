#include "alg.h"
#include "cuda/simd.cuh"
#include <stdio.h>

void c_process_add(float* a, float* b, float* out, int n) {
    launch_cuda_add(a, b, out, n);
}
