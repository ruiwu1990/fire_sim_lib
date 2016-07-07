#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H
#include <math.h>
#include <stdio.h>

const int INF = 32767;

__global__ void Accelerate(float*, float*, float*, int, float);

__global__ void TestCrownRate(float*, float*, float*, int,
                              float*, float*);

__device__ float Clamp(float, float, float);

#endif