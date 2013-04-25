/*
 * PCG_Kernels.h
 *
 *  Created on: Apr 21, 2013
 *      Author: airthimble
 */

#ifndef PCG_KERNELS_H_
#define PCG_KERNELS_H_
#include "common.h"
#include "MatStruct.h"
#include "VecStruct.h"

__device__ MatStruct getSubMat(MatStruct Mat, size_t blockRow, size_t blockCol, size_t blockSize);
__device__ float* GetSubMatrix(float* A, int blockRow, int blockCol, size_t pitch);
__global__ void Kernel_UpdateResidual(VecStruct Vec_B, MatStruct Mat_A, VecStruct Vec_X, VecStruct Vec_R);
__global__ void Kernel_DotProduct(VecStruct Vec_A, VecStruct Vec_B, float* result);
__global__ void Kernel_MatrixVectorMultiply(MatStruct Mat_A, VecStruct Vec_B, VecStruct Vec_Result);
__global__ void Kernel_VVAdd(VecStruct Vec_d, VecStruct Vec_r, float* Beta);

#endif /* PCG_KERNELS_H_ */
