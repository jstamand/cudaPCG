/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "PCG_Kernels.h"

__device__ MatStruct getSubMat(MatStruct Mat, size_t blockRow, size_t blockCol, size_t blockSize)
{
	MatStruct subMat;
	subMat.pitch = Mat.pitch;
	subMat.data = (float*)((char*)Mat.data + blockRow*blockSize*Mat.pitch) + blockCol*blockSize;
	return subMat;
}

__device__ float* GetSubMatrix(float* A, int blockRow, int blockCol, size_t pitch)
{
	return (float*)((char*)A + LOCAL_WORKGROUP_DIM*blockRow*pitch) + LOCAL_WORKGROUP_DIM * blockCol;
}

__global__ void Kernel_UpdateResidual(VecStruct Vec_B, MatStruct Mat_A, VecStruct Vec_X, VecStruct Vec_R)
{
	float CValue = 0.0;
	MatStruct subMat;

	//add dummy +1 to avoid shared bank conflict
	__shared__ float strip[LOCAL_WORKGROUP_DIM];
	__shared__ float block[LOCAL_WORKGROUP_DIM+1][LOCAL_WORKGROUP_DIM];

	//calculation loop
	for(int m = 0; m < (Mat_A.paddedWidth/LOCAL_WORKGROUP_DIM); m++){

		//load a strip
		strip[threadIdx.x] = Vec_X.data[threadIdx.x + m*LOCAL_WORKGROUP_DIM];

		//load a block cooperatively
		subMat = getSubMat(Mat_A, blockIdx.x, m, LOCAL_WORKGROUP_DIM);
		//#pragma unroll
		for(int i = 0; i < LOCAL_WORKGROUP_DIM; i++)
			block[i][threadIdx.x] = *((float*)((char*)subMat.data + i*Mat_A.pitch) + threadIdx.x + m*LOCAL_WORKGROUP_DIM);

		//cycle through updating each running total
		//#pragma unroll
		for(int i = 0; i < LOCAL_WORKGROUP_DIM; i++){
			CValue += strip[i]*block[threadIdx.x][i];
		}
		__syncthreads();
	}

	//load from vector b
	float VecElement = Vec_B.data[threadIdx.x + blockDim.x*blockIdx.x];

	Vec_R.data[threadIdx.x + blockDim.x*blockIdx.x] = VecElement - CValue;
}

__global__ void Kernel_MatrixVectorMultiply(MatStruct Mat_A, VecStruct Vec_B, VecStruct Vec_Result)
{
	float CValue = 0.0;
	MatStruct subMat;

	__shared__ float strip[LOCAL_WORKGROUP_DIM];
	__shared__ float block[LOCAL_WORKGROUP_DIM+1][LOCAL_WORKGROUP_DIM];

	//calculation loop
	#pragma unroll
	for(int m = 0; m < Mat_A.paddedWidth/LOCAL_WORKGROUP_DIM; m++){

		//load a strip
		strip[threadIdx.x] = Vec_B.data[threadIdx.x + m*LOCAL_WORKGROUP_DIM];

		//load a block cooperatively
		subMat = getSubMat(Mat_A, blockIdx.x, m, LOCAL_WORKGROUP_DIM);
		//#pragma unroll
		for(int i = 0; i < LOCAL_WORKGROUP_DIM; i++)
			block[i][threadIdx.x] = *((float*)((char*)subMat.data + i*Mat_A.pitch) + threadIdx.x + m*LOCAL_WORKGROUP_DIM);

		//cycle through and update a running total
		//#pragma unroll
		for(int i = 0; i < LOCAL_WORKGROUP_DIM; i++)
			CValue += strip[i]*block[threadIdx.x][i];

		__syncthreads();
	}

	Vec_Result.data[threadIdx.x + blockDim.x*blockIdx.x] = CValue;
}

__global__ void Kernel_VVAdd(VecStruct Vec_d, VecStruct Vec_r, float* Beta)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	Vec_d.data[tid] = Vec_r.data[tid] + Beta[0]*Vec_d.data[tid];
}

__global__ void Kernel_DotProduct(VecStruct Vec_A, VecStruct Vec_B, float* result)
{
	unsigned int tid = threadIdx.x;
	float temp;
	__shared__ float share[LOCAL_WORKGROUP_DIM];

	//reduce vectors to width of one block
	temp = 0.0;
	while(tid < Vec_A.paddedLength){
		temp += Vec_A.data[tid]*Vec_B.data[tid];
		tid += LOCAL_WORKGROUP_DIM;
	}
	share[threadIdx.x] = temp;
	__syncthreads();

	//groups of 16 execute synchronously
	if(LOCAL_WORKGROUP_DIM >= 1024){
		if(threadIdx.x < 512)
			share[threadIdx.x] = share[threadIdx.x] + share[threadIdx.x + 512];
	__syncthreads();
	}

	if(LOCAL_WORKGROUP_DIM >= 512){
			if(threadIdx.x < 256)
				share[threadIdx.x] = share[threadIdx.x] + share[threadIdx.x + 256];
	__syncthreads();
	}

	if(LOCAL_WORKGROUP_DIM >= 256){
			if(threadIdx.x < 128)
				share[threadIdx.x] = share[threadIdx.x] + share[threadIdx.x + 128];
	__syncthreads();
	}

	if(LOCAL_WORKGROUP_DIM >= 128){
			if(threadIdx.x < 64)
				share[threadIdx.x] = share[threadIdx.x] + share[threadIdx.x + 64];
	__syncthreads();
	}

	if(LOCAL_WORKGROUP_DIM >= 64){
			if(threadIdx.x < 32)
				share[threadIdx.x] = share[threadIdx.x] + share[threadIdx.x + 32];
	__syncthreads();
	}


	//groups of 16 execute synchronously
	if(threadIdx.x < 32 && LOCAL_WORKGROUP_DIM >= 32)
				share[threadIdx.x] = share[threadIdx.x] + share[threadIdx.x + 16];
	if(threadIdx.x < 8)
				share[threadIdx.x] = share[threadIdx.x] + share[threadIdx.x + 8];
	if(threadIdx.x < 4)
				share[threadIdx.x] = share[threadIdx.x] + share[threadIdx.x + 4];
	if(threadIdx.x < 2)
				share[threadIdx.x] = share[threadIdx.x] + share[threadIdx.x + 2];
	if(threadIdx.x < 1)
				result[0] = share[threadIdx.x] + share[threadIdx.x + 1];
}
