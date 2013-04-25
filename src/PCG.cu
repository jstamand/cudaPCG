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
#include "PCG.h"
#include "PCG_Kernels.h"

void PCG::checkProgress()
{
	Vector progressVec(dev_Vec_X->getLength(), HOST, true);
	dev_Vec_X->transfer(&progressVec, cudaMemcpyDeviceToHost, true);
	for(int i = 0; i < progressVec.getLength(); i++)
		std::cout << progressVec.getData()[i] << std::endl;
}

void PCG::init() throw(std::runtime_error)
{

	iterations = 0;
	Update_Residual();

	//copy residual vector into direction vector
	HANDLE_ERROR(cudaMemcpy(dev_Vec_B->getData(), dev_Vec_R->getData(), dev_Vec_B->getPaddedLength()*sizeof(float), cudaMemcpyDeviceToDevice));
	std::cout << "Printing Vec R\n";
	dev_Vec_R->print();
	std::cout << "Printing Vec B\n";
	dev_Vec_B->print();

	//copy residual vector back for testing
	Vector* hostResidual = new Vector(dev_Vec_R->getLength(), HOST, LOCAL_WORKGROUP_DIM);
	dev_Vec_R->transfer(hostResidual, cudaMemcpyDeviceToHost, true);

	//calculate delta_new
	Kernel_DotProduct<<<vecBlocks,vecThreads>>>(dev_Vec_R->getStruct(), dev_Vec_R->getStruct(), dev_delta_new);

	//transfer delta_new back
	HANDLE_ERROR(cudaMemcpy(&delta_new, dev_delta_new, sizeof(float), cudaMemcpyDeviceToHost));

	delta_base = delta_new;
}

void PCG::Update_Residual()
{
	Kernel_UpdateResidual<<<vecBlocks, vecThreads>>>(dev_Vec_B->getStruct(), dev_Mat_A->getStruct(), dev_Vec_X->getStruct(), dev_Vec_R->getStruct());
}

void PCG::Update_Alpha()
{
	Kernel_DotProduct<<<vecBlocks,vecThreads>>>(dev_Vec_D->getStruct(), dev_Vec_Q->getStruct(), dev_alpha);
	HANDLE_ERROR(cudaMemcpy(&alpha, dev_alpha, sizeof(float), cudaMemcpyDeviceToHost));
	alpha = delta_new/alpha;
}

void PCG::Update_X()
{
	Kernel_VVAdd<<<vecBlocks,vecThreads>>>(dev_Vec_X->getStruct(), dev_Vec_D->getStruct(), dev_alpha);
}

void PCG::Update_Deltas()
{
	delta_old = delta_new;
	dev_Vec_R->print();
	Kernel_DotProduct<<<vecBlocks,vecThreads>>>(dev_Vec_R->getStruct(), dev_Vec_R->getStruct(), dev_delta_new);

	//transfer delta_new back
	HANDLE_ERROR(cudaMemcpy(&delta_new, dev_delta_new, sizeof(float), cudaMemcpyDeviceToHost));
}

void PCG::Update_direction()
{
	Kernel_DotProduct<<<vecBlocks,vecThreads>>>(dev_Vec_D->getStruct(), dev_Vec_R->getStruct(), dev_Beta);
}

void PCG::Update_Q()
{
	Kernel_MatrixVectorMultiply<<<vecBlocks,vecThreads>>>(dev_Mat_A->getStruct(), dev_Vec_D->getStruct(), dev_Vec_Q->getStruct());
}

void PCG::solve()
{
	//init the process
	init();

	while((iterations < maxIterations) && (delta_new > epsilon*epsilon*delta_base))
	{
		Update_Q();
		Update_Alpha();
		Update_X();
		Update_Residual();
		Update_Deltas();
		Beta = delta_new / delta_old;
		Update_direction();
		iterations++;

		checkProgress();
	}
}

PCG::PCG(Vector* dev_Vec_X, Vector* dev_Vec_B, Matrix* dev_Mat_A, int maxIterations) throw(std::logic_error)
		:dev_Vec_X(dev_Vec_X), dev_Vec_B(dev_Vec_B), dev_Mat_A(dev_Mat_A), maxIterations(maxIterations), iterations(0)
{
	if((dev_Vec_X->getLocation() != DEVICE) || (dev_Vec_B->getLocation() != DEVICE)
		|| (dev_Mat_A->getLocation() != DEVICE))
		throw(std::logic_error("All matrices must be on the device!"));

	//allocate needed things on the GPU
	HANDLE_ERROR(cudaMalloc(&dev_alpha, sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&dev_Beta, sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&dev_delta_new, sizeof(float)));

	dev_Vec_R = new Vector(dev_Vec_B->getLength(), DEVICE, LOCAL_WORKGROUP_DIM);
	dev_Vec_D = new Vector(dev_Vec_B->getLength(), DEVICE, LOCAL_WORKGROUP_DIM);
	dev_Vec_Q = new Vector(dev_Vec_B->getLength(), DEVICE, LOCAL_WORKGROUP_DIM);

	//pre-determine blocks/threads needed
	vecThreads.x = LOCAL_WORKGROUP_DIM;
	vecThreads.y = 1;
	vecThreads.z = 1;

	vecBlocks.x = dev_Vec_B->getPaddedLength() / LOCAL_WORKGROUP_DIM;
	vecBlocks.y = 1;
	vecBlocks.z = 1;

	epsilon = 0.001;
}
