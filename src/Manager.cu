/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "Manager.h"
#include "PCG.h"

Manager::Manager(char* fileX, char* matX, char* fileY, char* vecY)
{
	//load data from file system
	Y = matInterface.loadVector(fileY, vecY);
	A = matInterface.loadMatrix(fileX, matX, ROW_MAJOR);

	HANDLE_ERROR(cudaStreamCreate(&stream1));
	testPCG();
}

void Manager::testPCG()
{
	std::cout << "Testing PCG(just CG for now)..." << std::endl;

	X = new Vector(A->getWidth(), HOST, LOCAL_WORKGROUP_DIM);
	X->setVector(1.0);
	X->setPadding(0.0);
	Vector* dev_Vec_X = new Vector(X->getLength(), DEVICE, LOCAL_WORKGROUP_DIM);
	X->transferAsync(dev_Vec_X, stream1, cudaMemcpyHostToDevice, true);
	cudaStreamSynchronize(stream1);

	Vector* dev_Vec_Y = new Vector(Y->getLength(), DEVICE, LOCAL_WORKGROUP_DIM);
	X->setPadding(0.0);
	Y->transferAsync(dev_Vec_Y, stream1, cudaMemcpyHostToDevice, true);
	cudaStreamSynchronize(stream1);

	Matrix* dev_Mat_A = new Matrix(A->getWidth(), A->getHeight(), ROW_MAJOR, DEVICE, LOCAL_WORKGROUP_DIM);
	A->setPadding(0.0);
	A->transferAsync(dev_Mat_A, stream1, cudaMemcpyHostToDevice);
	cudaStreamSynchronize(stream1);

	PCG pcg(dev_Vec_X, dev_Vec_Y, dev_Mat_A, 10);
	pcg.solve();

	Vector* X = new Vector(dev_Vec_X->getLength(), HOST, LOCAL_WORKGROUP_DIM);
	dev_Vec_X->transferAsync(X, stream1, cudaMemcpyDeviceToHost, true);
	cudaStreamSynchronize(stream1);
}

//for testing the GPU-PCG algorithm
void Manager::run()
{

	//allocate memory spaces on device
	Matrix* dev_Mat_A = new Matrix(A->getWidth(), A->getHeight(), COLUMN_MAJOR, DEVICE, LOCAL_WORKGROUP_DIM);
	Vector* dev_Vec_X = new Vector(A->getWidth(), DEVICE, LOCAL_WORKGROUP_DIM);
	Vector* dev_Vec_Y = new Vector(Y->getLength(), DEVICE, LOCAL_WORKGROUP_DIM);

	//copy data matrix, output vector to device
	Y->transferAsync(dev_Vec_Y, stream1, cudaMemcpyHostToDevice, true);
	A->transferAsync(dev_Mat_A, stream1, cudaMemcpyHostToDevice);
	cudaStreamSynchronize(stream1);

	//call the PCG algorithm
	PCG testPCG(dev_Vec_X, dev_Vec_Y, dev_Mat_A, 10);


}

Manager::~Manager()
{

}
