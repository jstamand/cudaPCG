/*
 * Manager.h
 *
 *  Created on: Apr 16, 2013
 *      Author: airthimble
 */

#ifndef MANAGER_H_
#define MANAGER_H_
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"
#include "Matrix.h"
#include "Vector.h"
#include "matlabInterface.h"

class Manager
{
public:
	Manager(char* fileX, char* matX, char* fileY, char* vecY);
	void run();
	~Manager();
	cudaStream_t stream1;
private:
	MatlabDataInterface matInterface;

	Matrix* A;
	Vector* Y;
	Vector* X;

	void testPCG();
};

#endif /* MANAGER_H_ */
