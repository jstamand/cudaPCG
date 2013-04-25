/*
 * Matrix.h
 *
 *  Created on: Mar 9, 2013
 *      Author: j817s517
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <cuda_runtime.h>
#include "common.h"
#include <stdexcept>
#include "MatStruct.h"

enum MatrixStorageType{COLUMN_MAJOR, ROW_MAJOR};

class Matrix{

public:
	Matrix(size_t width, size_t height, MatrixStorageType storageType, DataLocation location) throw(std::logic_error);
	Matrix(size_t width, size_t height, MatrixStorageType, DataLocation location, size_t paddingMultiple) throw(std::logic_error);
	void transferAsync(Matrix* destMat, cudaStream_t stream, cudaMemcpyKind kind) throw(std::logic_error);
	DataLocation getLocation();
	size_t getWidth();
	size_t getPaddedWidth();
	size_t getHeight();
	size_t getPaddedHeight();
	MatrixStorageType getStoredAs();
	bool isPadded();
	float* getData();
	size_t getPitch();
	void setPadding(float value) throw(std::logic_error);
	void setMatrix(float value) throw(std::logic_error);
	MatStruct getStruct();

private:
	MatStruct Mat;
	size_t extendToMultiple(int length, size_t multiple);
	MatrixStorageType storedAs;
	DataLocation location;
	size_t paddingMultiple;
};

#endif /* MATRIX_H_ */
