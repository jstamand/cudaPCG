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
#include <iostream>
#include "common.h"
#include "Matrix.h"

Matrix::Matrix(size_t width, size_t height, MatrixStorageType storageType, DataLocation location)
			throw(std::logic_error)
			: storedAs(storageType), location(location), paddingMultiple(0)
{
	Mat.width = width;
	Mat.height = height;
	Mat.data = NULL;
	Mat.isPadded = false;
	Mat.paddedHeight = 0;
	Mat.paddedWidth = 0;

	//calculate pitch based on matrix storage format
	//make the allocation on the host or device
	//if on device -- let device choose pitch
	if(location == HOST && storedAs == ROW_MAJOR){
		Mat.pitch = (Mat.width)*sizeof(float);
		HANDLE_ERROR(cudaMallocHost( &(Mat.data), (Mat.width)*(Mat.height)*sizeof(float)));
	}
	else if(location == DEVICE && storedAs == ROW_MAJOR)
		HANDLE_ERROR(cudaMallocPitch(&Mat.data, &Mat.pitch, Mat.width*sizeof(float), Mat.height));
	//else if(location == HOST && storedAs == COLUMN_MAJOR){
	//	HANDLE_ERROR(cudaMallocPitch(&Mat->data, &pitch, height*sizeof(float), width));
	//}
	else if( storedAs == COLUMN_MAJOR)
		throw(std::logic_error("Matrix::Matrix -- Only ROW_MAJOR supported at this time!\n"));
	else
		throw(std::logic_error("Matrix::Matrix -- unknown location given!\n"));

	return;
}

Matrix::Matrix(size_t width, size_t height, MatrixStorageType storageType, DataLocation location, size_t paddingMultiple)
			throw(std::logic_error)
{
	this->storedAs = storageType;
	this->location = location;
	this->paddingMultiple = paddingMultiple;

	Mat.width = width;
	Mat.height = height;
	Mat.data = NULL;
	Mat.isPadded = true;

	Mat.paddedHeight = extendToMultiple(height, paddingMultiple);
	Mat.paddedWidth = extendToMultiple(width, paddingMultiple);

	//calculate pitch based on matrix storage format
	//make the allocation on the host or device
	//if on device -- let device choose pitch
	if(location == HOST && storedAs == ROW_MAJOR){
		Mat.pitch = Mat.paddedWidth*sizeof(float);
		HANDLE_ERROR(cudaMallocHost( &Mat.data, Mat.paddedWidth*Mat.paddedHeight*sizeof(float)));
	}
	else if(location == DEVICE && storedAs == ROW_MAJOR)
		HANDLE_ERROR(cudaMallocPitch(&Mat.data, &Mat.pitch, Mat.paddedWidth*sizeof(float), Mat.paddedHeight));
	//else if(location == HOST && storedAs == COLUMN_MAJOR){
	//	HANDLE_ERROR(cudaMallocPitch(&Mat->data, &pitch, height*sizeof(float), width));
	//}
	else if( storedAs == COLUMN_MAJOR)
		throw(std::logic_error("Matrix::Matrix -- Only ROW_MAJOR supported at this time!\n"));
	else
		throw(std::logic_error("Matrix::Matrix -- unknown location given!\n"));

	return;
}


void Matrix::transferAsync(Matrix* destMat, cudaStream_t stream, cudaMemcpyKind kind) throw(std::logic_error)
{
	//start async copy of data to device
	if(storedAs == ROW_MAJOR && destMat->getStoredAs() == ROW_MAJOR){
		if(this->isPadded() && destMat->isPadded())
			HANDLE_ERROR( cudaMemcpy2DAsync(destMat->getData(), destMat->getPitch(), Mat.data, Mat.pitch, Mat.paddedWidth*sizeof(float), Mat.paddedHeight, kind, stream));
		else if((this->isPadded() == false) && (destMat->isPadded() == false))
			HANDLE_ERROR( cudaMemcpy2DAsync(destMat->getData(), destMat->getPitch(), Mat.data, Mat.pitch, Mat.width*sizeof(float), Mat.height, kind, stream));
	}
	else
		throw(std::logic_error("Only ROW_MAJOR stored matrices supported at this time!"));
}

DataLocation Matrix::getLocation()
{
	return location;
}

size_t Matrix::getWidth()
{
	return Mat.width;
}

size_t Matrix::getPaddedWidth()
{
	return Mat.paddedWidth;
}

size_t Matrix::getHeight()
{
	return Mat.height;
}

size_t Matrix::getPaddedHeight()
{
	return Mat.paddedHeight;
}

MatrixStorageType Matrix::getStoredAs()
{
	return storedAs;
}

float* Matrix::getData()
{
	return Mat.data;
}

size_t Matrix::getPitch()
{
	return Mat.pitch;
}

bool Matrix::isPadded()
{
	return Mat.isPadded;
}

void Matrix::setPadding(float value) throw(std::logic_error)
{
	if(location != HOST)
		throw(std::logic_error("Only able to zero pad data on the HOST"));

	if(Mat.isPadded){
		if(storedAs == ROW_MAJOR)
			for(int i = Mat.width; i < Mat.paddedWidth; i++)
				for(int j = Mat.height; j < Mat.paddedHeight; j++)
					Mat.data[i + Mat.paddedWidth*j] = value;

		else//COLUMN_MAJOR
			for(int i = Mat.height; i < Mat.paddedHeight; i++)
				for(int j = Mat.width; j < Mat.paddedWidth; j++)
					Mat.data[i + Mat.paddedWidth*j] = value;
	}
	else
		throw(std::logic_error("Cannot setPadding on a non-padded matrix!"));
}


void Matrix::setMatrix(float value) throw(std::logic_error)
{

	if((this->location == DEVICE) && (storedAs == ROW_MAJOR)){
		for(int i = 0; i < Mat.width; i++)
			for(int j = 0; j < Mat.height; j++)
				Mat.data[i + j*Mat.paddedHeight] = value;
	}
	else
		throw(std::logic_error("setPadding Error!\n"));

}

size_t Matrix::extendToMultiple(int length, size_t multiple)
{
	size_t temp = 0;
	while(length > 0){
		temp ++;
		length -= multiple;
	}
	return temp*multiple;
}

MatStruct Matrix::getStruct()
{
	return Mat;
}
