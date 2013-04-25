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
#include <matio.h>
#include "matlabInterface.h"
#include "Vector.h"
#include "common.h"

Vector* MatlabDataInterface::loadVector(char* fileName, char* vecName)
{
	matvar_t* vec;
	Vector* vector;

	vec = loadMatvar(fileName, vecName);
	vector = convertMatvarToVector(vec);

	return vector;
}

Matrix* MatlabDataInterface::loadMatrix(char* fileName, char* matName, MatrixStorageType storeType)
{
	matvar_t* mat;
	Matrix* matrix;

	mat = loadMatvar(fileName, matName);
	matrix = convertMatvarToMatrix(mat, storeType);

	return matrix;
}

Vector* MatlabDataInterface::convertMatvarToVector(matvar_t *vec)
{
	Vector* vector;


	if(vec->rank != 2){
		std::cerr << "Expected " << vec->name << " to be 2-dimensional" << std::endl;
		std::cerr << "It is actually " << vec->rank << " dimensional" << std::endl;
		exit(EXIT_FAILURE);
	}

	//get the (column or row) vector length
	size_t vecLength = 0;
	if(vec->dims[0] == 1)
		vecLength = vec->dims[1];
	else
		vecLength = vec->dims[0];


	//make the memory allocation
	vector = new Vector(vecLength, HOST, LOCAL_WORKGROUP_DIM);

	//check the matvar internal storage type,
	//convert to float and store in our vect
	if(vec->data_type == MAT_T_DOUBLE){
		for(int i = 0; i < vecLength; i++){
			vector->getData()[i] = (float)*(double*)((char*)vec->data + i*vec->data_size);
		}
	}
	else if(vec->data_type == MAT_T_SINGLE){
		for(int i = 0; i < vecLength; i++){
			vector->getData()[i] = *(float*)((char*)vec->data + i*vec->data_size);
		}
	}
	else{
		//TODO: add support for matlab boolean type and integer types
		std::cerr << "Mat file stored in unknown format!" << std::endl;
		exit(EXIT_FAILURE);
	}

	return vector;
}

Matrix* MatlabDataInterface::convertMatvarToMatrix(matvar_t *mat, MatrixStorageType storeType)
{
	Matrix* matrix;
	matio_types type;

	if(mat->rank != 2){
		std::cerr << "Expected " << mat->name << " to be 2-dimensional" << std::endl;
		std::cerr << "It is actually " << mat->rank << " dimensional" << std::endl;
		exit(EXIT_FAILURE);
	}

	//allocate requested matrix in requested storage format
	if(storeType == COLUMN_MAJOR){
		matrix = new Matrix(mat->dims[0], mat->dims[1], storeType, HOST, LOCAL_WORKGROUP_DIM);
	}
	else //store ROW MAJOR
	{
		matrix = new Matrix(mat->dims[1], mat->dims[0], storeType, HOST, LOCAL_WORKGROUP_DIM);
	}

	//check the mat files internal storage primitive
	type = mat->data_type;

	//convert the matlab data to our Matrix data type format
	if(type == MAT_T_DOUBLE){
		if(storeType == COLUMN_MAJOR){
			for(int i = 0; i < matrix->getWidth(); i++){
				for(int j = 0; j < matrix->getHeight(); j++){
					matrix->getData()[i*matrix->getPaddedHeight() + j] = (float)*(double*)((char*)mat->data + i*mat->dims[0]*mat->data_size + j*mat->data_size);
				}
			}
		}
		else if(storeType == ROW_MAJOR){
			for(int i = 0; i < matrix->getPaddedWidth(); i++){
				for(int j = 0; j < matrix->getPaddedHeight(); j++){
					if(i < matrix->getWidth() && j < matrix->getHeight())
						matrix->getData()[j*matrix->getPaddedWidth() + i] = (float)*(double*)((char*)mat->data + i*mat->dims[0]*mat->data_size + j*mat->data_size);
				}
			}
		}
	}
	else if(type == MAT_T_SINGLE){
		if(storeType == COLUMN_MAJOR){
			for(int i = 0; i < matrix->getWidth(); i++){
				for(int j = 0; j < matrix->getHeight(); j++){
					//matrix->data[j*matrix->width + i] = *(float*)((char*)mat->data + i*mat->dims[1]*mat->data_size + j*mat->data_size);
				}
			}
		}
		else if(storeType == ROW_MAJOR){
			for(int i = 0; i < matrix->getWidth(); i++){
				for(int j = 0; j < matrix->getHeight(); j++){
					//matrix->data[j*matrix->width + i] = *(float*)((char*)mat->data + i*mat->dims[1]*mat->data_size + j*mat->data_size);
				}
			}
		}
	}
	else{
		//TODO: add support for matlab boolean type and integer types
		std::cerr << "Mat file stored in unknown format!" << std::endl;
		exit(EXIT_FAILURE);
	}

	return matrix;
}

matvar_t* MatlabDataInterface::loadMatvar(char* fileName, char* matName)
{
	mat_t *matfp = NULL;
	matvar_t *matvar = NULL;

	//Open the file
	matfp = Mat_Open( fileName, MAT_ACC_RDONLY);
	if(!matfp){
		std::cerr << "Unable to open " << fileName << std::endl;
		exit(EXIT_FAILURE);
	}

	//read the variable from the mat file
	matvar = Mat_VarRead(matfp, matName);
	if(!matvar){
		std::cerr << "Unable to Read " << matvar << " from " << fileName << std::endl;
		exit(EXIT_FAILURE);
	}

	//close mat file and return the matvar
	Mat_Close(matfp);
	return matvar;
}
