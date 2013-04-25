#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Vector.h"

Vector::Vector(size_t length, DataLocation location) throw(std::logic_error)
{
	Vec.length = length;
	Vec.isPadded = false;
	Vec.paddedLength = 0;
	Vec.data = NULL;
	this->location = location;

	if(location == HOST){
		//always allocate pinned memory on the host
		HANDLE_ERROR(cudaMallocHost( &Vec.data, length*sizeof(float)));
	}
	else if(location == DEVICE)
		HANDLE_ERROR(cudaMalloc(&Vec.data, length*sizeof(float)));
	else
		throw(std::logic_error("Vector::Vector -- Unknown location given!\n"));

	return;
}

Vector::Vector(size_t length, DataLocation location, size_t paddingMultiple) throw(std::logic_error)
{
	Vec.length = length;
	Vec.isPadded = true;
	Vec.paddedLength = extendToMultiple(length, paddingMultiple);
	Vec.data = NULL;
	this->location = location;

	if(location == HOST){
		//always allocate pinned memory on the host
		HANDLE_ERROR(cudaMallocHost( &Vec.data, Vec.paddedLength*sizeof(float)));
	}
	else if(location == DEVICE)
		HANDLE_ERROR(cudaMalloc(&Vec.data, Vec.paddedLength*sizeof(float)));
	else
		throw(std::logic_error("Vector::Vector -- Unknown location given!\n"));

	return;
}

Vector::~Vector()
{
	if((Vec.data != NULL) && (location == DEVICE))
		cudaFree(Vec.data);
	else if((Vec.data != NULL) && (location == HOST))
		cudaFreeHost(Vec.data);
}

void Vector::print()
{
	if(this->location == DEVICE){
		Vector printVec(this->getLength(), HOST, true);
		this->transfer(&printVec, cudaMemcpyDeviceToHost, true);
		for(int i = 0; i < printVec.getLength(); i++)
			std::cout << printVec.getData()[i] << std::endl;
	}
	else
		for(int i = 0; i < Vec.length; i++)
			std::cout << Vec.data[i] << std::endl;
}

void Vector::transferAsync(Vector* destVec, cudaStream_t stream, cudaMemcpyKind kind, bool transferPadding) throw(std::logic_error)
{
	if( destVec->getLength() != this->getLength())
		throw(std::logic_error("Vecter::transferAsync -- vectors must be same length!\n"));

	if(transferPadding){
		if(destVec->isPadded() && this->isPadded()){
			HANDLE_ERROR(cudaMemcpyAsync(destVec->getData(), this->getData(), destVec->getPaddedLength()*sizeof(float), kind, stream));
		}
		else
			throw(std::logic_error("Vector::transferAsync -- padding mismatch failure!\n"));
	}
	else{
		if(!destVec->isPadded() && !this->isPadded()){
			HANDLE_ERROR(cudaMemcpyAsync(destVec->getData(), this->getData(), destVec->getLength()*sizeof(float), kind));
		}
		else
			throw(std::logic_error("Vector::transferAsync -- padding mismatch faiure!\n"));
	}
	return;
}

void Vector::transfer(Vector* destVec, cudaMemcpyKind kind, bool transferPadding) throw(std::logic_error)
{
	if(destVec->getLength() != this->getLength())
		throw(std::logic_error("Vector::transferAsync -- vectors must be same length!\n"));

	if(transferPadding){
		if(destVec->isPadded() && this->isPadded()){
			HANDLE_ERROR(cudaMemcpy(destVec->getData(), this->getData(), destVec->getPaddedLength()*sizeof(float), kind));
		}
		else
			throw(std::logic_error("Vector::transferAsync -- padding mismatch failure!\n"));
	}
	else{
		if(!destVec->isPadded() && !this->isPadded()){
			HANDLE_ERROR(cudaMemcpy(destVec->getData(), this->getData(), destVec->getLength()*sizeof(float), kind));
		}
		else
			throw(std::logic_error("Vector::transferAsync -- padding mismatch faiure!\n"));
	}
	return;
}

DataLocation Vector::getLocation()
{
	return location;
}

size_t Vector::getLength()
{
	return Vec.length;
}

size_t Vector::getPaddedLength()
{
	return Vec.paddedLength;
}

bool Vector::isPadded()
{
	return Vec.paddedLength;
}

VecStruct Vector::getStruct()
{
	return Vec;
}

float& Vector::operator[](size_t index)
{
	if(index >= Vec.length)
		throw(std::logic_error("index out or range"));
	else if (location == DEVICE)
		throw(std::logic_error("Vector is on the Device!"));

	return Vec.data[index];
}

void Vector::setVector(float value)
{
	if(location == DEVICE)
		throw(std::logic_error("setVector only implemented for host vectors!"));

	for(int i = 0; i < Vec.length; i++)
		Vec.data[i] = value;
}

void Vector::setPadding(float value)
{
	if(location == DEVICE)
		throw(std::logic_error("setVector only implemented for host vectors!"));

	for(int i = Vec.length; i < Vec.paddedLength; i++)
		Vec.data[i] = value;
}

size_t Vector::extendToMultiple(int length, size_t multiple)
{
	size_t temp = 0;
	while(length > 0){
		temp ++;
		length -= multiple;
	}
	return temp*multiple;
}

float* Vector::getData()
{
	return Vec.data;
}
