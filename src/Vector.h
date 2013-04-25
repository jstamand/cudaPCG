/*
 * Vector.h
 *
 *  Created on: Mar 12, 2013
 *      Author: j817s517
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include <cuda_runtime.h>
#include <stdexcept>
#include "common.h"
#include "VecStruct.h"

class Vector{

public:
	//constructors
	Vector(size_t length, DataLocation location) throw(std::logic_error);
	Vector(size_t length, DataLocation location, size_t paddingMultiple) throw(std::logic_error);
	~Vector();
	void transfer(Vector* destVec, cudaMemcpyKind kind, bool transferPadding) throw(std::logic_error);
	void transferAsync(Vector* destVec, cudaStream_t stream, cudaMemcpyKind kind, bool transferPadding) throw(std::logic_error);

	//setters / getters
	DataLocation getLocation();
	size_t getLength();
	size_t getPaddedLength();
	bool isPadded();
	VecStruct getStruct();

	float &operator[](size_t index);
	
	void setPadding(float value);
	void setVector(float value);
	float* getData();
	void print();

private:
	VecStruct Vec;
	DataLocation location;
	size_t extendToMultiple(int length, size_t multiple);
};

#endif /* VECTOR_H_ */
