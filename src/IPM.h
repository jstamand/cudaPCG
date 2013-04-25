/*
 * IMP.h
 *
 *  Created on: Apr 16, 2013
 *      Author: airthimble
 */

#ifndef IMP_H_
#define IMP_H_
#include <stdexcept>
#include "common.h"
#include "Matrix.h"
#include "Vector.h"
#include <string>
#include "PCG.h"


class IPM
{
private:
	float t;				//forces IPM down central path
	float lambda;			//L1 Regularization parameter
	float epsilon;			//duality gap needed for algorithm termination
	size_t PCG_MaxIter;

	Matrix* A;				//data matrix
	Matrix* devA;
	Vector* Y;				//output vector
	Vector* devY;
	Vector* X;				//model coefficients
	Vector* U;				//optimization constraints
	Vector* devU;
	Vector* devXU;			//contains both X and U
	Vector* devG;			//contains the gradient
	Matrix* devH;			//Hessian Matrix
	Matrix* devP;			//preconditioner for PCG

	unsigned int numSamples;
	unsigned int numFeatures;

	//PCG pcg;
	cudaStream_t stream;

public:
	IPM(Matrix* A, Vector* Y);
	void solve();
};

void IPM::solve()
{
	//create a cuda stream on a cuda-capable device
	HANDLE_ERROR(cudaStreamCreate(&stream));

	//create copies of items that are getting copied to the device
	devXU = new Vector(X->getLength() + U->getLength(), DEVICE);
	devG = new Vector(devXU->getLength(), DEVICE);
	//devH = new Matrix

}

IPM::IPM(Matrix* A, Vector* Y): A(A), Y(Y), lambda(0.001), t(1/0.001), epsilon(0.001), devA(NULL), devU(NULL), PCG_MaxIter(5000)
{
	//error checking
	if(A->getLocation() != HOST)
		throw(std::logic_error("Matrix A must be located on the host"));

	if(Y->getLocation() != HOST)
		throw(std::logic_error("Vector Y must be located on the host"));

	//get the matrix dimensions and check for errors
	numSamples = A->getHeight();
	numFeatures = A->getWidth();

	if(Y->getLength() != numSamples)
		throw(std::logic_error("Data Matrix and Output Vector dimension mismatch"));

	//initialization routine
	X = new Vector(numFeatures, HOST);
	U = new Vector(numFeatures, HOST);

	for(int i = 0; i < numFeatures; i++){
		(*X)[i] = 0;
		(*U)[i] = 1;
	}
}



#endif /* IMP_H_ */
