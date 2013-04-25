/*
 * PCG.h
 *
 *  Created on: Apr 15, 2013
 *      Author: airthimble
 */

#ifndef PCG_H_
#define PCG_H_

#include "Matrix.h"
#include "Vector.h"
#include "cublas_v2.h"
#include <stdexcept>

class PCG{

private:
	Matrix* dev_Mat_A;		//Data Matrix
	Vector* dev_Vec_B;		//Output vector
	Vector* dev_Vec_X;			//Model Coefficients
	Matrix* dev_Mat_P;			//Pre-conditioner Matrix
	Vector* dev_Vec_R;			//Residual
	Vector* dev_Vec_D;			//Search direction
	Vector* dev_Vec_Q;			//Something???

	float alpha;
	float Beta;
	float epsilon;
	float delta_new;
	float delta_old;
	float delta_base;

	float *dev_alpha;
	float *dev_Beta;
	float *dev_delta_new;

	unsigned int iterations;
	unsigned int maxIterations;

	void init() throw(std::runtime_error);

	dim3 vecBlocks;
	dim3 vecThreads;

	void Update_Q();
	void Update_Alpha();
	void Update_X();
	void Update_Residual();
	void Update_Deltas();
	void Update_direction();

	void checkProgress();

public:
	PCG(Vector* dev_X, Vector* dev_b, Matrix* dev_A, int maxIterations) throw(std::logic_error);
	void solve();
};

#endif /* PCG_H_ */
