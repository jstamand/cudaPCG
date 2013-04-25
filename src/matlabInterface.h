/*
 * matlabInterface.h
 *
 *  Created on: Mar 9, 2013
 *      Author: j817s517
 */

#ifndef MATLABINTERFACE_H_
#define MATLABINTERFACE_H_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <matio.h>
#include <math.h>

#include "Matrix.h"
#include "Vector.h"

class MatlabDataInterface
{
public:
	//loads a matrix from a matlab file
	Matrix* loadMatrix(char* fileName, char* matName, MatrixStorageType storeType);
	Vector* loadVector(char* fileName, char*vecName);

private:
	Vector* convertMatvarToVector(matvar_t *vec);
	Matrix* convertMatvarToMatrix(matvar_t *mat, MatrixStorageType storeType);
	matvar_t* loadMatvar(char* fileName, char* matName);
};




#endif /* MATLABINTERFACE_H_ */
