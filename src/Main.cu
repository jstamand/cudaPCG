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
#include "Manager.h"

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	Manager man("../../datasets/wine/X.mat", "X", "../../datasets/wine/Y.mat", "Y");
}
