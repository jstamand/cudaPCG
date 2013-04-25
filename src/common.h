/*
 * common.h
 *
 *  Created on: Mar 9, 2013
 *      Author: j817s517
 */

#ifndef COMMON_H_
#define COMMON_H_
#include <iostream>
#include <stdio.h>
//common things shared in the project
enum DataLocation{HOST, DEVICE};

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define NUM_CUDA_STREAMS					14
#define LOCAL_WORKGROUP_DIM					16

#endif /* COMMON_H_ */
