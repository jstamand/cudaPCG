#ifndef VECSTRUCT_H_
#define VECSTRUCT_H_

typedef struct {
	size_t length;
	bool isPadded;
	size_t paddedLength;
	float* data;
}VecStruct;

#endif
