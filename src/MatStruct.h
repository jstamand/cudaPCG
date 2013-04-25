#ifndef MATSTRUCT_H_
#define MATSTRUCT_H_

typedef struct{
	size_t width;
	size_t height;
	bool isPadded;
	size_t paddedWidth;
	size_t paddedHeight;
	float* data;
	size_t pitch;
}MatStruct;

#endif
