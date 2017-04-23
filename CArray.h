#ifndef PSCRCL_CARRAY
#define PSCRCL_CARRAY

#include "common.h"

namespace pscrCL {

template <typename T>
class CArray {
public:
	CArray() {
		data = 0;
		size = 0;
	}
	explicit CArray(int _size) {
		if(_size <= 0) 
			throw UnknownError("CArray::Invalid array size exception.");
		data = new T[_size];
		size = _size;
	}
	CArray(const CArray &_a) {
		data = 0;
		size = 0;
		if(0 < _a.getSize()) {
			data = new T[_a.getSize()];
			size = _a.getSize();
			for(int i = 0; i < size; i++)
				(*this)[i] = _a[i];
		}
	}
	CArray& operator=(const CArray &_a) {
		if(this == &_a)
			return *this;
		
		if(0 < size)
			delete [] data;
		
		data = 0;
		size = 0;
		if(0 < _a.getSize()) {
			data = new T[_a.getSize()];
			size = _a.getSize();
			for(int i = 0; i < size; i++)
				(*this)[i] = _a[i];
		}
		
		return *this;
	}
	~CArray() {
		if(0 < size)
			delete [] data;
	}
	int getSize() const {
		return size;
	}
	int getSizeInBytes() const {
		return getSize() * sizeof(T);
	}
	const T* getPointer() const {
		return data;
	}
	T* getPointer() {
		return data;
	}
	const T& operator[](int _i) const {
#if FULL_DEBUG
		if(_i < 0 || size <= _i)
			throw UnknownError("CArray::Out of bounds exception.");
#endif
		return data[_i];
	}
	T& operator[](int _i) {
#if FULL_DEBUG
		if(_i < 0 || size <= _i)
			throw UnknownError("CArray::Out of bounds exception.");
#endif
		return data[_i];
	}
private:
	T *data;
	int size;
};

}

#endif