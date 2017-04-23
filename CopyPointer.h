/*
 *  Created on: May 22, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_AUTO_POINTER
#define PSCRCL_AUTO_POINTER

#include "common.h"

namespace pscrCL {

// An automatic pointer.
template <typename T>
class CopyPointer {
public:
	
	// Null pointer exception
	class NullPointerException : public Exception {
	public:
		NullPointerException(const char* message) : Exception(message) {}
	};
	
	CopyPointer() {
		ptr = 0;
	}
	CopyPointer(const T* _a) {
		if(_a)
			ptr = new T(*_a);
		else
			ptr = 0;
	}
	CopyPointer(const CopyPointer<T>& _old) {
		if(_old.ptr)
			ptr = new T(_old);
		else
			ptr = 0;
	}
	CopyPointer<T>& operator=(const CopyPointer<T>& _a) {
		if(this == &_a)
			return *this;

		if(ptr)
			delete ptr;

		if(_a.ptr)
			ptr = new T(*_a.ptr);
		else
			ptr = 0;
		
		return *this;
	}
	CopyPointer<T>& operator=(const T* _a) {
		if(ptr)
			delete ptr;

		if(_a)
			ptr = new T(*_a);
		else
			ptr = 0;
		
		return *this;
	}
	~CopyPointer() {
		if(ptr)
			delete ptr;
	}
	CopyPointer<T>& operator<<=(T* _a) {
		if(ptr)
			delete ptr;
		
		ptr = _a;
		
		return *this;
	}
	T& operator*() {
		if(ptr == 0)
			throw NullPointerException(
					"CopyPointer null pointer exception");
		return *ptr;
	}
	const T& operator*() const {
		if(ptr == 0)
			throw NullPointerException(
					"CopyPointer null pointer exception");
		return *ptr;
	}
	T* operator->() {
		if(ptr == 0)
			throw NullPointerException(
					"CopyPointer null pointer exception");
		return ptr;
	}
	const T* operator->() const {
		if(ptr == 0)
			throw NullPointerException(
					"CopyPointer null pointer exception");
		return ptr;
	}
	T* getPtr() {
		return ptr;
	}
	const T* getPtr() const {
		return ptr;
	}
	bool isNull() const {
		return ptr == 0;
	}
private:
	T *ptr;
};

}
#endif
