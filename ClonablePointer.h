/*
 *  Created on: Jan 13, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_CLONABLE_POINTER
#define PSCRCL_CLONABLE_POINTER

#include "Clonable.h"

namespace pscrCL {

// An automatic pointer which is used together with the Clonable class. Calls
// the clone() member function when copying.
template <typename T>
class ClonablePointer {
public:
	
	// Null pointer exception
	class NullPointerException : public Exception {
	public:
		NullPointerException(const char* message) : Exception(message) {}
	};
	
	ClonablePointer() {
		ptr = 0;
	}
	ClonablePointer(const T* _a) {
		if(_a)
			ptr = (T*) _a->clone();
		else
			ptr = 0;
	}
	ClonablePointer(const ClonablePointer<T>& _old) {
		if(_old.ptr)
			ptr = (T*) _old.ptr->clone();
		else
			ptr = 0;
	}
	ClonablePointer<T>& operator=(const ClonablePointer<T>& _a) {
		if(this == &_a)
			return *this;

		if(ptr)
			delete ptr;

		if(_a.ptr)
			ptr = (T*) _a.ptr->clone();
		else
			ptr = 0;
		
		return *this;
	}
	ClonablePointer<T>& operator=(const T* _a) {
		if(ptr)
			delete ptr;

		if(_a)
			ptr = (T*) _a->clone();
		else
			ptr = 0;
		
		return *this;
	}
	~ClonablePointer() {
		if(ptr)
			delete ptr;
	}
	ClonablePointer<T>& operator<<=(T* _a) {
		if(ptr)
			delete ptr;
		
		ptr = _a;
		
		return *this;
	}
	T& operator*() {
		if(ptr == 0)
			throw NullPointerException(
					"ClonablePointer null pointer exception");
		return *ptr;
	}
	const T& operator*() const {
		if(ptr == 0)
			throw NullPointerException(
					"ClonablePointer null pointer exception");
		return *ptr;
	}
	T* operator->() {
		if(ptr == 0)
			throw NullPointerException(
					"ClonablePointer null pointer exception");
		return ptr;
	}
	const T* operator->() const {
		if(ptr == 0)
			throw NullPointerException(
					"ClonablePointer null pointer exception");
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
