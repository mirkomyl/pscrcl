/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_CLONABLE
#define PSCRCL_CLONABLE

namespace pscrCL {

// An abstract class defining an object which capable of forming a copy of
// itself. Is used together with the ClonablePointer class (see below).
class Clonable {
public:
	virtual ~Clonable() {};

	// This member function should return a pointer to a copy of the object.
	// The caller is responsible for deallocating the allocated memory.
	virtual Clonable* clone() const = 0;
};

}
#endif
