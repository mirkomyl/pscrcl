/*
 *  Created on: Dec 2, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_BOUNDARIES
#define PSCRCL_BOUNDARIES

#include "CArray.h"

namespace pscrCL {

// A class for storing information on how the system is divided into partial
// solution during each recursion step.
class Boundaries {
public:
	explicit Boundaries(int n);

	// Returns the system size
	int getN() const;
	int getK() const;
	
	// Returns a array containing information on how the system is divided 
	// into partial solutions during i'th recursion step. 
	const int* getArrayForStep(int i) const;
	
	int getArrayForStepSize(int i) const;
	
	// Returns the whole array.
	const int* getArray() const;
	
	int getArraySize() const;
private:
	int n;
	CArray<int> bounds;
};
	
}

#endif