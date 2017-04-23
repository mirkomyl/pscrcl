/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_ABSTRACT_OPTIMIZER_HELPER
#define PSCRCL_ABSTRACT_OPTIMIZER_HELPER

#include "OptValues.h"

namespace pscrCL {

// An abstract helper class which is used when evaluating the value of the
// object function.
class AbstractOptimizerHelper {
public:
	// Not initialized exception. Thrown when attempting to use the helper
	// without initialization or preparation.
	class NotReadyException : public UnknownError {
	public:
		NotReadyException(const char* message) : UnknownError(message) {}
	};

	virtual ~AbstractOptimizerHelper() {};

	// Tells the helper to allocate all necessary resources. Should be called
	// before the first prepare/evaluate.
	virtual void initialize() = 0;
	
	// Tells the helper to recover from invalid Commanqueue situation.
	virtual void recover() = 0;

	// Tells helper to prepare for run with the given OptValues.
	virtual bool prepare(const OptValues& values) = 0;

	// Runs the the solver using the OptValues given earlier through prepare()
	// member function.
	virtual void run() = 0;

	// Tells the helper to finalize the OpenCL command queue. Blocking.
	virtual void finalize() = 0;

	// Tells helper to release the unnecessary resources. Should be called only
	// when all desired evaluations have been carried out.
	virtual void release() = 0;
};

}
#endif
