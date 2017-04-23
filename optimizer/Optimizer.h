/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_OPT_GROUP
#define PSCRCL_OPT_GROUP

#include <memory>

#include "../common.h"
#include "OptParam.h"
#include "AbstractOptimizerHelper.h"

namespace pscrCL {

// Optimizer class.
class Optimizer {
public:
	// Invalid optimization parameter id exception. Thrown when invalid
	// parameter id is detected.
	class InvalidIdException : public UnknownError {
	public:
		InvalidIdException(const char* message) : UnknownError(message) {}
		InvalidIdException(pscrCL_opt_id id) :
			UnknownError(("Invalid opt. parameter id: " +
					toString(id)).c_str()) {}
	};

	// Duplicated optimization parameter id exception. Thrown when duplicated
	// parameters is detected.
	class DuplicatedIdException : public UnknownError {
	public:
		DuplicatedIdException(const char* message) : UnknownError(message) {}
		DuplicatedIdException(pscrCL_opt_id id) :
			UnknownError(("Duplicated opt. parameter id: " +
					toString(id)).c_str()) {}
	};

	// Member function for adding new parameters
	void addParam(const OptParam& newParam);

	// Member function for adding new parameters
	void addParams(const Optimizer &optimizer);
	
	// Interprets the parameter value corresponding to the given id
	int interped(pscrCL_opt_id id, const OptValues& values) const;
	
	// Reverse interprets the parameter value corresponding to the given id
	int revInterped(pscrCL_opt_id id, int revValue) const;

	// Member function which return the default parameter values
	OptValues getDefaultValues() const;

	// Member function which return optimized parameter values
	OptValues getOptimizedValues(AbstractOptimizerHelper& helper) const;

	// Member function which returns the parameter values in a format suitable
	// for the OpenCL compiler.
	std::string getCompilerArgs(const OptValues& values) const;

	// Member function which returns the parameter values in a human readable
	// format
	std::string getString(const OptValues& values) const;
	
	Optimizer operator+(const Optimizer& a) const;

private:
	// Member function which calculates the value of the object function
	float objectFunction(
			AbstractOptimizerHelper& helper,
			const OptValues& values) const;

	typedef std::map<pscrCL_opt_id, OptParam> ParamContainer;

	ParamContainer params;
};

}

#endif
