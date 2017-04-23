/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_OPT_VALUES
#define PSCRCL_OPT_VALUES

#include <map>
#include <vector>
#include "../common.h"

namespace pscrCL {

// Container class for the parameter values
class OptValues {
public:
	// Invalid id exception. Thrown when trying to access non existing
	// parameter value
	class InvalidIdException : public UnknownError {
	public:
		InvalidIdException(const char* message) : UnknownError(message) {}
		InvalidIdException(pscrCL_opt_id id) :
			UnknownError(("Invalid parameter id: " + toString(id)).c_str()) {}
	};

	// Member function which is used to add a new parameter value or modify
	// a parameter value
	void setValue(pscrCL_opt_id id, int value);

	// Member function which returns the value of a parameter corresponding
	// to the given id
	int getValue(pscrCL_opt_id id) const;

	// Member function which returns the value of a parameter corresponding
	// to the given id
	int operator()(pscrCL_opt_id id) const ;

	// Member function which returns a reference to the value of a parameters
	// with a given id
	int& operator()(pscrCL_opt_id id);

	// The parameter values can also be accessed in the order they were added.
	// This can be done using the following member functions:
	const int& operator[](int idx) const ;
	int& operator[](int idx);
	int size() const;

	// Returns true if optimization parameter value exists
	bool hasValue(pscrCL_opt_id id) const;

private:
	typedef std::vector<int> ValuesContainer;
	typedef std::map<pscrCL_opt_id, ValuesContainer::size_type>
		LookupContainer;
	typedef std::pair<pscrCL_opt_id, ValuesContainer::size_type> LookupPair;

	ValuesContainer values;

	// A lookup "table" which is used to map parameter id:s to the parameter
	// values
	LookupContainer lookup;
};

}

#endif
