/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "../common.h"
#include "OptValues.h"

using namespace pscrCL;

void OptValues::setValue(pscrCL_opt_id _id, int _value) {
	LookupContainer::const_iterator it = lookup.find(_id);
	if(it == lookup.end()) {
		// Add a new parameter
		values.push_back(_value);
		lookup.insert(LookupPair(_id, values.size()-1));
	} else {
		// Modify an existing value
		(*this)[it->second] = _value;
	}
}

int OptValues::getValue(pscrCL_opt_id _id) const {
	LookupContainer::const_iterator it = lookup.find(_id);

	if(it != lookup.end())
		return (*this)[it->second];

	throw InvalidIdException(_id);
}

int OptValues::operator()(pscrCL_opt_id _id) const {
	return getValue(_id);
}

int& OptValues::operator()(pscrCL_opt_id _id) {
	LookupContainer::const_iterator it = lookup.find(_id);

	if(it != lookup.end())
		return (*this)[it->second];

	throw InvalidIdException(_id);
}

const int& OptValues::operator[](int _idx) const {
	return values.at(_idx);
}

int& OptValues::operator[](int _idx) {
	return values.at(_idx);
}

int OptValues::size() const {
	return values.size();
}

bool OptValues::hasValue(pscrCL_opt_id _id) const {
	if(lookup.find(_id) != lookup.end())
		return true;

	return false;
}