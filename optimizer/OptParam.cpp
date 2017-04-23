/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "OptParam.h"
#include <iostream>

using namespace pscrCL;

OptParam::OptParam(
		pscrCL_opt_id _id,
		const std::string& _name,
		const std::string& _clName,
		const AbstractOptValueInterpreter& _interpreter,
		int _defaultValue) :
		name(_name), clName(_clName), interpreter(&_interpreter) {

	id = _id;
	defaultValue = interpreter->backConvert(_defaultValue);
}

pscrCL_opt_id OptParam::getId() const {
	return id;
}

const std::string& OptParam::getName() const {
	return name;
}

const std::string& OptParam::getClName() const {
	return clName;
}

int OptParam::getDefaultInternalValue() const {
	return defaultValue;
}

int OptParam::getDefaultExternalValue() const {
	return interpret(getDefaultInternalValue());
}

int OptParam::interpret(int _intValue) const {
	return interpreter->convert(_intValue);
}

int OptParam::revInterpret(int _extValue) const{
	return interpreter->backConvert(_extValue);
}

bool OptParam::hasClName() const {
	return getClName() != "";
}
