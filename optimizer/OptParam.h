/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_OPT_PARAM
#define PSCRCL_OPT_PARAM

#include <string>
#include "OptValueInterpreter.h"
#include "../common.h"
#include "../ClonablePointer.h"

namespace pscrCL {

/*
 * This class defines an optimization parameter. Each parameter has an unique
 * id, name, an optional internal name used in the openCL source files, a value
 * interpreter and the default value.
 */
class OptParam {
public:
	// Creates a new parameter
	OptParam(pscrCL_opt_id                     id,
			const std::string                 &name,
			const std::string&                 clName,
			const AbstractOptValueInterpreter &interpreter,
			int                                defaultValue);

	// Returns the id
	pscrCL_opt_id getId() const;

	// Returns the name
	const std::string& getName() const;

	// Returns the internal name used in the OpenCL source files
	const std::string& getClName() const;

	// Returns the default internal value
	int getDefaultInternalValue() const;

	// Which returns the default external value
	int getDefaultExternalValue() const;

	// Member function which return the external value of the parameter
	int interpret(int intValue) const;
	
	// Member function which return the external value of the parameter
	int revInterpret(int extValue) const;

	// Return true if the internal name used in the cl source files is not
	// empty
	bool hasClName() const;

private:
	pscrCL_opt_id id;
	std::string name;
	std::string clName;
	ClonablePointer<AbstractOptValueInterpreter> interpreter;
	int defaultValue;
};

template <>
class ToStringHelper<OptParam> {
public:
	static std::string to(const OptParam& param) {
		return  "id = " + toString(param.getId()) +
				", name = " + param.getName() +
				", clName = " + param.getClName() +
				", default value = " +
				toString(param.getDefaultExternalValue());
	}
};

}

#endif
