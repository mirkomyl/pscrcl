/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_OPT_VALUE_INTERPRETER
#define PSCRCL_OPT_VALUE_INTERPRETER

#include <cmath>
#include "../Clonable.h"

namespace pscrCL {

// Abstract interpreter. Used to convert internal parameter values into
// external values and vice versa
class AbstractOptValueInterpreter : public Clonable {
public:
	// Converts an internal value into a external value
	virtual int convert(int intValue) const = 0;

	// Converts a external value into an internal value
	virtual int backConvert(int extValue) const = 0;
};

// Interpreter, which does nothing, i.e, x -> x
class IdenticalOptValueInterpreter : public AbstractOptValueInterpreter {
public:
	virtual int convert(int intValue) const {return intValue; }
	virtual int backConvert(int extValue) const {return extValue; }
	virtual Clonable* clone() const {
		return new IdenticalOptValueInterpreter(*this);
	}
};

// x -> |x|
class AbsOptValueInterpreter : public AbstractOptValueInterpreter {
public:
	virtual int convert(int intValue) const {return std::abs(intValue); }
	virtual int backConvert(int extValue) const {return std::abs(extValue); }
	virtual Clonable* clone() const {
		return new AbsOptValueInterpreter(*this);
	}
};

// i -> 2^i
class Pow2OptValueInterpreter : public AbstractOptValueInterpreter {
public:
	virtual int convert(int intValue) const {return 1<<(intValue); }
	virtual int backConvert(int extValue) const {
		return log(extValue)/log(2.0);
	}
	virtual Clonable* clone() const {
		return new Pow2OptValueInterpreter(*this);
	}
};

// i -> 4^i
class Pow4OptValueInterpreter : public AbstractOptValueInterpreter {
public:
	virtual int convert(int intValue) const { return 1<<(2*(intValue)); }
	virtual int backConvert(int extValue) const {
		return log(extValue)/log(4.0);
	}
	virtual Clonable* clone() const {
		return new Pow4OptValueInterpreter(*this);
	}
};

// i -> 1 if i is odd or 0 if i is even
class BoolOptValueInterpreter : public AbstractOptValueInterpreter {
public:
	virtual int convert(int intValue) const { return intValue & 0x1 ? 1 : 0; }
	virtual int backConvert(int extValue) const {
		return extValue & 0x1 ? 1 : 0;
	}
	virtual Clonable* clone() const {
		return new BoolOptValueInterpreter(*this);
	}
};

// i -> Operator(i)
template <typename Operator, typename Inverse>
class OperatorValueInterpreter : public AbstractOptValueInterpreter {
public:
	explicit OperatorValueInterpreter(int _coef) { coef = _coef; }
	virtual int convert(int intValue) const {
		return Operator()(intValue, coef);
	}
	virtual int backConvert(int extValue) const {
		return Inverse()(extValue, coef);
	}
	virtual Clonable* clone() const {
		return new OperatorValueInterpreter<Operator, Inverse>(*this);
	}
private:
	int coef;
};

typedef OperatorValueInterpreter<std::multiplies<int>, std::divides<int> >
	AddOptValueInterpreter;

}

#endif
