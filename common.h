/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_COMMON
#define PSCRCL_COMMON

#include <cstddef>
#include <typeinfo>
#include <sstream>
#include <complex>
#include <cstddef>
#include <sys/time.h>
#include <math.h>
#include <CL/cl.hpp>

// Global macros

#define LOG4(a) (log(a)/log(4))
#define POW4(a) (1<<(2*(a)))
#define LOG2(a) (log(a)/log(2))
#define POW2(a) (1<<(a))

#define K_2(n) ((int)(LOG2(n)+1))
#define K_4(n) ((int)(LOG4(n)+1))

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define PI 3.141592653589793238462

#define MAX_N1 		POW4(10)
#define MAX_N2 		POW4(10)
#define MAX_N3 		POW4(10)

#define DIVCEIL(a, b) ((a) % (b) != 0 ? (a) / (b) + 1 : (a) / (b))

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define X1_BUFFER_SIZE		4*LDF1
#define A1_DIAG_STRIDE		0*LDF1
#define A1_CODIAG_STRIDE	1*LDF1
#define M1_DIAG_STRIDE		2*LDF1
#define M1_CODIAG_STRIDE	3*LDF1

#define X2_BUFFER_SIZE		4*LDF2
#define A2_DIAG_STRIDE		0*LDF2
#define A2_CODIAG_STRIDE	1*LDF2
#define M2_DIAG_STRIDE		2*LDF2
#define M2_CODIAG_STRIDE	3*LDF2

// Mode flags
#define PSCRCL_PREC_FLOAT			0x000	// Single precision arithmetics
#define PSCRCL_PREC_DOUBLE			0x001	// Double precision arithmetics
#define PSCRCL_NUM_REAL				0x000	// Real numbers
#define PSCRCL_NUM_COMPLEX			0x002	// Complex numbers

#define PSCRCL_M1_TRIDIAG			0x010	// M1 is tridiagonal
#define PSCRCL_M2_TRIDIAG			0x020	// M2 is tridiagonal
#define PSCRCL_M3_TRIDIAG			0x040	// M3 is tridiagonal

#define PSCRCL_MAX_N1				99999
#define PSCRCL_MAX_N2				99999
#define PSCRCL_MAX_N3				99999

/* Parameter id:s */
#define PSCRCL_OPT_PARAM_L3_PREFIX							300
#define PSCRCL_OPT_PARAM_L3_D								301
#define PSCRCL_OPT_PARAM_L3_HILODOUBLE						302
#define PSCRCL_OPT_PARAM_L3_SHARED_ISOLATED_ACCESS			303
#define PSCRCL_OPT_PARAM_L3_NEWTON_DIV						304
#define PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_SIZE			305
#define PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_PER_SYS		306
#define PSCRCL_OPT_PARAM_L3_A_WG_SIZE						307
#define PSCRCL_OPT_PARAM_L3_BCD_WG_SIZE						308
#define PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE					309
#define PSCRCL_OPT_PARAM_L3_PCR_LIMIT						310
#define PSCRCL_OPT_PARAM_L3_PCR_STEPS						311
#define PSCRCL_OPT_PARAM_L3_PARALLEL_STAGE_A1				312
#define PSCRCL_OPT_PARAM_L3_PARALLEL_STAGE_A2				313

#define PSCRCL_OPT_PARAM_L2_PREFIX							200
#define PSCRCL_OPT_PARAM_L2_D								201
#define PSCRCL_OPT_PARAM_L2_SHARED_ISOLATED_ACCESS			202
#define PSCRCL_OPT_PARAM_L2_NEWTON_DIV						203
#define PSCRCL_OPT_PARAM_L2_STAGE_11_WG_SIZE				204
#define PSCRCL_OPT_PARAM_L2_STAGE_11_WG_PER_VECTOR			205
#define PSCRCL_OPT_PARAM_L2_STAGE_12A_WG_SIZE				206
#define PSCRCL_OPT_PARAM_L2_STAGE_12A_WG_PER_VECTOR			207
#define PSCRCL_OPT_PARAM_L2_STAGE_12C_WG_SIZE				208
#define PSCRCL_OPT_PARAM_L2_STAGE_12C_WG_PER_VECTOR			209
#define PSCRCL_OPT_PARAM_L2_STAGE_Y2A_MAX_SUM_SIZE_EXP		210
#define PSCRCL_OPT_PARAM_L2_STAGE_Y2B_MAX_SUM_SIZE_EXP		211
#define PSCRCL_OPT_PARAM_L2_STAGE_Y2B_WG_SIZE				212
#define PSCRCL_OPT_PARAM_L2_STAGE_Y2B_WG_PER_VECTOR			213
#define PSCRCL_OPT_PARAM_L2_STAGE_21_WG_SIZE				214
#define PSCRCL_OPT_PARAM_L2_STAGE_21_WG_PER_VECTOR			215
#define PSCRCL_OPT_PARAM_L2_STAGE_22A_WG_SIZE				216
#define PSCRCL_OPT_PARAM_L2_STAGE_22A_WG_PER_VECTOR			217
#define PSCRCL_OPT_PARAM_L2_STAGE_22C_WG_SIZE				218
#define PSCRCL_OPT_PARAM_L2_STAGE_22C_WG_PER_VECTOR			219
#define PSCRCL_OPT_PARAM_L2_VECTOR_LOAD_HELPER				220
#define PSCRCL_OPT_PARAM_L2_MATRIX_LOAD_HELPER				221

#define PSCRCL_OPT_PARAM_L1_PREFIX							100
#define PSCRCL_OPT_PARAM_L1_D								101
#define PSCRCL_OPT_PARAM_L1_SHARED_ISOLATED_ACCESS			102
#define PSCRCL_OPT_PARAM_L1_NEWTON_DIV						103
#define PSCRCL_OPT_PARAM_L1_STAGE_11_WG_SIZE				104
#define PSCRCL_OPT_PARAM_L1_STAGE_11_WG_PER_VECTOR			105
#define PSCRCL_OPT_PARAM_L1_STAGE_12A_WG_SIZE				106
#define PSCRCL_OPT_PARAM_L1_STAGE_12A_WG_PER_VECTOR			107
#define PSCRCL_OPT_PARAM_L1_STAGE_12C_WG_SIZE				108
#define PSCRCL_OPT_PARAM_L1_STAGE_12C_WG_PER_VECTOR			109
#define PSCRCL_OPT_PARAM_L1_STAGE_Y2A_MAX_SUM_SIZE_EXP		110
#define PSCRCL_OPT_PARAM_L1_STAGE_Y2B_MAX_SUM_SIZE_EXP		111
#define PSCRCL_OPT_PARAM_L1_STAGE_Y2B_WG_SIZE				112
#define PSCRCL_OPT_PARAM_L1_STAGE_Y2B_WG_PER_VECTOR			113
#define PSCRCL_OPT_PARAM_L1_STAGE_21_WG_SIZE				114
#define PSCRCL_OPT_PARAM_L1_STAGE_21_WG_PER_VECTOR			115
#define PSCRCL_OPT_PARAM_L1_STAGE_22A_WG_SIZE				116
#define PSCRCL_OPT_PARAM_L1_STAGE_22A_WG_PER_VECTOR			117
#define PSCRCL_OPT_PARAM_L1_STAGE_22C_WG_SIZE				118
#define PSCRCL_OPT_PARAM_L1_STAGE_22C_WG_PER_VECTOR			119
#define PSCRCL_OPT_PARAM_L1_VECTOR_LOAD_HELPER				120
#define PSCRCL_OPT_PARAM_L1_MATRIX_LOAD_HELPER				121

namespace pscrCL {

typedef unsigned int mode_flag;
typedef int pscrCL_opt_id;

// Error, warning and debugging messages

const std::string errorMsgBegin = "(error) pscrCL";
const std::string debugMsgBegin = "(debug) pscrCL";
const std::string warningMsgBegin = "(warning) pscrCL";

/*
 * Solver mode selector
 */
class PscrCLMode {
public:
	PscrCLMode(mode_flag _mode) {
		mode = _mode;
	}
	
	// Returns true if solver is in single precision mode
	bool precFloat() const {
		return mode & PSCRCL_PREC_FLOAT;
	}
	
	// Return true if solver is in double precision mode
	bool precDouble() const {
		return mode & PSCRCL_PREC_DOUBLE;
	}
	
	// Return true if the solver is in real numbers mode
	bool numReal() const {
		return mode & PSCRCL_NUM_REAL;
	}
	
	// Return true if the solver is in complex number mode
	bool numComplex() const {
		return mode & PSCRCL_NUM_COMPLEX;
	}
	
	// Returns true if the matrix M1 is tridiagonal
	bool m1Tridiag() const {
		return mode & PSCRCL_M1_TRIDIAG;
	}
	
	// Return true if the matrix M2 is tridiagonal
	bool m2Tridiag() const {
		return mode & PSCRCL_M2_TRIDIAG;
	}
	
	// Return true if the matrix M3 is tridiagonal
	bool m3Tridiag() const {
		return mode & PSCRCL_M3_TRIDIAG;
	}
	std::size_t getVarSize() const {
	if(numComplex())
		return precDouble() ? sizeof(std::complex<double>) :
			sizeof(std::complex<float>);
	else
		return precDouble() ? sizeof(double) : sizeof(float);
	}

	int getDSize() const {
		if(numComplex())
			return precDouble() ? 1 : 2;
		else
			return precDouble() ? 2 : 4;
	}
private:
	mode_flag mode;
};
	
/*
 * General exception.
 */
class Exception : public std::exception {
public:
	Exception(const char* message = "General pscrCL exception.") : message(message) {}
    Exception(const std::string &message) : message(message) {}
	virtual ~Exception() throw() {}
	virtual const char* what() const throw() { return message.c_str(); }
private:
	std::string message;
};

/*
 * Unknown error. Usually means that there is a bug.
 */
class UnknownError : public Exception {
public:
	UnknownError(const char* message = "Unknown pscrCL exception.") : Exception(message) {}
	UnknownError(const std::string &message) : Exception(message) {}
};

/*
 * Invalid arguments. 
 */
class InvalidArgs : public Exception {
public:
	InvalidArgs(const char* message = "Invalid arguments pscrCL exception.") : Exception(message) {}
	InvalidArgs(const std::string &message) : Exception(message) {}
};

/*
 * Invalid parameters.
 */
class InvalidOptParams : public Exception {
public:
	InvalidOptParams(const char* message = "Invalid parameters pscrCL exception.") : Exception(message) {}
	InvalidOptParams(const std::string &message) : Exception(message) {}
};

/*
 * External library error.
 */
class ExtLibError : public Exception {
public:
	ExtLibError(const char* message = "External library pscrCL exception.") : Exception(message) {}
	ExtLibError(const std::string &message) : Exception(message) {}
};

/*
 * OpenCL error.
 */
class OpenCLError : public ExtLibError {
public:
	OpenCLError(const char* message = "OpenCL library pscrCL exception.") : ExtLibError(message) {
		containsErrorCode = false;
	}
	OpenCLError(const std::string &message) : ExtLibError(message) {}
	OpenCLError(cl_int _err) : ExtLibError("OpenCL library pscrCL exception.") {
		containsErrorCode = true;
		err = _err;
	}
	bool hasErrorCode() const {
		return containsErrorCode;
	}
	cl_int getErrorCode() const {
		return err;
	}
private:
	bool containsErrorCode;
	cl_int err;
};

inline int NEXTPOW2(int v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

// Real-time timer

class Timer {
public:
	Timer() {
		running = false;
		ready = false;
	}
	void begin() {
		gettimeofday(&begin_time, NULL);
		running = true;
		ready = false;
	}
	void end() {
		if(!running)
			throw UnknownError("The timer is not running.");
		gettimeofday(&end_time, NULL);
		running = false;
		ready = true;
	}
	double getTime() const {
		return getEndTime() - getBeginTime();
	}
	double getBeginTime() const {
		if(!running && !ready)
			throw UnknownError("The timer is not running or ready.");
		return begin_time.tv_sec + begin_time.tv_usec*1.0E-6;
	}
	double getEndTime() const {
		if(!ready)
			throw UnknownError("The timer is not ready.");
		return end_time.tv_sec + end_time.tv_usec*1.0E-6;
	}
private:
	bool running ;
	bool ready;
	struct timeval begin_time;
	struct timeval end_time;
};

// Random number generator helpers

template <typename T>
class RandI {
public:
	static T i(T low, T high) {
		return rand() % (high-low+1) + low;
	}
};

typedef RandI<int> RandInt;

template <typename T>
class RandF {
public:
	static T r(T o, T r) {
		return i(o-r, o+r);
	}
	static T i(T low, T high) {
		return (high-low)*((T)rand()/(T)RAND_MAX) + low;
	}
};

template <typename PREC>
class RandF <std::complex<PREC> > {
public:
	static std::complex<PREC> r(std::complex<PREC> o, PREC r) {
		PREC arg = RandF<PREC>::i(0,2*PI);
		PREC s = RandF<PREC>::i(0,r);

		return o + std::complex<PREC>(s*cos(arg), s*sin(arg));
	}
	static std::complex<PREC> i(
			std::complex<PREC> low,
			std::complex<PREC> high) {
		PREC p = RandF<PREC>::i(0,1);
		return (1.0-p)*low + p*high;
	}
};

// Object to string functions and helpers

template <typename T>
class ToStringHelper {
public:
	static std::string to(const T& value) {
		std::stringstream ss;
		ss << value;
		return ss.str();
	}
};

template <>
class ToStringHelper<cl::Buffer> {
public:
	static std::string to(const cl::Buffer& value) {
		return "<device address>";
	}
};

template <>
class ToStringHelper<bool> {
public:
	static std::string to(const bool& value) {
		return value ? "true" : "false";
	}
};

template <>
class ToStringHelper<PscrCLMode> {
public:
	static std::string to(const PscrCLMode& value) {
		return std::string("mode(") + 
			"complex=" + ToStringHelper<bool>::to(value.numComplex()) + "," + 
			"double=" + ToStringHelper<bool>::to(value.precDouble()) + "," +
			"m1Tridiag=" + ToStringHelper<bool>::to(value.m1Tridiag()) + "," +
			"m2Tridiag=" + ToStringHelper<bool>::to(value.m2Tridiag()) + "," +
			"m3Tridiag=" + ToStringHelper<bool>::to(value.m3Tridiag()) + ")";
	}
};

template <typename T>
std::string toString(const T& value) {
	return ToStringHelper<T>::to(value);
}

// Mode flag parsers

size_t getVarSize(const PscrCLMode &mode);
int getDSize(const PscrCLMode &mode);

// Error message transformer

std::string CLErrorMessage(cl_int code);

}

#endif
