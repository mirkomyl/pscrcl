
/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <time.h>
#include <stdio.h>
#include <complex>

#include "common.h"
#include "DeviceInformation.h"
#include "L3DeviceContext.h"

#include "cl/source.h"

// An estimate for the amount of local memory used to store the kernel
// arguments
#define CR_LOCAL_MEM_BIAS (10*sizeof(cl_mem))

// Kernel argument indexes
#define KARGS_GEN_GLOBAL_SYS_TMP				0
#define KARGS_GEN_GLOBAL_SYS_COEF_MATRIX		1
#define KARGS_GEN_GLOBAL_SYS_LAMBDA1			2
#define KARGS_GEN_GLOBAL_SYS_LAMBDA2			3
#define KARGS_GEN_GLOBAL_SYS_LAMBDA1_STRIDE		4
#define KARGS_GEN_GLOBAL_SYS_LAMBDA2_STRIDE		5
#define KARGS_GEN_GLOBAL_SYS_CH					6

#define KARGS_A1_DATA							0
#define KARGS_A1_TMP							1
#define KARGS_A1_R								2

#define KARGS_A2_DATA							0
#define KARGS_A2_TMP							1
#define KARGS_A2_R								2

#define KARGS_BCD_GEN_SYS_DATA					0
#define KARGS_BCD_GEN_SYS_COEF_MATRIX			1
#define KARGS_BCD_GEN_SYS_LAMBDA1				2
#define KARGS_BCD_GEN_SYS_LAMBDA2				3
#define KARGS_BCD_GEN_SYS_LAMBDA1_STRIDE		4
#define KARGS_BCD_GEN_SYS_LAMBDA2_STRIDE		5
#define KARGS_BCD_GEN_SYS_CH					6

#define KARGS_BCD_CPY_SYS_DATA					0
#define KARGS_BCD_CPY_SYS_TMP					1
#define KARGS_BCD_CPY_SYS_R						2

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

L3DeviceContext::L3DeviceContext(
		cl::Context      &_context,
		cl::Device       &_device,
		const OptValues  &_optValues,
		int               _n3,
		int               _l3Ldf1,
		int               _l3Ldf2,
		int               _l3MatLdf,
		bool              _multibleLambda,
		const PscrCLMode &_mode) :
		optimizer(createOptimizer(_device, _n3, _mode)),
		optValues(_optValues),
		helper(_context, _device, _mode),
		mode(_mode) {

#if DEBUG
	std::cout << debugLocation <<
			"L3DeviceContext::L3DeviceContext: Initializing solver, " <<
			"n3 = " << _n3 << ", " <<
			"mode = " << toString(_mode) << ". " << std::endl;
#endif

	argsAreSet = false;
	n3 = _n3;

	std::string check;
	if(!checkOptValues(_device, _optValues, _n3, _mode, &check))
		throw InvalidOptParams(check);
	
	DeviceInformation deviceHelper(_device);

	const int l3LocalMemSize = optimizer.interped(
		PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE, optValues);
	const int l3AWGSize = optimizer.interped(
		PSCRCL_OPT_PARAM_L3_A_WG_SIZE, optValues);
	
	// Calculate l3Ldf3
	if(n3 <= l3LocalMemSize) {
		int wBSize = deviceHelper.getWBSize();
		l3Ldf3 = DIVCEIL(n3, wBSize) * wBSize;
	} else {
		l3Ldf3 = l3AWGSize;
		
		while(l3Ldf3 < n3)
			l3Ldf3 *= 2;
	}
		
	cl::Program::Sources container;
	container.push_back(cl::Program::Sources::value_type((const char*) common_cl, common_cl_len));
	container.push_back(cl::Program::Sources::value_type((const char*) l3_kernel_cl, l3_kernel_cl_len));

	std::string additionalArgs =
			" -D N3=" + toString(n3) +
			" -D K3=" + toString(K_2(n3)) +
			" -D L3_LDF1=" + toString(_l3Ldf1) +
			" -D L3_LDF2=" + toString(_l3Ldf2) +
			" -D L3_LDF3=" + toString(getL3Ldf3()) +
			" -D L3_MAT_LDF=" + toString(_l3MatLdf) +
			" -D L2_MAT_LDF=0 -D L1_MAT_LDF=0 ";
	if(mode.m3Tridiag())
		additionalArgs += " -D M3_TRIDIAG=1";
	
	if(_multibleLambda)
		additionalArgs += " -D MULTIBLE_LAMBDA=1";
	else
		additionalArgs += " -D MULTIBLE_LAMBDA=0";
	
	helper.compileSource(optimizer, optValues, additionalArgs, container);

#if DEBUG
	std::cout << debugLocation <<
			"L3DeviceContext::L3DeviceContext: " \
			"Solver initialized." << std::endl;
#endif
}

bool L3DeviceContext::checkOptValues(
		const cl::Device &_device,
		const OptValues  &_optValues,
		int               _n3,
		const PscrCLMode &_mode,
		std::string      *_err) {
	
	Optimizer optimizer = createOptimizer(_device, _n3, _mode);
	
	DeviceInformation deviceHelper(_device);

	const int l3LocalMemSize = optimizer.interped(
		PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE, _optValues);
	const int l3AWGSize = optimizer.interped(
		PSCRCL_OPT_PARAM_L3_A_WG_SIZE, _optValues);
	const int l3BCDWGSize = optimizer.interped(
		PSCRCL_OPT_PARAM_L3_BCD_WG_SIZE, _optValues);
	
	const bool stageAEnabled = l3LocalMemSize < _n3;
    
	bool stageBEnabled = (stageAEnabled && l3BCDWGSize < l3AWGSize/2) ||
		(!stageAEnabled && l3BCDWGSize < _n3/2);
	
	// Check parameters

	const int dLim = getDSize(_mode);
	const int varSize = getVarSize(_mode);
	const int wBSize = deviceHelper.getWBSize();

	const long maxLocalMemSize = deviceHelper.getMaxLocalMemSize();
	//
	const int minWGSize = deviceHelper.getMinWGSize();
	const int maxWGSize = deviceHelper.getMaxWGSize();

	const int l3D = optimizer.interped(
			PSCRCL_OPT_PARAM_L3_D, _optValues);
	const int l3GenGlobalSysWGSize = optimizer.interped(
			PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_SIZE, _optValues);
	const int l3GenGlobalSysWGPerSys = optimizer.interped(
			PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_PER_SYS, _optValues);
	const int l3PCRLimit = optimizer.interped(
		PSCRCL_OPT_PARAM_L3_PCR_LIMIT, _optValues);
	const int l3PCRSteps = optimizer.interped(
		PSCRCL_OPT_PARAM_L3_PCR_STEPS, _optValues);

	// 1 <= L3_D <= 1/2/4
	if(l3D < 1 || dLim < l3D) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_D.";
		return false;
	}

	// General work group size checks
	
	if(l3GenGlobalSysWGSize < minWGSize ||
			maxWGSize < l3GenGlobalSysWGSize) {
		if(_err)
			*_err = "Invalid L3_GEN_GLOBAL_SYS_WG_SIZE.";
		return false;
	}

	if(l3AWGSize < minWGSize || maxWGSize < l3AWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_A_WG_SIZE.";
		return false;
	}

	if(l3BCDWGSize < minWGSize || maxWGSize < l3BCDWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_BCD_WG_SIZE.";
		return false;
	}
	
	// Calculate l3Ldf3
	int l3Ldf3;
	if(!stageAEnabled) {
		l3Ldf3 = DIVCEIL(_n3, wBSize) * wBSize;
	} else {
		l3Ldf3 = l3AWGSize;
		
		while(l3Ldf3 < _n3)
			l3Ldf3 *= 2;
	}

	// Check L3_GEN_GLOBAL_SYS_WG_SIZE and L3_GEN_GLOBAL_SYS_WG_PER_SYS
	
	if(l3GenGlobalSysWGPerSys < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_PER_SYS.";
		return false;
	}
		
	if(l3Ldf3 < l3GenGlobalSysWGSize*l3GenGlobalSysWGPerSys) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_SIZE " \
				"or PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_PER_SYS.";
		return false;
	}

	// Avoid having too much work-items
		
	if(stageAEnabled && l3Ldf3 < 2*l3AWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_A_WG_SIZE.";
		return false;
	}
	
	if(!stageAEnabled && DIVCEIL(_n3, wBSize) * wBSize < l3BCDWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_BCD_WG_SIZE";
		return false;
	}

	// If stage A is enabled, then L3_BCD_WG_SIZE <= L3_A_WG_SIZE / 2
	if(stageAEnabled && l3AWGSize/2 < l3BCDWGSize) {
		if(_err)
			*_err = "Invalid L3_A_WG_SIZE or L3_BCD_WG_SIZE";
		return false;
	}

	// Local memory usage limits
	
	if(maxLocalMemSize < 4*varSize*l3AWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_A_WG_SIZE.";
		return false;
	}
	
	if(maxLocalMemSize < 4*varSize*l3LocalMemSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE";
		return false;
	}
	
	if(l3LocalMemSize < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE";
		return false;
	}
	
	// The amount of allocated local memory must be a multiple of the work
	// group size when the stage B is enabled.
	if(stageBEnabled && l3LocalMemSize % (4*l3BCDWGSize) != 0) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE " \
			"or PSCRCL_OPT_PARAM_L3_BCD_WG_SIZE";
		return false;
	}
	
	if(stageBEnabled && l3LocalMemSize != NEXTPOW2(l3LocalMemSize)) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE.";
		return false;
	}
			
	// Check PCR limit
	if(l3PCRLimit < 0) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_PCR_LIMIT";
		return false;
	}
	
	// Check PCR steps
	if(l3PCRSteps < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_PCR_STEPS";
		return false;
	}
	
	// Local memory size sanity checks 
	if(l3LocalMemSize < l3BCDWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE " \
			"or PSCRCL_OPT_PARAM_L3_BCD_WG_SIZE";
		return false;
	}
		
	if(stageAEnabled && l3AWGSize != l3LocalMemSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE " \
			"or PSCRCL_OPT_PARAM_L3_A_WG_SIZE";
		return false;
	}
	
	// Everything is ok.
	if(_err)
		*_err = "";
	return true;
}

void L3DeviceContext::setArgs(
		const cl::Buffer& _devMemF,
		const cl::Buffer& _devMemTmp,
		const cl::Buffer& _devMemLambda1,
		const cl::Buffer& _devMemLambda2,
		const cl::Buffer& _devMemCoefMatrix) {

#if DEBUG
	std::cout << debugLocation <<
			"L3DeviceContext::setArgs: " \
			"Setting initial kernel arguments..." << std::endl;
#endif

	if(!helper.isCompiled()) {
			std::cerr << errorLocation <<
					"L3DeviceContext::setArgs: " \
					"Cannot set initial arguments for the L3 solver. " \
					"The source code has not been compiled." << std::endl;
			throw UnknownError("The source code has not been compiled.");
	}

	if(stageAEnabled()) {
		helper.setKernelArg("l3_gen_glo_sys",
				KARGS_GEN_GLOBAL_SYS_TMP, _devMemTmp);
		helper.setKernelArg("l3_gen_glo_sys",
				KARGS_GEN_GLOBAL_SYS_LAMBDA1, _devMemLambda1);
		helper.setKernelArg("l3_gen_glo_sys",
				KARGS_GEN_GLOBAL_SYS_LAMBDA2, _devMemLambda2);
		helper.setKernelArg("l3_gen_glo_sys",
				KARGS_GEN_GLOBAL_SYS_COEF_MATRIX, _devMemCoefMatrix);

		helper.setKernelArg("l3_bcd_cpy_sys",
				KARGS_BCD_CPY_SYS_DATA, _devMemF);
		helper.setKernelArg("l3_bcd_cpy_sys",
				KARGS_BCD_CPY_SYS_TMP, _devMemTmp);


		helper.setKernelArg("l3_a1", KARGS_A1_DATA, _devMemF);
		helper.setKernelArg("l3_a1", KARGS_A1_TMP, _devMemTmp);

		helper.setKernelArg("l3_a2", KARGS_A2_DATA, _devMemF);
		helper.setKernelArg("l3_a2", KARGS_A2_TMP, _devMemTmp);
	} else {
		helper.setKernelArg("l3_bcd_gen_sys",
				KARGS_BCD_GEN_SYS_DATA, _devMemF);
		helper.setKernelArg("l3_bcd_gen_sys",
				KARGS_BCD_GEN_SYS_LAMBDA1, _devMemLambda1);
		helper.setKernelArg("l3_bcd_gen_sys",
				KARGS_BCD_GEN_SYS_LAMBDA2, _devMemLambda2);
		helper.setKernelArg("l3_bcd_gen_sys",
				KARGS_BCD_GEN_SYS_COEF_MATRIX, _devMemCoefMatrix);
	}

	argsAreSet = true;
}

int L3DeviceContext::getL3Ldf3() const {
	return l3Ldf3;
}

size_t L3DeviceContext::getRequiredTmpSizePerSystem() {
		int local_mem_size = optimizer.interped(
			PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE, optValues);
	
	// Stage A not neccessary?
	if(n3 <= local_mem_size) 
		return 0;
	
	return 3*getL3Ldf3()*getVarSize(mode);
}


void L3DeviceContext::run(
		CommandQueue &_queue,
		int           _lambda1Stride,
		int           _lambda2Stride,
		int	          _count1,
		int           _count2,
		const void   *_ch) {

	if(!helper.isCompiled()) {
			std::cerr << errorLocation <<
					"L3DeviceContext::run: " \
					"Cannot run solver. " \
					"The source code has not been compiled." << std::endl;
			throw UnknownError();
	}

	if(!argsAreSet) {
		std::cerr << errorLocation <<
				"L3DeviceContext::run: " \
				"Cannot run solver. " \
				"Initial kernel arguments are not set properly." << std::endl;
		throw UnknownError();
	}

#if FULL_DEBUG
	std::cout << debugLocation <<
			"L3DeviceContext::run: Queuing kernels, " <<
			"lambda1Stride = " << _lambda1Stride << ", " <<
			"lambda2Stride = " << _lambda2Stride << ", " <<
			"count1 = " << _count1 << ", " <<
			"count2 = " << _count2 << ", " <<
			"ch = ";

	if(mode.numComplex())
		if(mode.precDouble())
			std::cout << *((std::complex<double>*)_ch);
		else
			std::cout << *((std::complex<float>*)_ch);
	else
		if(mode.precDouble())
			std::cout << *((double*)_ch);
		else
			std::cout << *((float*)_ch);

	std::cout << "..." << std::endl;
#endif

	if(stageAEnabled()) {

#if FULL_DEBUG
			std::cout << debugLocation <<
					"L3DeviceContext::run: " \
					"Preparing to queue global system generator kernel..." <<
					std::endl;
#endif

		helper.setKernelArg("l3_gen_glo_sys",
				KARGS_GEN_GLOBAL_SYS_LAMBDA1_STRIDE, _lambda1Stride);
		helper.setKernelArg("l3_gen_glo_sys",
				KARGS_GEN_GLOBAL_SYS_LAMBDA2_STRIDE, _lambda2Stride);

		// Set ch
		if(mode.numComplex()) {
			if(mode.precDouble()) {
				helper.setKernelArg(
						"l3_gen_glo_sys",
						KARGS_GEN_GLOBAL_SYS_CH,
						*((std::complex<double>*)_ch));
			} else {
				helper.setKernelArg(
						"l3_gen_glo_sys",
						KARGS_GEN_GLOBAL_SYS_CH,
						*((std::complex<float>*)_ch));
			}
		} else {
			if(mode.precDouble()) {
				helper.setKernelArg(
						"l3_gen_glo_sys",
						KARGS_GEN_GLOBAL_SYS_CH,
						*((double*)_ch));
			} else {
				helper.setKernelArg(
						"l3_gen_glo_sys",
						KARGS_GEN_GLOBAL_SYS_CH,
						*((float*)_ch));
			}
		}

		// Generate global systems
		helper.enqueueKernel(
				_queue, "l3_gen_glo_sys",
				KernelHelper::GroupCount(
						_count1, _count2,
						optimizer.interped(
								PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_PER_SYS,
								optValues)),
				optimizer.interped(PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_SIZE,
						optValues));

	} else {

		helper.setKernelArg("l3_bcd_gen_sys",
				KARGS_BCD_GEN_SYS_LAMBDA1_STRIDE, _lambda1Stride);
		helper.setKernelArg("l3_bcd_gen_sys",
				KARGS_BCD_GEN_SYS_LAMBDA2_STRIDE, _lambda2Stride);

		// Set ch
		if(mode.numComplex())
			if(mode.precDouble())
				helper.setKernelArg("l3_bcd_gen_sys",
						KARGS_BCD_GEN_SYS_CH, *((std::complex<double>*)_ch));
			else
				helper.setKernelArg("l3_bcd_gen_sys",
						KARGS_BCD_GEN_SYS_CH, *((std::complex<float>*)_ch));
		else
			if(mode.precDouble())
				helper.setKernelArg("l3_bcd_gen_sys",
						KARGS_BCD_GEN_SYS_CH, *((double*)_ch));
			else
				helper.setKernelArg("l3_bcd_gen_sys",
						KARGS_BCD_GEN_SYS_CH, *((float*)_ch));
	}

	int parallelStageA1 = optimizer.interped(
			PSCRCL_OPT_PARAM_L3_PARALLEL_STAGE_A1, optValues);
	int parallelStageA2 = optimizer.interped(
			PSCRCL_OPT_PARAM_L3_PARALLEL_STAGE_A2, optValues);
	int localMemSize = optimizer.interped(
			PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE, optValues);
	int bcdWGSize = optimizer.interped(
			PSCRCL_OPT_PARAM_L3_BCD_WG_SIZE, optValues);
	int aWGSize = optimizer.interped(
			PSCRCL_OPT_PARAM_L3_A_WG_SIZE, optValues);

	//int cr_ldf = getL3Ldf3();
	int n = getN3();
	//int k = K_2(n);

	if(stageAEnabled()) {
		int r = 1;
		while(localMemSize < n/POW2(r-1)) {

#if FULL_DEBUG
			std::cout << debugLocation <<
					"L3DeviceContext::run: " \
					"Preparing to queue stage A reduction kernel, " <<
					"r = " << toString(r) << "..." << std::endl;
#endif

			int totalPartCount = DIVCEIL(n, POW2(r)*aWGSize);
			int wGCount2 = parallelStageA1 ? totalPartCount : 1;

			helper.setKernelArg("l3_a1", KARGS_A1_R, r);
			helper.enqueueKernel(_queue, "l3_a1",
					KernelHelper::GroupCount(_count1, _count2, wGCount2),
					aWGSize, "r = " + toString(r));

			if(parallelStageA1) {
				r++;
			} else {
				r += LOG2(n/aWGSize) + 1;
				break;
			}
		}

#if FULL_DEBUG
		std::cout << debugLocation <<
					"L3DeviceContext::run: " \
					"Preparing to queue kernel for stages B, C and D, " <<
					"initial r = " << toString(r) << "..." << std::endl;
#endif

		helper.setKernelArg("l3_bcd_cpy_sys", KARGS_BCD_CPY_SYS_R, r);
		helper.enqueueKernel(_queue, "l3_bcd_cpy_sys",
				KernelHelper::GroupCount(_count1, _count2),
				bcdWGSize, "r = " + toString(r));

		r -= 2;
		while(0 <= r) {

#if FULL_DEBUG
			std::cout << debugLocation <<
					"L3DeviceContext::run: " \
					"Preparing to queue stage A back substitution kernel, " <<
					"r = " << toString(r) << "..." << std::endl;
#endif

			int totalPartCount = DIVCEIL(n, POW2(r+1)*aWGSize);
			int wGCount2 = parallelStageA2 ? totalPartCount : 1;

			helper.setKernelArg("l3_a2", KARGS_A2_R, r);
			helper.enqueueKernel(_queue, "l3_a2",
					KernelHelper::GroupCount(_count1, _count2, wGCount2),
					aWGSize, "r = " + toString(r));
			
			if(parallelStageA2)
				r--;
			else 
				r = -1;
		}
	} else {

#if FULL_DEBUG
		std::cout << debugLocation <<
					"L3DeviceContext::run: " \
					"Preparing to queue kernel for stages B, C and D..." <<
					std::endl;
#endif

		helper.enqueueKernel(_queue, "l3_bcd_gen_sys",
				KernelHelper::GroupCount(_count1, _count2), bcdWGSize);
	}

#if FULL_DEBUG
			std::cout << debugLocation <<
					"L3DeviceContext::run: All kernels queued." << std::endl;
#endif

}

Optimizer L3DeviceContext::createOptimizer(
			const cl::Device &_device,
			int               _n3,
			const PscrCLMode &_mode) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"L3DeviceContext::createOptimizer: Creating an Optimizer object..." 
		<< std::endl;
#endif

	Optimizer optimizer;

	//
	// Get device specific parameters
	//

	int dLim = getDSize(_mode);
	int varSize = getVarSize(_mode);

	DeviceInformation deviceHelper(_device);
	long maxLocalMemSize = deviceHelper.getMaxLocalMemSize();
	int wBSize = deviceHelper.getWBSize();
	int minWGSize = deviceHelper.getMinWGSize();
	int maxWGSize = 256;//deviceHelper.getMaxWGSize();

	// Calculate vector size
	int l3D;
	if(dLim*wBSize <= _n3)
		l3D = dLim;
	else
		l3D = 1;

	//
	// Add global parameters
	//

	// Add vector size opt. parameter
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_D,
					"l3_d", "D",
					Pow2OptValueInterpreter(), l3D));

	// Add local memory storage pattern opt. parameter
	if(_mode.precDouble())
		optimizer.addParam(
				OptParam(
						PSCRCL_OPT_PARAM_L3_HILODOUBLE,
						"l3_hilodouble", "HILODOUBLE",
						BoolOptValueInterpreter(), 0));

	// Add global memory access pattern opt. parameter, i.e., use local memory
	// and synchronization when accessing isolated global memory addresses
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_SHARED_ISOLATED_ACCESS,
					"l3_shared_isolated_access", "L3_SHARED_ISOLATED_ACCESS",
					BoolOptValueInterpreter(), 0));
	
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_NEWTON_DIV,
					"l3_newton_div", "NEWTON_DIV",
					BoolOptValueInterpreter(), 1));

	//
	// System generation related parameters
	//

	// Calculate work group size for the system generation kernel
	int l3GenGlobalSysWGSize = minWGSize;
	while(l3GenGlobalSysWGSize < _n3 &&
			l3GenGlobalSysWGSize+wBSize <= maxWGSize)
		l3GenGlobalSysWGSize += wBSize;

	// Add work group size opt. parameter for the system generation kernel
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_SIZE,
					"l3_l3GenGlobalSysWGSize", "L3_GEN_GLOBAL_SYS_WG_SIZE",
					AddOptValueInterpreter(wBSize), l3GenGlobalSysWGSize));

	int l3GenGlobalSysWGPerSys = MAX(1, _n3 / l3GenGlobalSysWGSize);

	// Add "number of work groups assigned to one system" opt. parameter
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_GEN_GLOBAL_SYS_WG_PER_SYS,
					"l3_l3GenGlobalSysWGPerSys",
					"L3_GEN_GLOBAL_SYS_WG_PER_SYS",
					AbsOptValueInterpreter(), l3GenGlobalSysWGPerSys));

	//
	// Calculate parameters dependent on stage of the stage A boolean
	//

	// Enable stage A if there is not enough local memory
	int l3StageA = maxLocalMemSize < _n3*4*varSize ? 1 : 0;

	int l3AWGSize, l3StageB, l3BCDWGSize, l3LocalMemSize;
	
	for(;;) {
		if(l3StageA) {

			// Calculate work group size for the stages A and B/C/D
			l3AWGSize = 2*minWGSize;
			l3BCDWGSize = minWGSize;
			while(2*l3AWGSize < _n3 && 2*(l3BCDWGSize+wBSize) <= maxWGSize &&
					2*4*varSize*(l3BCDWGSize+wBSize) <= maxLocalMemSize) {
				l3BCDWGSize += wBSize;
				l3AWGSize = 2*l3BCDWGSize;
			}

			// The number of remaining even numbered rows is already smaller than
			// the maximum work group size. Thus there is no need for the stage B
			l3StageB = 0;

			// Set the amount of local memory
			// TODO: Figure out a more optimal setting
			l3LocalMemSize = 2*l3BCDWGSize;
		} else {
			// Enable stage B if the maximum size of the work group is smaller then
			// the number of even numbered rows.
			l3StageB = 2*maxWGSize < _n3;

			// Calculate the work group size for stages B, C and D
			l3BCDWGSize = minWGSize;
			while(2*l3BCDWGSize <= _n3 && l3BCDWGSize+wBSize <= maxWGSize && 
					2*4*varSize*(l3BCDWGSize+wBSize) <= maxLocalMemSize)
				l3BCDWGSize += wBSize;

			// These have no effect when using the default parameters. However,
			// it is a good idea to set these to some reasonable values.
			l3AWGSize = 2*l3BCDWGSize;

			// Set the amount of local memory
			if(l3StageB) {
				l3LocalMemSize = 2*4*l3BCDWGSize;
				while(l3LocalMemSize < _n3)
					l3LocalMemSize *= 2;
			} else {
				l3LocalMemSize = wBSize;
				while(l3LocalMemSize < _n3)
					l3LocalMemSize += wBSize;
			}
		}
		
		if(!l3StageA && maxLocalMemSize < 4*l3LocalMemSize*varSize)
			l3StageA = 1;
		else 
			break;
	} 

	//
	// Add parameters dependent on stage of the stage A boolean
	//

	// Add stage A work group size opt. parameter
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_A_WG_SIZE,
					"l3_l3AWGSize", "L3_A_WG_SIZE",
					AddOptValueInterpreter(wBSize), l3AWGSize));

	// Add stage B/C/D work group size opt. parameter
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_BCD_WG_SIZE,
					"l3_l3BCDWGSize", "L3_BCD_WG_SIZE",
					AddOptValueInterpreter(wBSize), l3BCDWGSize));

	// Add local memory buffer size opt. parameter
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE,
					"l3_local_mem_size", "L3_LOCAL_MEM_SIZE",
					AddOptValueInterpreter(wBSize), l3LocalMemSize));

	//
	// Other parameters
	//

	// Add a parameter which specifies when the size of the reduced system is
	// small enough to be solved using the parallel cyclic reduction stage
	// FIXME: In some reason, large values will mess of the whole process
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_PCR_LIMIT,
					"l3_pcr_limit", "L3_PCR_LIMIT",
					AddOptValueInterpreter(wBSize), wBSize));
	
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_PCR_STEPS,
					"l3_pcr_steps", "L3_PCR_STEPS",
					AbsOptValueInterpreter(), LOG2(wBSize)));
	

	// During the stage A, each sub-system is divided into multiple segments.
	// The segments are processed in groups of four. If this parameter is 1,
	// then these groups are processed in parallel
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_PARALLEL_STAGE_A1,
					"l3_parallel_stage_a1", "L3_PARALLEL_STAGE_A1",
					BoolOptValueInterpreter(), 1));

	// A similar parameter for the back substitution stage
	optimizer.addParam(
			OptParam(
					PSCRCL_OPT_PARAM_L3_PARALLEL_STAGE_A2,
					"l3_parallel_stage_a2", "L3_PARALLEL_STAGE_A2",
					BoolOptValueInterpreter(), 1));

	return optimizer;
}


int L3DeviceContext::getN3() const {
	return n3;
}

bool L3DeviceContext::stageAEnabled() const {
	int local_mem_size =
			optimizer.interped(PSCRCL_OPT_PARAM_L3_LOCAL_MEM_SIZE, optValues);

	return local_mem_size < n3;
}

bool L3DeviceContext::stageBEnabled() const {
	int l3AWGSize =
			optimizer.interped(PSCRCL_OPT_PARAM_L3_A_WG_SIZE, optValues);
	int l3BCDWGSize =
			optimizer.interped(PSCRCL_OPT_PARAM_L3_BCD_WG_SIZE, optValues);

	return (stageAEnabled() && l3BCDWGSize < l3AWGSize/2) ||
		(!stageAEnabled() && l3BCDWGSize < n3/2);
}

