/*
 *  Created on: Nov 28, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "DeviceContextHelper.h"
#include "common.h"
#include "DeviceInformation.h"

#include "cl/source.h"

// These should match the ones in pscrCL.h
#define LX_D								 1
#define LX_SHARED_ISOLATED_ACCESS			 2
#define LX_NEWTON_DIV						 3
#define LX_STAGE_11_WG_SIZE					 4
#define LX_STAGE_11_WG_PER_VECTOR			 5
#define LX_STAGE_12A_WG_SIZE				 6
#define LX_STAGE_12A_WG_PER_VECTOR			 7
#define LX_STAGE_12C_WG_SIZE				 8
#define LX_STAGE_12C_WG_PER_VECTOR			 9
#define LX_STAGE_Y2A_MAX_SUM_SIZE_EXP		10
#define LX_STAGE_Y2B_MAX_SUM_SIZE_EXP		11
#define LX_STAGE_Y2B_WG_SIZE				12
#define LX_STAGE_Y2B_WG_PER_VECTOR			13
#define LX_STAGE_21_WG_SIZE					14
#define LX_STAGE_21_WG_PER_VECTOR			15
#define LX_STAGE_22A_WG_SIZE				16
#define LX_STAGE_22A_WG_PER_VECTOR			17
#define LX_STAGE_22C_WG_SIZE				18
#define LX_STAGE_22C_WG_PER_VECTOR			19
#define LX_VECTOR_LOAD_HELPER				20
#define LX_MATRIX_LOAD_HELPER				21

#define KARGS_STAGE_11_F					0
#define KARGS_STAGE_11_G					1
#define KARGS_STAGE_11_GUIDE1				2
#define KARGS_STAGE_11_GUIDE2				3
#define KARGS_STAGE_11_EIGEN				4
#define KARGS_STAGE_11_I					5

#define KARGS_STAGE_12A_V					0
#define KARGS_STAGE_12A_TMP					1
#define KARGS_STAGE_12A_GUIDE				2
#define KARGS_STAGE_12A_EIGEN				3
#define KARGS_STAGE_12A_I					4
#define KARGS_STAGE_12A_MAX_SUM_SIZE		5

#define KARGS_STAGE_Y2B_TMP					0
#define KARGS_STAGE_Y2B_GUIDE				1
#define KARGS_STAGE_Y2B_SUM_STEP			2
#define KARGS_STAGE_Y2B_I					3
#define KARGS_STAGE_Y2B_MAX_SUM_SIZE		4

#define KARGS_STAGE_12C_F					0
#define KARGS_STAGE_12C_TMP					1
#define KARGS_STAGE_12C_GUIDE				2
#define KARGS_STAGE_12C_I					3
#define KARGS_STAGE_12C_CH					4
#define KARGS_STAGE_12C_MATX				5
#define KARGS_STAGE_12C_MATXM1				6
#define KARGS_STAGE_12C_LAMBDA				7
#define KARGS_STAGE_12C_LAMBDA_STRIDE		8
#define KARGS_STAGE_12C_MATXM2				7

#define KARGS_STAGE_21_F					0
#define KARGS_STAGE_21_G					1
#define KARGS_STAGE_21_GUIDE1				2
#define KARGS_STAGE_21_GUIDE2				3
#define KARGS_STAGE_21_EIGEN				4
#define KARGS_STAGE_21_I					5
#define KARGS_STAGE_21_CH					6
#define KARGS_STAGE_21_MATX					7
#define KARGS_STAGE_21_MATXM1				8
#define KARGS_STAGE_21_LAMBDA				9
#define KARGS_STAGE_21_LAMBDA_STRIDE		10
#define KARGS_STAGE_21_MATXM2				9

#define KARGS_STAGE_22A_V					0
#define KARGS_STAGE_22A_TMP					1
#define KARGS_STAGE_22A_GUIDE				2
#define KARGS_STAGE_22A_EIGEN				3
#define KARGS_STAGE_22A_I					4
#define KARGS_STAGE_22A_MAX_SUM_SIZE		5

#define KARGS_STAGE_22C_F					0
#define KARGS_STAGE_22C_TMP					1
#define KARGS_STAGE_22C_GUIDE				2
#define KARGS_STAGE_22C_I					3

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

// Select opt. parameter id prefix
int getIDPrefix(DeviceContextHelper::LevelType _level) {
    return _level == DeviceContextHelper::level2 ? 
		PSCRCL_OPT_PARAM_L2_PREFIX : PSCRCL_OPT_PARAM_L1_PREFIX;
}

DeviceContextHelper::DeviceContextHelper(
		cl::Context                 &_context,
		cl::Device                  &_device,
		const OptValues             &_optValues,
		const Boundaries            &_bounds,
		const EigenSectionInterface &_eigenSection,
		int                          _upperBound,
		int                          _lowerBound,
		int                          _l1MatLdf,
		int                          _l2MatLdf,
		int                          _l3MatLdf,
		int                          _lxLdf2,
		int                          _lxLdf3,
		int                          _lxp1Ldf2,
		int                          _lxp1Ldf3,
		int                          _nx,
		int                          _nxp1,
		int                          _l3Ldf1,
		int                          _n3,
		const PscrCLMode            &_mode,
		LevelType                    _level,
		bool                         _multibleLambda) :
		context(_context),
		kernelHelper(_context, _device, _mode),
		optimizer(createOptimizer(_device, _n3, _mode, _level)),
		optValues(_optValues),
		guide(
			_bounds, 
			POW2(optimizer.interped(
				getIDPrefix(_level)+LX_STAGE_Y2A_MAX_SUM_SIZE_EXP, _optValues)),
			POW2(optimizer.interped(
				getIDPrefix(_level)+LX_STAGE_Y2B_MAX_SUM_SIZE_EXP, _optValues)),
			_upperBound, 
			_lowerBound
 			),
		eigenSection(&_eigenSection),
		mode(_mode) {

#if DEBUG
	std::cout << debugLocation <<
		"DeviceContextHelper::DeviceContextHelper (level " << _level << 
		"): Initializing solver, " <<
		"upperBound = " << _upperBound << ", " <<
		"lowerBound = " << _lowerBound << ", " <<
		"n3 = " << _n3 << ", " <<
		"mode = " << toString(_mode) << ". " << std::endl;
#endif

	if(_level != level1 && _level != level2) {
		std::cerr << errorLocation <<
			"DeviceContextHelper::DeviceContextHelper (level " << level << 
			"): Invalid level." << std::endl;
		throw UnknownError();
	}
		
	argsAreSet = false;
	isAllocated = false;
	chIsSet = false;
	
	mode = _mode;
	level = _level;
	multibleLambda = _multibleLambda;
	pn = _lowerBound - _upperBound;
	nx = _nx;
	n3 = _n3;
	nxp1 = _nxp1;
	lxLdf3 = _lxLdf3;
	
	std::string check;
	if(!checkOptParams(
		_device, _optValues, _lxLdf3, _n3, _mode, _level, &check)) {
		throw InvalidOptParams(check);
	}
		
	const int idPrefix = getIDPrefix(_level);
		
	maxSumSizeA = POW2(optimizer.interped(
		idPrefix + LX_STAGE_Y2A_MAX_SUM_SIZE_EXP, _optValues));
	
	maxSumSizeB = POW2(optimizer.interped(
		idPrefix + LX_STAGE_Y2B_MAX_SUM_SIZE_EXP, _optValues));
		
	cl::Program::Sources container;
	container.push_back(cl::Program::Sources::value_type((const char*) common_cl, common_cl_len));
	container.push_back(cl::Program::Sources::value_type((const char*) lx_kernel_cl, lx_kernel_cl_len));
	
	std::string additionalArgs =
		(_level == level2 ? 
			std::string(" -D LEVEL2=1") : std::string(" -D LEVEL2=0")) + 
		" -D PNX=" + toString(_lowerBound - _upperBound) + 
		" -D L1_MAT_LDF=" + toString(_l1MatLdf) + 
		" -D L2_MAT_LDF=" + toString(_l2MatLdf) +
		" -D L3_MAT_LDF=" + toString(_l3MatLdf) +
		" -D L3_LDF1=" + toString(_l3Ldf1);
		
	if(_level == level2 && multibleLambda && mode.m2Tridiag())
		additionalArgs += " -D MULTIBLE_LAMBDA=1";
	else
		additionalArgs += " -D MULTIBLE_LAMBDA=0";
		
	if(_level == level1 && mode.m1Tridiag())
		additionalArgs += " -D M1_TRIDIAG=1";
	else
		additionalArgs += " -D M1_TRIDIAG=0";
	
	if(mode.m2Tridiag())
		additionalArgs += " -D M2_TRIDIAG=1";
	else
		additionalArgs += " -D M2_TRIDIAG=0";
	
	if(mode.m3Tridiag())
		additionalArgs += " -D M3_TRIDIAG=1";
	else
		additionalArgs += " -D M3_TRIDIAG=0";
	
	if(_level == level1) {
		additionalArgs += 
			" -D N1=" + toString(_nx) +
			" -D L1_LDF2=" + toString(_lxLdf2) +
			" -D L1_LDF3=" + toString(_lxLdf3) +
			" -D N2=" + toString(_nxp1) +
			" -D L2_LDF2=" + toString(_lxp1Ldf2) +
			" -D L2_LDF3=" + toString(_lxp1Ldf3) +
			" -D N3=" + toString(n3);
	}
	
	if(_level == level2) {
		additionalArgs += 
			" -D N2=" + toString(_nx) +
			" -D L2_LDF2=" + toString(_lxLdf2) +
			" -D L2_LDF3=" + toString(_lxLdf3) +
			" -D N3=" + toString(_nxp1) +
			" -D L3_LDF2=" + toString(_lxp1Ldf2) +
			" -D L3_LDF3=" + toString(_lxp1Ldf3);
	}

	kernelHelper.compileSource(optimizer, optValues, additionalArgs, container);
	
	kernelHelper.renameKernel("lx_stage_11",  getAlias("lx_stage_11"));
	kernelHelper.renameKernel("lx_stage_12a", getAlias("lx_stage_12a"));
	kernelHelper.renameKernel("lx_stage_12c", getAlias("lx_stage_12c"));
	kernelHelper.renameKernel("lx_stage_21",  getAlias("lx_stage_21"));
	kernelHelper.renameKernel("lx_stage_22a", getAlias("lx_stage_22a"));
	kernelHelper.renameKernel("lx_stage_22c", getAlias("lx_stage_22c"));
	kernelHelper.renameKernel("lx_stage_y2b", getAlias("lx_stage_y2b"));
		
#if DEBUG
	std::cout << debugLocation <<
		"DeviceContextHelper::DeviceContextHelper (level " << level << 
		"): Solver initialized." << std::endl;
#endif
	
}

void DeviceContextHelper::setArgs(
	cl::Buffer &_devMemF,
	cl::Buffer &_devMeG,
	cl::Buffer &_devMemTmp,
	cl::Buffer &_devMemMatX,
	cl::Buffer &_devMemMatXp1,
	cl::Buffer &_devMisc) {
	
#if DEBUG
	std::cout << debugLocation << 
		"DeviceContextHelper::setArgs (level " << level << 
		"): Setting initial kernel arguments..." << std::endl;
#endif

	argsAreSet = false;
		
	kernelHelper.setKernelArg(
		getAlias("lx_stage_11"), KARGS_STAGE_11_F, _devMemF);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_11"), KARGS_STAGE_11_G, _devMeG);
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12a"), KARGS_STAGE_12A_V, _devMeG);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12a"), KARGS_STAGE_12A_TMP, _devMemTmp);
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_y2b"), KARGS_STAGE_Y2B_TMP, _devMemTmp);
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12c"), KARGS_STAGE_12C_F, _devMemF);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12c"), KARGS_STAGE_12C_TMP, _devMemTmp);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12c"), KARGS_STAGE_12C_MATX, _devMemMatX);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12c"), KARGS_STAGE_12C_MATXM1, _devMemMatXp1);
	if(level == level2 && multibleLambda && mode.m2Tridiag())
		kernelHelper.setKernelArg(
			getAlias("lx_stage_12c"), KARGS_STAGE_12C_LAMBDA, _devMisc);
	if(level == level1)
		kernelHelper.setKernelArg(
			getAlias("lx_stage_12c"), KARGS_STAGE_12C_MATXM2, _devMisc);
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_21"), KARGS_STAGE_21_F, _devMemF);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_21"), KARGS_STAGE_21_G, _devMeG);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_21"), KARGS_STAGE_21_MATX, _devMemMatX);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_21"), KARGS_STAGE_21_MATXM1, _devMemMatXp1);
	if(level == level2 && multibleLambda && mode.m2Tridiag())
		kernelHelper.setKernelArg(
			getAlias("lx_stage_21"), KARGS_STAGE_21_LAMBDA, _devMisc);
	if(level == level1)
		kernelHelper.setKernelArg(
			getAlias("lx_stage_21"), KARGS_STAGE_21_MATXM2, _devMisc);
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_22a"), KARGS_STAGE_22A_V, _devMeG);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_22a"), KARGS_STAGE_22A_TMP, _devMemTmp);
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_22c"), KARGS_STAGE_22C_F, _devMemF);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_22c"), KARGS_STAGE_22C_TMP, _devMemTmp);
	
	argsAreSet = true;	
	
}

void DeviceContextHelper::setCh(const void* _ch) {
	
	chIsSet = false;
	
	// Set ch
	if(mode.numComplex()) {
		if(mode.precDouble()) {
			kernelHelper.setKernelArg(
					getAlias("lx_stage_12c"),
					KARGS_STAGE_12C_CH,
					*((std::complex<double>*)_ch));
			kernelHelper.setKernelArg(
					getAlias("lx_stage_21"),
					KARGS_STAGE_21_CH,
					*((std::complex<double>*)_ch));
		} else {
			kernelHelper.setKernelArg(
					getAlias("lx_stage_12c"),
					KARGS_STAGE_12C_CH,
					*((std::complex<float>*)_ch));
			kernelHelper.setKernelArg(
					getAlias("lx_stage_21"),
					KARGS_STAGE_21_CH,
					*((std::complex<float>*)_ch));
		}
	} else {
		if(mode.precDouble()) {
			kernelHelper.setKernelArg(
					getAlias("lx_stage_12c"),
					KARGS_STAGE_12C_CH,
					*((double*)_ch));
			kernelHelper.setKernelArg(
					getAlias("lx_stage_21"),
					KARGS_STAGE_21_CH,
					*((double*)_ch));
		} else {
			kernelHelper.setKernelArg(
					getAlias("lx_stage_12c"),
					KARGS_STAGE_12C_CH,
					*((float*)_ch));
			kernelHelper.setKernelArg(
					getAlias("lx_stage_21"),
					KARGS_STAGE_21_CH,
					*((float*)_ch));
		}
	}
	
	chIsSet = true;
}


void DeviceContextHelper::allocate(CommandQueue &_queue) {
	
#if DEBUG
	std::cout << debugLocation << 
		"DeviceContextHelper::allocate (level " << level << 
		"): Allocating device memory..." << std::endl;
#endif
	
	cl_int err;
	
	isAllocated = false;
	
	int totalAllocated = 0;
	
	// Allocate memory for eigenVectors
	
	eigenVectors = cl::Buffer(
		context, 
		CL_MEM_READ_WRITE, 
		eigenSection->getEigenVectorsSize(), 
		0, 
		&err);
	
	totalAllocated += eigenSection->getEigenVectorsSize();
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot allocate memory for eigendata. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	err = _queue.enqueueWriteBuffer(
		eigenVectors, 
		false, 
		0, 
		eigenSection->getEigenVectorsSize(), 
		eigenSection->getEigenVectors(),
		0, 0,
		"eigenVectors");
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot write into the eigenVectors buffer. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_11"), KARGS_STAGE_11_EIGEN, eigenVectors);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12a"), KARGS_STAGE_12A_EIGEN, eigenVectors);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_21"), KARGS_STAGE_21_EIGEN, eigenVectors);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_22a"), KARGS_STAGE_22A_EIGEN, eigenVectors);
	
	// StageY1 guiding information
	
	stageY1Guide1 = cl::Buffer(
		context, 
		CL_MEM_READ_WRITE, 
		guide.getStageY1GuideSize(),
		0, &err);
	
	totalAllocated += guide.getStageY1GuideSize();
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot allocate memory for stage11/stage21 guiding information. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	err = _queue.enqueueWriteBuffer(
		stageY1Guide1, 
		false, 
		0, 
		guide.getStageY1GuideSize(), 
		guide.getStageY1GuidePointer(),
		0, 0,
		"stageY1Guide1");
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot write into the stage11/stage21 guiding information " <<
		"buffer. " << CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	// StageY1 shared guiding information
	
	stageY1Guide2 = cl::Buffer(
		context, 
		CL_MEM_READ_WRITE, 
		guide.getStageY1SharedGuideSize(),
		0, &err);
	
	totalAllocated += guide.getStageY1SharedGuideSize();
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot allocate memory for stage11/stage21 shared guiding " <<
		"information. " << CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	err = _queue.enqueueWriteBuffer(
		stageY1Guide2, 
		false, 
		0, 
		guide.getStageY1SharedGuideSize(), 
		guide.getStageY1SharedGuidePointer(),
		0, 0,
		"stageY1Guide2");
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot write into the stage11/stage21 shared guiding " <<
		"information buffer. " << CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	// Stage12a guiding information
	
	stage12AGuide = cl::Buffer(
		context, 
		CL_MEM_READ_WRITE, 
		guide.getStage12AGuideSize(),
		0, &err);
	
	totalAllocated += guide.getStage12AGuideSize();
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot allocate memory for stage12a guiding information. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	err = _queue.enqueueWriteBuffer(
		stage12AGuide, 
		false, 
		0, 
		guide.getStage12AGuideSize(), 
		guide.getStage12AGuidePointer(),
		0, 0,
		"stage12AGuide");
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot write into the stage12a guiding information buffer. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	// Stage12b guiding information
	
	stage12BGuide = cl::Buffer(
		context, 
		CL_MEM_READ_WRITE, 
		guide.getStage12BGuideSize(),
		0, &err);
	
	totalAllocated += guide.getStage12BGuideSize();
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot allocate memory for stage12b guiding information. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	err = _queue.enqueueWriteBuffer(
		stage12BGuide, 
		false, 
		0, 
		guide.getStage12BGuideSize(), 
		guide.getStage12BGuidePointer(),
		0, 0,
		"stage12BGuide");
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot write into the stage12b guiding information buffer. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	// Stage12c guiding information
	
	stage12CGuide = cl::Buffer(
		context, 
		CL_MEM_READ_WRITE, 
		guide.getStage12CGuideSize(),
		0, &err);
	
	totalAllocated += guide.getStage12CGuideSize();
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot allocate memory for stage12c guiding information. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	err = _queue.enqueueWriteBuffer(
		stage12CGuide, 
		false, 
		0, 
		guide.getStage12CGuideSize(), 
		guide.getStage12CGuidePointer(),
		0, 0,
		"stage12CGuide");
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot write into the stage12c guiding information buffer. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	// Stage22a guiding information
	
	stage22AGuide = cl::Buffer(
		context, 
		CL_MEM_READ_WRITE, 
		guide.getStage22AGuideSize(),
		0, &err);
	
	totalAllocated += guide.getStage22AGuideSize();
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot allocate memory for stage22a guiding information. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	err = _queue.enqueueWriteBuffer(
		stage22AGuide, 
		false, 
		0, 
		guide.getStage22AGuideSize(), 
		guide.getStage22AGuidePointer(),
		0, 0,
		"stage22AGuide");
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot write into the stage22a guiding information buffer. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	// Stage22b guiding information
	
	stage22BGuide = cl::Buffer(
		context, 
		CL_MEM_READ_WRITE, 
		guide.getStage22BGuideSize(),
		0, &err);
	
	totalAllocated += guide.getStage22BGuideSize();
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot allocate memory for stage22b guiding information. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	err = _queue.enqueueWriteBuffer(
		stage22BGuide, 
		false, 
		0, 
		guide.getStage22BGuideSize(), 
		guide.getStage22BGuidePointer(),
		0, 0,
		"stage22BGuide");
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot write into the stage22b guiding information buffer. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	// Stage22c guiding information
	
	stage22CGuide = cl::Buffer(
		context, 
		CL_MEM_READ_WRITE, 
		guide.getStage22CGuideSize(),
		0, &err);
	
	totalAllocated += guide.getStage22CGuideSize();
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot allocate memory for stage22c guiding information. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	err = _queue.enqueueWriteBuffer(
		stage22CGuide, 
		false, 
		0, 
		guide.getStage22CGuideSize(), 
		guide.getStage22CGuidePointer(),
		0, 0,
		"stage22CGuide");
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Cannot write into the stage22c guiding information buffer. " << 
		CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	kernelHelper.setKernelArg(
		getAlias("lx_stage_11"), KARGS_STAGE_11_GUIDE1, stageY1Guide1);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_11"), KARGS_STAGE_11_GUIDE2, stageY1Guide2);
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12a"), KARGS_STAGE_12A_GUIDE, stage12AGuide);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12c"), KARGS_STAGE_12C_GUIDE, stage12CGuide);
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_21"), KARGS_STAGE_21_GUIDE1, stageY1Guide1);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_21"), KARGS_STAGE_21_GUIDE2, stageY1Guide2);
	
	kernelHelper.setKernelArg(
		getAlias("lx_stage_22a"), KARGS_STAGE_22A_GUIDE, stage22AGuide);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_22c"), KARGS_STAGE_22C_GUIDE, stage22CGuide);
	
#if FULL_DEBUG
	std::cout << debugLocation <<
		"DeviceContextHelper::allocate (level " << level << 
		"): Allocated " << totalAllocated << " bytes." << std::endl;
#endif
	
	isAllocated = true;
}

void DeviceContextHelper::free() {
	
#if DEBUG
	std::cout << debugLocation << 
		"DeviceContextHelper::free (level " << level << 
		"): Freeing device memory..." << std::endl;
#endif
	
	isAllocated = false;
	eigenVectors = cl::Buffer();
	stage12AGuide = cl::Buffer();
	stage12BGuide = cl::Buffer();
	stage12AGuide = cl::Buffer();
	stage12BGuide = cl::Buffer();
	stage12CGuide = cl::Buffer();
	stage22AGuide = cl::Buffer();
	stage22BGuide = cl::Buffer();
	stage22CGuide = cl::Buffer();
}

size_t DeviceContextHelper::getRequiredTmpSizePerSystem() {
/*
	const int k = LOG4(nx) + 1;

	int memAmount = 0;
	for(int i = 0; i <= k-1; i++) {
		const int partSumCountA = 
			DIVCEIL(guide.getHostGuide(i).maxSumSize, maxSumSizeA);
		
		memAmount = MAX(memAmount, guide.getHostGuide(i).stage1BSumCount * 
			partSumCountA);
		memAmount = MAX(memAmount, guide.getHostGuide(i).stage2BSumCount * 
			partSumCountA);
	}
	
 	return memAmount *
		(level == level1 ? nxp1*lxLdf3 : lxLdf3) * mode.getVarSize();
*/
	return pn*(level == level1 ? nxp1*lxLdf3 : lxLdf3) * getVarSize(mode);
}

int DeviceContextHelper::getSystemCount(int i) const {
	return guide.getHostGuide(i).totalSystemCount;
}

cl::Context& DeviceContextHelper::getContext() {
	return context;
}

void DeviceContextHelper::queueStage11(
		CommandQueue& _queue,
		int           _secondaryIndexSpaceSize,
		int           _i
		) {

#if FULL_DEBUG
	std::cout << debugLocation << 
		"DeviceContextHelper::queueStage11 (level " << level << 
		"): Enqueueing stage11..." << std::endl;
#endif
	
	checkStatus();
	
	kernelHelper.setKernelArg(getAlias("lx_stage_11"), KARGS_STAGE_11_I, _i);
	
    const int idPrefix = getIDPrefix(level);
	
	const int systemCount = guide.getHostGuide(_i-1).totalSystemCount;
	
	kernelHelper.enqueueKernel(
		_queue, getAlias("lx_stage_11"),
		KernelHelper::GroupCount(
			systemCount, _secondaryIndexSpaceSize, optimizer.interped(
				idPrefix + LX_STAGE_11_WG_PER_VECTOR, optValues)),
		optimizer.interped(idPrefix + LX_STAGE_11_WG_SIZE, optValues));
	
}

void DeviceContextHelper::queueStage12a(
		CommandQueue& _queue,
		int           _secondaryIndexSpaceSize,
		int           _i) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"DeviceContextHelper::queueStage12a (level " << level << 
		"): Enqueueing stage12 part A..." << std::endl;
#endif
	
	checkStatus();
	
	const int idPrefix = getIDPrefix(level);
	
	const int sumCount = guide.getHostGuide(_i-1).stage1SumCount;
	const int maxSumSize = guide.getHostGuide(_i-1).maxSumSize;
	const int bSumCount = guide.getHostGuide(_i-1).stage1BSumCount;
	
	// Enqueue stage_12a kernel
	
	const int partSumCountA = DIVCEIL(maxSumSize, maxSumSizeA);
	
	kernelHelper.setKernelArg(getAlias("lx_stage_12a"), KARGS_STAGE_12A_I, _i);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_12a"), KARGS_STAGE_12A_MAX_SUM_SIZE, maxSumSize);
	
	kernelHelper.enqueueKernel(
		_queue, getAlias("lx_stage_12a"),
		KernelHelper::GroupCount(
			sumCount*partSumCountA, 
			_secondaryIndexSpaceSize, 
			optimizer.interped(
				idPrefix + LX_STAGE_12A_WG_PER_VECTOR, 
				optValues)),
		optimizer.interped(idPrefix + LX_STAGE_12A_WG_SIZE, optValues));
	
	// Enqueue stage_y2b kernel
	
	int partSumCountB = partSumCountA == 1 ? 
		0 : DIVCEIL(partSumCountA, maxSumSizeB);
	
	if(0 < partSumCountB && 0 < bSumCount) {
		kernelHelper.setKernelArg(
			getAlias("lx_stage_y2b"), KARGS_STAGE_Y2B_I, _i);
		kernelHelper.setKernelArg(
			getAlias("lx_stage_y2b"), KARGS_STAGE_Y2B_MAX_SUM_SIZE, maxSumSize);
		kernelHelper.setKernelArg(
			getAlias("lx_stage_y2b"), KARGS_STAGE_Y2B_GUIDE, stage12BGuide);
	}
	
	for(int sumStep = 1; 0 < partSumCountB && 0 < bSumCount; sumStep++) {
		
		kernelHelper.setKernelArg(
			getAlias("lx_stage_y2b"), KARGS_STAGE_Y2B_SUM_STEP, sumStep);
		
		kernelHelper.enqueueKernel(
			_queue, getAlias("lx_stage_y2b"),
			KernelHelper::GroupCount(
				bSumCount*partSumCountB, 
				_secondaryIndexSpaceSize, 
				optimizer.interped(
					idPrefix + LX_STAGE_Y2B_WG_PER_VECTOR, 
					optValues)),
			optimizer.interped(idPrefix + LX_STAGE_Y2B_WG_SIZE, optValues));
		
		partSumCountB = partSumCountB == 1 ?
			0 : DIVCEIL(partSumCountB, maxSumSizeB);
	}
}

void DeviceContextHelper::queueStage12b(
		CommandQueue& _queue,
		int           _secondaryIndexSpaceSize,
		int           _i,
		int           _lambdaStride) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"DeviceContextHelper::queueStage12b (level " << level << 
		"): Enqueueing stage12 part B..." << std::endl;
#endif
	
	checkStatus();
	
	const int idPrefix = getIDPrefix(level);
	
	// Enqueue stage_12c kernel
	
	kernelHelper.setKernelArg(getAlias("lx_stage_12c"), KARGS_STAGE_12C_I, _i);
	if(level == level2 && multibleLambda && mode.m2Tridiag())
		kernelHelper.setKernelArg(getAlias("lx_stage_12c"), 
			KARGS_STAGE_12C_LAMBDA_STRIDE, _lambdaStride);
	
	
	kernelHelper.enqueueKernel(
		_queue, getAlias("lx_stage_12c"),
		KernelHelper::GroupCount(
			guide.getHostGuide(_i-1).stage1UpdateCount, 
			_secondaryIndexSpaceSize, 
			optimizer.interped(
				idPrefix + LX_STAGE_12C_WG_PER_VECTOR, 
				optValues)),
		optimizer.interped(idPrefix + LX_STAGE_12C_WG_SIZE, optValues));
}

void DeviceContextHelper::queueStage21(
		CommandQueue& _queue,
		int           _secondaryIndexSpaceSize,
		int           _i,
		int           _lambdaStride) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"DeviceContextHelper::queueStage21 (level " << level << 
		"): Enqueueing stage21..." << std::endl;
#endif
	
	checkStatus();
	
	kernelHelper.setKernelArg(getAlias("lx_stage_21"), KARGS_STAGE_21_I, _i);
	if(level == level2 && multibleLambda && mode.m2Tridiag())
		kernelHelper.setKernelArg(getAlias("lx_stage_21"), 
			KARGS_STAGE_21_LAMBDA_STRIDE, _lambdaStride);
	
    const int idPrefix = getIDPrefix(level);
	
	const int systemCount = guide.getHostGuide(_i).totalSystemCount;
	
	kernelHelper.enqueueKernel(
		_queue, getAlias("lx_stage_21"),
		KernelHelper::GroupCount(
			systemCount, _secondaryIndexSpaceSize, optimizer.interped(
				idPrefix + LX_STAGE_21_WG_PER_VECTOR, optValues)),
		optimizer.interped(idPrefix + LX_STAGE_21_WG_SIZE, optValues));
	
}


void DeviceContextHelper::queueStage22a(
		CommandQueue& _queue,
		int           _secondaryIndexSpaceSize,
		int           _i) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"DeviceContextHelper::queueStage22a (level " << level << 
		"): Enqueueing stage22 part A..." << std::endl;
#endif
	
	checkStatus();
	
	const int idPrefix = getIDPrefix(level);
	
	const int sumCount = guide.getHostGuide(_i).stage2UpdateCount;
	const int maxSumSize = guide.getHostGuide(_i).maxSumSize;
	const int bSumCount = guide.getHostGuide(_i).stage2BSumCount;
	
	// Enqueue stage_22a kernel
	
	const int partSumCountA = DIVCEIL(maxSumSize, maxSumSizeA);
	
	kernelHelper.setKernelArg(getAlias("lx_stage_22a"), KARGS_STAGE_22A_I, _i);
	kernelHelper.setKernelArg(
		getAlias("lx_stage_22a"), KARGS_STAGE_22A_MAX_SUM_SIZE, maxSumSize);
	
	kernelHelper.enqueueKernel(
		_queue, getAlias("lx_stage_22a"),
		KernelHelper::GroupCount(
			sumCount*partSumCountA, 
			_secondaryIndexSpaceSize, 
			optimizer.interped(
				idPrefix + LX_STAGE_22A_WG_PER_VECTOR, 
				optValues)),
		optimizer.interped(idPrefix + LX_STAGE_22A_WG_SIZE, optValues));
	
	// Enqueue stage_y2b kernel
	
	int partSumCountB = partSumCountA == 1 ? 
		0 : DIVCEIL(partSumCountA, maxSumSizeB);
	
	if(0 < partSumCountB && 0 < bSumCount) {
		kernelHelper.setKernelArg(
			getAlias("lx_stage_y2b"), KARGS_STAGE_Y2B_I, _i+1);
		kernelHelper.setKernelArg(
			getAlias("lx_stage_y2b"), KARGS_STAGE_Y2B_MAX_SUM_SIZE, maxSumSize);
		kernelHelper.setKernelArg(
			getAlias("lx_stage_y2b"), KARGS_STAGE_Y2B_GUIDE, stage22BGuide);
	}
		
	for(int sumStep = 1; 0 < partSumCountB && 0 < bSumCount; sumStep++) {
		
		kernelHelper.setKernelArg(
			getAlias("lx_stage_y2b"), KARGS_STAGE_Y2B_SUM_STEP, sumStep);
		
		kernelHelper.enqueueKernel(
			_queue, getAlias("lx_stage_y2b"),
			KernelHelper::GroupCount(
				bSumCount*partSumCountB, 
				_secondaryIndexSpaceSize, 
				optimizer.interped(
					idPrefix + LX_STAGE_Y2B_WG_PER_VECTOR, 
					optValues)),
			optimizer.interped(idPrefix + LX_STAGE_Y2B_WG_SIZE, optValues));
		
		partSumCountB = partSumCountB == 1 ?
			0 : DIVCEIL(partSumCountB, maxSumSizeB);
	}
}

void DeviceContextHelper::queueStage22b(
		CommandQueue& _queue,
		int           _secondaryIndexSpaceSize,
		int           _i) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"DeviceContextHelper::queueStage22b (level " << level << 
		"): Enqueueing stage22 part B..." << std::endl;
#endif
	
	checkStatus();
	
	const int idPrefix = getIDPrefix(level);
	
	// Enqueue stage_22c kernel
	
	kernelHelper.setKernelArg(getAlias("lx_stage_22c"), KARGS_STAGE_22C_I, _i);
	
	kernelHelper.enqueueKernel(
		_queue, getAlias("lx_stage_22c"),
		KernelHelper::GroupCount(
			guide.getHostGuide(_i).stage2UpdateCount, 
			_secondaryIndexSpaceSize, 
			optimizer.interped(
				idPrefix + LX_STAGE_22C_WG_PER_VECTOR, 
				optValues)),
		optimizer.interped(idPrefix + LX_STAGE_22C_WG_SIZE, optValues));
}

bool DeviceContextHelper::checkStatus(bool _throwException) {
	if(!argsAreSet) {
		if(_throwException) {
			std::cerr << errorLocation << 
			"DeviceContextHelper::checkStatus() (level " << level << 
			"): Initial kernel arguments " << "has not been set." << std::endl;
			
			throw UnknownError();
		}
		return false;
	}
	
	if(!isAllocated) {
		if(_throwException) {
			std::cerr << errorLocation <<
			"DeviceContextHelper::checkStatus() (level " << level << 
			"): Memory buffers " << "are not allocated." << std::endl;
			
			throw UnknownError();
		}
	}
	
	if(!chIsSet) {
		if(_throwException) {
			std::cerr << errorLocation <<
			"DeviceContextHelper::checkStatus() (level " << level << 
			"): Ch is not set. " << std::endl;
			
			throw UnknownError();
		}
	}
	
	return true;
}

std::string DeviceContextHelper::getAlias(const std::string &name) const {
	std::string ret = name;
	ret.replace(0, 3, level == level1 ? "l1_" : "l2_");
	return ret;
}

Optimizer DeviceContextHelper::createOptimizer(
    const cl::Device& _device,
    int               _n3,
    const PscrCLMode &_mode,
	LevelType         _level) {
	
	if(_level != level1 && _level != level2) {
		std::cerr << errorLocation <<
			"DeviceContextHelper::createOptimizer (level " << _level << 
			"): Invalid level." << std::endl;
		throw UnknownError();
	}
	
    Optimizer optimizer;

    //
    // Get the device specific parameters
    //

    int dLim = getDSize(_mode);
    //int varSize = getVarSize(_mode);

    DeviceInformation deviceHelper(_device);
    //long maxLocalMemSize = deviceHelper.getMaxLocalMemSize();
	const int wBSize = deviceHelper.getWBSize();
    const int minWGSize = deviceHelper.getMinWGSize();
    const int maxWGSize = deviceHelper.getMaxWGSize();
	const long maxLocalMemSize = deviceHelper.getMaxLocalMemSize();
	const int varSize = getVarSize(_mode);
	
    // Calculate the vector size
    int l3D;
    if(dLim*wBSize <= _n3)
        l3D = dLim;
    else
        l3D = 1;

    // Select opt. parameter id prefix
    int idPrefix = getIDPrefix(_level);

	// Select ipt. parameter name prefix
    std::string namePrefix = _level == level2 ? "l2_" : "l1_";

	// Add vector size opt. parameter
    optimizer.addParam(
        OptParam(
            idPrefix + LX_D,
            namePrefix + "d", "D",
            Pow2OptValueInterpreter(), l3D));
	
	// Add global memory access pattern opt. parameter, i.e., use local memory
	// and synchronization when accessing isolated global memory addresses
	optimizer.addParam(
		OptParam(
			idPrefix + LX_SHARED_ISOLATED_ACCESS,
			namePrefix + "shared_isolated_access", 
			"LX_SHARED_ISOLATED_ACCESS",
			BoolOptValueInterpreter(), 0));
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_NEWTON_DIV,
			namePrefix + "newton_div", 
			"NEWTON_DIV",
			BoolOptValueInterpreter(), 1));
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_VECTOR_LOAD_HELPER,
			namePrefix + "vector_load_helper", 
			"LX_VECTOR_LOAD_HELPER",
			BoolOptValueInterpreter(), 0));
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_MATRIX_LOAD_HELPER,
			namePrefix + "matrix_load_helper", 
			"LX_MATRIX_LOAD_HELPER",
			BoolOptValueInterpreter(), 0));
			
	// Calculate the work group size of stage12c-kernel
	int stage12CWGSize = minWGSize;
	while(stage12CWGSize < _n3 &&
		stage12CWGSize+wBSize <= maxWGSize &&
		varSize*(stage12CWGSize+wBSize) <= maxLocalMemSize)
	stage12CWGSize += wBSize;
	
	int stage12CWGPerVec = MAX(1, _n3 / stage12CWGSize);
	
	// Calculate the work group size of stage21-kernel
	int stage21WGSize = minWGSize;
	while(stage21WGSize < _n3 &&
		stage21WGSize+wBSize <= maxWGSize &&
		varSize*(stage21WGSize+wBSize) <= maxLocalMemSize)
	stage21WGSize += wBSize;
	
	int stage21WGPerVec = MAX(1, _n3 / stage21WGSize);
	
	// Calculate the work group size for all other scalar kernels
	int wGSize = minWGSize;
	while(wGSize < _n3 &&
			wGSize+wBSize <= maxWGSize)
		wGSize += wBSize;
	
	int wGPerVec = MAX(1, _n3 / wGSize);
	
	// Calculate the work group size for all vector kernels
	int dWGSize = minWGSize;
	while(l3D*dWGSize < _n3 &&
			dWGSize+wBSize <= maxWGSize)
		dWGSize += wBSize;
	
	int dWGPerVec = MAX(1, _n3 / (l3D * dWGSize));

	//
	// Opt. parameters for lx_stage_11
	//
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_11_WG_SIZE,
			namePrefix + "stage_11_wg_size", "LX_STAGE_11_WG_SIZE",
			AddOptValueInterpreter(wBSize), dWGSize));

	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_11_WG_PER_VECTOR,
			namePrefix + "stage_11_wg_per_vector",
			"LX_STAGE_11_WG_PER_VECTOR",
			IdenticalOptValueInterpreter(), dWGPerVec));
	
	//
	// Opt. parameters for lx_stage_12a
	//
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_12A_WG_SIZE,
			namePrefix + "stage_12a_wg_size", "LX_STAGE_12A_WG_SIZE",
			AddOptValueInterpreter(wBSize), dWGSize));

	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_12A_WG_PER_VECTOR,
			namePrefix + "stage_12a_wg_per_vector",
			"LX_STAGE_12A_WG_PER_VECTOR",
			IdenticalOptValueInterpreter(), dWGPerVec));
	
	//
	// Opt. parameters for lx_stage_12c
	//
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_12C_WG_SIZE,
			namePrefix + "stage_12c_wg_size", "LX_STAGE_12C_WG_SIZE",
			AddOptValueInterpreter(wBSize), stage12CWGSize));

	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_12C_WG_PER_VECTOR,
			namePrefix + "stage_12c_wg_per_vector",
			"LX_STAGE_12C_WG_PER_VECTOR",
			IdenticalOptValueInterpreter(), stage12CWGPerVec));

	//
	// Vector summation opt. parameters
	//
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_Y2A_MAX_SUM_SIZE_EXP,
			namePrefix + "stage_y2a_max_sum_size_exp", 
		   "LX_STAGE_Y2A_MAX_SUM_SIZE_EXP",
			IdenticalOptValueInterpreter(), 3));
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_Y2B_MAX_SUM_SIZE_EXP,
			namePrefix + "stage_y2b_max_sum_size_exp", 
		   "LX_STAGE_Y2B_MAX_SUM_SIZE_EXP",
			IdenticalOptValueInterpreter(), 3));
	
	//
	// Opt. parameters for lx_stage_y2b
	//
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_Y2B_WG_SIZE,
			namePrefix + "stage_y2b_wg_size", "LX_STAGE_Y2B_WG_SIZE",
			AddOptValueInterpreter(wBSize), dWGSize));

	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_Y2B_WG_PER_VECTOR,
			namePrefix + "stage_y2b_wg_per_vector",
			"LX_STAGE_Y2B_WG_PER_VECTOR",
			IdenticalOptValueInterpreter(), dWGPerVec));
	
	//
	// Opt. parameters for lx_stage_21
	//
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_21_WG_SIZE,
			namePrefix + "stage_12_wg_size", "LX_STAGE_21_WG_SIZE",
			AddOptValueInterpreter(wBSize), stage21WGSize));

	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_21_WG_PER_VECTOR,
			namePrefix + "stage_12_wg_per_vector",
			"LX_STAGE_21_WG_PER_VECTOR",
			IdenticalOptValueInterpreter(), stage21WGPerVec));
	
	//
	// Opt. parameters for lx_stage_22a
	//
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_22A_WG_SIZE,
			namePrefix + "stage_22a_wg_size", "LX_STAGE_22A_WG_SIZE",
			AddOptValueInterpreter(wBSize), dWGSize));

	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_22A_WG_PER_VECTOR,
			namePrefix + "stage_22a_wg_per_vector",
			"LX_STAGE_22A_WG_PER_VECTOR",
			IdenticalOptValueInterpreter(), dWGPerVec));
	
	//
	// Opt. parameters for lx_stage_22c
	//
	
	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_22C_WG_SIZE,
			namePrefix + "stage_22c_wg_size", "LX_STAGE_22C_WG_SIZE",
			AddOptValueInterpreter(wBSize), wGSize));

	optimizer.addParam(
		OptParam(
			idPrefix + LX_STAGE_22C_WG_PER_VECTOR,
			namePrefix + "stage_22c_wg_per_vector",
			"LX_STAGE_22C_WG_PER_VECTOR",
			IdenticalOptValueInterpreter(), wGPerVec));
	
    return optimizer;
}

bool DeviceContextHelper::checkOptParams(
		const cl::Device &_device,
		const OptValues  &_optValues,
		int               _lxLdf3,
		int               _n3,
		const PscrCLMode &_mode,
		LevelType         _level,
		std::string      *_err) {
	
	Optimizer optimizer = createOptimizer(_device, _n3, _mode, _level);

	const int idPrefix = getIDPrefix(_level);
	std::string namePrefix = _level == level2 ? "L2" : "L1";
	
	const int lxD = optimizer.interped(idPrefix + LX_D, _optValues);
	
	const int lxStageY2AMaxSumSizeExp = optimizer.interped(
		idPrefix + LX_STAGE_Y2A_MAX_SUM_SIZE_EXP, _optValues);
	const int lxStageY2BMaxSumSizeExp = optimizer.interped(
		idPrefix + LX_STAGE_Y2B_MAX_SUM_SIZE_EXP, _optValues);
	
	const int lxStage11WGSize = 
		optimizer.interped(idPrefix + LX_STAGE_11_WG_SIZE, _optValues);
	const int lxStage11WGPerVector = 
		optimizer.interped(idPrefix + LX_STAGE_11_WG_PER_VECTOR, _optValues);
	
	const int lxStage12AWGSize = 
		optimizer.interped(idPrefix + LX_STAGE_12A_WG_SIZE, _optValues);
	const int lxStage12AWGPerVector = 
		optimizer.interped(idPrefix + LX_STAGE_12A_WG_PER_VECTOR, _optValues);
	
	const int lxStage12CWGSize = 
		optimizer.interped(idPrefix + LX_STAGE_12C_WG_SIZE, _optValues);
	const int lxStage12CWGPerVector = 
		optimizer.interped(idPrefix + LX_STAGE_12C_WG_PER_VECTOR, _optValues);
	
	const int lxStage21WGSize = 
		optimizer.interped(idPrefix + LX_STAGE_21_WG_SIZE, _optValues);
	const int lxStage21WGPerVector = 
		optimizer.interped(idPrefix + LX_STAGE_21_WG_PER_VECTOR, _optValues);
	
	const int lxStage22AWGSize = 
		optimizer.interped(idPrefix + LX_STAGE_22A_WG_SIZE, _optValues);
	const int lxStage22AWGPerVector = 
		optimizer.interped(idPrefix + LX_STAGE_22A_WG_PER_VECTOR, _optValues);
	
	const int lxStage22CWGSize = 
		optimizer.interped(idPrefix + LX_STAGE_22C_WG_SIZE, _optValues);
	const int lxStage22CWGPerVector = 
		optimizer.interped(idPrefix + LX_STAGE_22C_WG_PER_VECTOR, _optValues);
	
	const int lxStageY2BWGSize = 
		optimizer.interped(idPrefix + LX_STAGE_Y2B_WG_SIZE, _optValues);
	const int lxStageY2BWGPerVector = 
		optimizer.interped(idPrefix + LX_STAGE_Y2B_WG_PER_VECTOR, _optValues);
	
	// TODO: Check _optValues
	
	DeviceInformation deviceHelper(_device);
	
	const long maxLocalMemSize = deviceHelper.getMaxLocalMemSize();
	const int minWGSize = deviceHelper.getMinWGSize();
	const int maxWGSize = deviceHelper.getMaxWGSize();
	const int dLim = getDSize(_mode);
	const size_t varSize = getVarSize(_mode);
	
	// 1 <= LX_D <= 1/2/4
	if(lxD < 1 || dLim < lxD) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + "_D.";
		return false;
	}
	
	// Work group size tests
		
	if(lxStage11WGSize < minWGSize || maxWGSize < lxStage11WGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_11_WG_SIZE.";
		return false;
	}
	if(lxStage12AWGSize < minWGSize || maxWGSize < lxStage12AWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_12A_WG_SIZE.";
		return false;
	}
	if(lxStage12CWGSize < minWGSize || maxWGSize < lxStage12CWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_12C_WG_SIZE.";
		return false;
	}

	if(lxStage21WGSize < minWGSize || maxWGSize < lxStage21WGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_21_WG_SIZE.";
		return false;
	}
	if(lxStage22AWGSize < minWGSize || maxWGSize < lxStage22AWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_22A_WG_SIZE.";
		return false;
	}
	if(lxStage22CWGSize < minWGSize || maxWGSize < lxStage22CWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_22C_WG_SIZE.";
		return false;
	}
	
	if(lxStageY2BWGSize < minWGSize || maxWGSize < lxStageY2BWGSize) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_Y2B_WG_SIZE.";
		return false;
	}
	
	// Work group per vector test
	
	if(lxStage11WGPerVector < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_11_WG_PER_VECTOR.";
		return false;
	}
	if(2*_lxLdf3 < lxStage11WGSize * lxStage11WGPerVector) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_11_WG_SIZE or PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_11_WG_PER_VECTOR.";
		return false;
	}
		
	if(lxStage12AWGPerVector < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_12A_WG_PER_VECTOR.";
		return false;
	}
	if(2*_lxLdf3 < lxStage12AWGSize * lxStage12AWGPerVector) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_12A_WG_SIZE or PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_12A_WG_PER_VECTOR.";
		return false;
	}
	
	if(lxStage12CWGPerVector < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_12C_WG_PER_VECTOR.";
		return false;
	}
	if(2*_lxLdf3 < lxStage12CWGSize * lxStage12CWGPerVector) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_12C_WG_SIZE or PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_12C_WG_PER_VECTOR.";
		return false;
	}
		
	if(lxStage21WGPerVector < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_21_WG_PER_VECTOR.";
		return false;
	}
	if(2*_lxLdf3 < lxStage21WGSize * lxStage21WGPerVector) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_21_WG_SIZE or PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_21_WG_PER_VECTOR.";
		return false;
	}
		
	if(lxStage22AWGPerVector < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_22A_WG_PER_VECTOR.";
		return false;
	}
	if(2*_lxLdf3 < lxStage22AWGSize * lxStage22AWGPerVector) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_22A_WG_SIZE or PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_22A_WG_PER_VECTOR.";
		return false;
	}
	
	if(lxStage22CWGPerVector < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_22C_WG_PER_VECTOR.";
		return false;
	}
	if(2*_lxLdf3 < lxStage22CWGSize * lxStage22CWGPerVector) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_22C_WG_SIZE or PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_22C_WG_PER_VECTOR.";
		return false;
	}
		
	if(lxStageY2BWGPerVector < 1) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_Y2B_WG_PER_VECTOR.";
		return false;
	}
	if(2*_lxLdf3 < lxStageY2BWGSize * lxStageY2BWGPerVector) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_Y2B_WG_SIZE or PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_Y2B_WG_PER_VECTOR.";
		return false;
	}
		
	// Max sum size tests
		
	if(lxStageY2AMaxSumSizeExp < 2 || 8 < lxStageY2AMaxSumSizeExp) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_Y2A_MAX_SUM_SIZE_EXP.";
		return false;
	}
	if(lxStageY2BMaxSumSizeExp < 2 || 8 < lxStageY2BMaxSumSizeExp) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix +
				"_STAGE_Y2B_MAX_SUM_SIZE_EXP.";
		return false;
	}
		
	// Local memory tests
	
	if(maxLocalMemSize < (int)(lxStage12CWGSize*varSize)) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_12C_WG_SIZE.";
		return false;
	}
		
	if(maxLocalMemSize < (int)(lxStage21WGSize*varSize)) {
		if(_err)
			*_err = "Invalid PSCRCL_OPT_PARAM_" + namePrefix + 
				"_STAGE_21_WG_SIZE.";
		return false;
	}
		
	if(_err)
		*_err = "";
	return true;
	
}