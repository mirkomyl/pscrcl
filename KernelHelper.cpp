/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <fstream>
#include <streambuf>

#include "common.h"
#include "DeviceInformation.h"
#include "KernelHelper.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

KernelHelper::KernelHelper(
		cl::Context      &_context,
		cl::Device       &_device,
		const PscrCLMode &_mode) :
		context(_context), device(_device), mode(_mode) {

	compiled = false;

	// Pre-initialization checks

	if((mode.precDouble()) &&
			!DeviceInformation::hasDoublePrecisionSupport(device)) {
		std::cerr << errorLocation <<
				"KernelHelper::KernelHelper: OpenCL device does not " \
				"have a double precision support." << std::endl;
		throw OpenCLError();
	}

}

bool KernelHelper::isCompiled() const {
	return compiled;
}

void KernelHelper::compileSource(
		const Optimizer&            _optimizer,
		const OptValues&            _optValues,
		const std::string&          _optionalArgs,
		const cl::Program::Sources& _sources) {

#if FULL_DEBUG
	std::cout << debugLocation <<
			"KernelHelper::compileSource: Compiling cl-source files..." <<
			std::endl;

	Timer timer;
	timer.begin();
#endif

	cl_int err;

	// Creates a new opencl program-object
	cl::Program program = cl::Program(context, _sources, &err);
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
				"Cannot create program object." << std::endl;
		throw OpenCLError(err);
	}

	// Builds opencl program
	std::vector<cl::Device> tmpDevices;
	tmpDevices.push_back(device);

	std::string compilerArgs =
			"-D DOUBLE=" + toString(mode.precDouble() ? 1 : 0) + " " +
			"-D COMPLEX=" + toString(mode.numComplex() ? 1 : 0) + " " +
			"-cl-std=CL1.2 " +
			(mode.precFloat() ? "-cl-single-precision-constant " : "") +
			_optionalArgs +
			_optimizer.getCompilerArgs(_optValues);

#if FULL_DEBUG
	std::cout << debugLocation <<
		"KernelHelper::compileSource: Compiler arguments: " << 
		compilerArgs << std::endl;
#endif

	err = program.build(tmpDevices, compilerArgs.c_str());

	if(err == -9999)
		std::cout << toString(compilerArgs) << std::endl;
	
#if !FULL_DEBUG
	if(err != CL_SUCCESS) {
#endif
		std::string log;
		program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &log);
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"KernelHelper::compileSource: OpenCL compiler output:" << 
				std::endl << log << std::endl << 
				errorLocation << "Cannot build program-object. " << 
				CLErrorMessage(err) << std::endl;
			throw OpenCLError(err);
		} else {
			std::cout << debugLocation <<
				"KernelHelper::compileSource: OpenCL compiler output:" << 
				std::endl << log;
		}
#if !FULL_DEBUG
	}
#endif

	std::vector<cl::Kernel> kernelVector;
	err = program.createKernels(&kernelVector);

	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
			"KernelHelper::compileSource: Cannot create kernels. " << 
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	std::vector<cl::Kernel>::const_iterator it;
	for(it = kernelVector.begin(); it != kernelVector.end(); it++) {
		std::string kernelName = "<unknown>";
		err = it->getInfo(CL_KERNEL_FUNCTION_NAME, &kernelName);


#if FULL_DEBUG
		std::cout << debugLocation <<
			"KernelHelper::compileSource: Adding kernel " << kernelName << 
			"." << std::endl;
#endif

		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"KernelHelper::compileSource: Cannot get kernel name. " << 
				CLErrorMessage(err) << std::endl;
			throw OpenCLError(err);
		}

		kernels.insert(std::pair<std::string, cl::Kernel>(kernelName, *it));
	}

#if FULL_DEBUG
	timer.end();

	std::cout << debugLocation <<
		"KernelHelper::compileSource: Compile time: " << timer.getTime() << 
		"s." << std::endl;
#endif

	compiled = true;

}

void KernelHelper::enqueueKernel(
		CommandQueue&             _queue,
		const std::string&        _kernel,
		const GroupCount&         _groups,
		int                       _localSize,
		const CLKernelLaunchInfo& _info) {

	if(!isCompiled()) {
		std::cerr << errorLocation <<
			"KernelHelper::enqueueKernel: Cannot queue kernel. The source " <<
			"code has not been compiled." << std::endl;
		throw UnknownError();
	}

#if FULL_DEBUG
		std::cout << debugLocation <<
			"KernelHelper::enqueueKernel: Queuing kernel " << _kernel << ", " <<
			"groups: " << toString(_groups) << ", " <<
			"work group size: " << _localSize;
			if(_info != "")
				std::cout << ", info = " << _info;
			std::cout << "." << std::endl;
#endif

	cl_int err;
	err = _queue.enqueueNDRangeKernel(
			_kernel,
			getKernel(_kernel),
			cl::NDRange(0,0,0),
			cl::NDRange(_groups.d1*_localSize, _groups.d2, _groups.d3),
			cl::NDRange(_localSize, 1, 1),
			0, 0, _info);

	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
			"KernelHelper::enqueueKernel: Cannot queue " << _kernel << ". " << 
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

#if FULL_DEBUG
	std::cout << debugLocation <<
		"KernelHelper::enqueueKernel: Finishing..." << std::endl;
	err = _queue.finish();
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
			"KernelHelper::enqueueKernel: Cannot finish " << _kernel << ". " <<
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
#endif
			
}

void KernelHelper::renameKernel(
	const std::string &_oldName, 
	const std::string &_newName) {

	if(!isCompiled()) {
		std::cerr << errorLocation <<
			"KernelHelper::renameKernel: Cannot find " << _oldName << 
			". The source code has not been compiled." << std::endl;
		throw UnknownError();
	}

	std::map<std::string, cl::Kernel>::iterator kernelIt = 
		kernels.find(_oldName);

	if(kernelIt == kernels.end()) {
		std::cerr << errorMsgBegin <<
			" / KernelHelper::renameKernel: Cannot find kernel " << _oldName << 
			". Invalid kernel name." << std::endl;
		throw UnknownError();
	}

#if FULL_DEBUG
	std::cout << debugLocation <<
		"KernelHelper::renameKernel: Renaming kernel " <<
		_oldName << " ==> " << _newName << "." << std::endl;
#endif
	
	cl::Kernel kernel = kernelIt->second;
	kernels.erase(kernelIt);
	kernels.insert(std::pair<std::string, cl::Kernel>(_newName, kernel));
}

cl::Kernel& KernelHelper::getKernel(const std::string _kernel) {

	if(!isCompiled()) {
		std::cerr << errorLocation <<
			"KernelHelper::getKernel: Cannot find " << _kernel << 
			". The source code has not been compiled." << std::endl;
		throw UnknownError();
	}

	std::map<std::string, cl::Kernel>::iterator kernelIt = kernels.find(_kernel);

	if(kernelIt == kernels.end()) {
		std::cerr << errorMsgBegin <<
			" / KernelHelper::getKernel: Cannot find kernel " << _kernel << 
			". Invalid kernel name." << std::endl;
		throw UnknownError();
	}

	return kernelIt->second;
}
