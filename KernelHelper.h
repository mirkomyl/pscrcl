/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_ABSTRACTDEVICECONTEXT
#define PSCRCL_ABSTRACTDEVICECONTEXT

#include <cstddef>
#include <iostream>
#include <CL/cl.hpp>

#include "common.h"
#include "optimizer/Optimizer.h"
#include "CommandQueue.h"

namespace pscrCL {

/*
 * An helper class which is used to manage the OpenCL sources and kernels.
 */
class KernelHelper {
public:
	// The number of work groups is stored into the following triplet
	class GroupCount {
	public:
		GroupCount(std::size_t _d1, std::size_t _d2 = 1, std::size_t _d3 = 1) {
			d1 = _d1;
			d2 = _d2;
			d3 = _d3;
		}
		std::size_t d1;
		std::size_t d2;
		std::size_t d3;
	};

	KernelHelper(
			cl::Context      &context,
			cl::Device       &device,
			const PscrCLMode &mode);

	// Return true if the OpenCL source is compiled
	bool isCompiled() const;

	// Compiles the given OpenCL source and prepares the found kernels
	void compileSource(
			const Optimizer& optimizer,
			const OptValues&            optValues,
			const std::string&          optionalArgs,
			const cl::Program::Sources& sources);

	template <typename T>
	void setKernelArg(const std::string& _kernel, cl_uint _id, T _value) {

		if(!isCompiled()) {
			std::cerr << errorMsgBegin <<
					" /  KernelHelper::setKernelArg: " \
					"Cannot set kernel argument. " \
					"The source code has not been compiled." << std::endl;
			throw UnknownError();
		}

#if FULL_DEBUG
		std::cout << debugMsgBegin <<
			" / KernelHelper::setKernelArg: " \
			"Setting kernel argument number " << _id <<
			" for kernel " << _kernel << " to " << toString(_value) <<
			"." << std::endl;
#endif

		cl_int err = getKernel(_kernel).setArg(_id, _value);

		if(err != CL_SUCCESS) {
			std::cerr << errorMsgBegin <<
				" / KernelHelper::setKernelArg: " \
				"Cannot set kernel argument number " << _id <<
				" for kernel " << _kernel << ". " << CLErrorMessage(err) <<
				"." << std::endl;
			throw OpenCLError(err);
		}
	}

	void enqueueKernel(
			CommandQueue& queue,
			const std::string& kernel,
			const GroupCount& groups,
			int localSize,
			const CLKernelLaunchInfo& info = NoLaunchInfo);
	
	void renameKernel(const std::string &oldName, const std::string &newName);

private:
	cl::Kernel& getKernel(const std::string kernel);

	bool compiled;
	cl::Context context;
	cl::Device device;
	PscrCLMode mode;
	std::map<std::string, cl::Kernel> kernels;
};

// Helper for the toString function. Converts the content of an
// KernelHelper::GroupCount into a string.
template <>
class ToStringHelper<KernelHelper::GroupCount> {
public:
	static std::string to(const KernelHelper::GroupCount& value) {
		return "(" + toString(value.d1) + ", " + toString(value.d2) +
				", " + toString(value.d3) + ")";
	}
};

}

#endif
