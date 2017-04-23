#include <iostream>
#include "common.h"
#include "KernelLaunch.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

KernelLaunch::KernelLaunch(
		const std::string& _name,
		const cl::Event& _event,
		CLKernelLaunchInfo _info) {
	name = _name;
	event = _event;
	info = _info;
}

const std::string& KernelLaunch::getName() const {
	return name;
}

const CLKernelLaunchInfo& KernelLaunch::getInfo() const {
	return info;
}

cl_ulong pscrCL::KernelLaunch::getTime(cl_profiling_info _paramName) const {
	cl_int err;

	// Check the kernel launch execution status.
	cl_int status;
	err = event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status);

	if(err != CL_SUCCESS) {
		std::cerr <<
			errorLocation << "KernelLaunch::getInfo: KernelLaunch: "\
			"Cannot get event status. " <<
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	if(status != CL_COMPLETE) {
		std::cerr <<
			errorLocation << "KernelLaunch::getInfo: Wrong status. " <<
				"Status: " << status << std::endl;
		throw OpenCLError(err);
	}

	// Read the actual timing informations
	cl_ulong time;
	err = event.getProfilingInfo(_paramName, &time);

	if(err != CL_SUCCESS) {
		std::cerr <<
			errorLocation << "KernelLaunch::getInfo: " \
			"Cannot get profiling time. " <<
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	return time;
}

double  KernelLaunch::getStartTime() const {
	return getTime(CL_PROFILING_COMMAND_START)*1.0e-9;
}

double  KernelLaunch::getStopTime() const {
	return getTime(CL_PROFILING_COMMAND_END)*1.0e-9;
}

double KernelLaunch::getTotalRuntime() const {
	return (getStopTime()-getStartTime());
}