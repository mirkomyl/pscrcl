/*
 *  Created on: Jan 13, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_KERNEL_LAUNCH
#define PSCRCL_KERNEL_LAUNCH

#include <map>
#include <CL/cl.hpp>

namespace pscrCL {

typedef std::string CLKernelLaunchInfo;
static const CLKernelLaunchInfo NoLaunchInfo = "";

// The timing and debugging information from each OpenCL kernel launch is
// stored into a KernelLaunch object. Each KernelLaunch object contains the
// name of the kernel, OpenCL event accosted with the launch and an information
// field.
class KernelLaunch {
public:
	KernelLaunch(
			const std::string& name,
			const cl::Event& event,
			CLKernelLaunchInfo info);

	const std::string& getName() const;
	const CLKernelLaunchInfo& getInfo() const;

	// Returns the time when the kernel launch began its execution in
	// seconds.  Should only be called once the kernel execution is completed.
	double getStartTime() const;

	// Returns the time when the kernel laucnh finished its execution in
	// seconds. Should only be called once the kernel execution is completed.
	double getStopTime() const;

	// Returns the total execution time of the kernel launch in seconds. Should
	// only be called onw the kernel execution is completed.
	double getTotalRuntime() const;

private:
	cl_ulong getTime(cl_profiling_info paramName) const;

	std::string name;
	cl::Event event;
	CLKernelLaunchInfo info;
};

}

#endif
