/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_DEVICE_HELPER
#define PSCRCL_DEVICE_HELPER

#include <CL/cl.hpp>

namespace pscrCL {

// A class which is used to probe useful information from the OpenCL
// device.
class DeviceInformation {
public:
	DeviceInformation(const cl::Device& device);

	int getMaxWGSize() const;
	int getMinWGSize() const;
	int getWBSize() const;
	cl_ulong getMaxLocalMemSize() const;
	bool isNvidiaDevice() const;
	bool hasDoublePrecisionSupport() const;

	static int getMaxWGSize(const cl::Device& device);
	static int getMinWGSize(const cl::Device& device);
	static int getWBSize(const cl::Device& device);
	static cl_ulong getMaxLocalMemSize(const cl::Device& device);
	static bool isNvidiaDevice(const cl::Device& device);
	static bool hasDoublePrecisionSupport(const cl::Device& device);

private:
	int maxWGSize;
	int minWGSize;
	int wBsize;
	cl_ulong maxLocalMemSize;
	bool nvidiaDevice;
	bool doublePrecisionSupport;
};

}

#endif
