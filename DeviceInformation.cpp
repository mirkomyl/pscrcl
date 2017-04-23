/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <iostream>
#include "common.h"
#include "DeviceInformation.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

DeviceInformation::DeviceInformation(
		const cl::Device& _device) {
	nvidiaDevice = isNvidiaDevice(_device);
	doublePrecisionSupport = hasDoublePrecisionSupport(_device);
	minWGSize = getMinWGSize(_device);
	maxWGSize = getMaxWGSize(_device);
	wBsize = getWBSize(_device);
	maxLocalMemSize = getMaxLocalMemSize(_device);
}

int DeviceInformation::getMaxWGSize() const {
	return maxWGSize;
}

int DeviceInformation::getMinWGSize() const {
	return minWGSize;
}

int DeviceInformation::getWBSize() const {
	return wBsize;
}

cl_ulong DeviceInformation::getMaxLocalMemSize() const {
	return maxLocalMemSize;
}

bool DeviceInformation::isNvidiaDevice() const {
	return nvidiaDevice;
}

bool DeviceInformation::hasDoublePrecisionSupport() const {
	return doublePrecisionSupport;
}

int DeviceInformation::getMaxWGSize(const cl::Device& _device) {
	cl_int err;

	std::vector<int> sizes;
	err = _device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &sizes);

	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
			"DeviceInformation::getMaxWGSize: Cannot query device info. " <<
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	return sizes[0];
}

int DeviceInformation::getMinWGSize(const cl::Device& _device) {
	if(isNvidiaDevice(_device))
		return 32;
	else
		return 64;
}

int DeviceInformation::getWBSize(const cl::Device& _device) {
	if(isNvidiaDevice(_device))
		return 32;
	else
		return 64;
}

cl_ulong DeviceInformation::getMaxLocalMemSize(const cl::Device& _device) {
	cl_int err;

	cl_ulong maxLocalMemSize;

	err = _device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &maxLocalMemSize);

	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
			"DeviceInformation::getMaxLocalMemSize: " \
			"Cannot query device info. " << CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	return maxLocalMemSize;
}

bool DeviceInformation::isNvidiaDevice(const cl::Device& _device) {
	cl_int err;

	std::string vendor;
	err = _device.getInfo(CL_DEVICE_VENDOR, &vendor);

	if(err != CL_SUCCESS) {
		std::cerr << errorLocation
				<< "DeviceInformation::isNvidiaDevice: " \
				"Cannot query device info. " << CLErrorMessage(err) <<
				std::endl;
		throw OpenCLError(err);
	}

	if(vendor.find("NVIDIA") != std::string::npos)
		return true;
	else
		return false;
}

bool DeviceInformation::hasDoublePrecisionSupport(const cl::Device& _device) {
	cl_int err;

	std::string extensions;
	err = _device.getInfo(CL_DEVICE_EXTENSIONS, &extensions);

	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
				"DeviceInformation::hasDoublePrecisionSupport: " \
				"Cannot query device extensions. " << CLErrorMessage(err) <<
				"." << std::endl;
		throw OpenCLError(err);
	}

	if(extensions.find("cl_khr_fp64") == std::string::npos &&
			extensions.find("cl_amd_fp64") == std::string::npos)
		return false;
	else
		return true;
}
