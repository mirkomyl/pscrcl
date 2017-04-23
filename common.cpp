/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <iostream>
#include "common.h"

using namespace pscrCL;

const std::string errorLocation = pscrCL::errorMsgBegin + " / common / ";
const std::string debugLocation = pscrCL::debugMsgBegin + " / common / ";

const std::pair<cl_int, const std::string> messages[] = {
	std::pair<cl_int, const std::string>(CL_SUCCESS, "Success"),
	std::pair<cl_int, const std::string>(CL_DEVICE_NOT_FOUND,
			"Device not found"),
	std::pair<cl_int, const std::string>(CL_DEVICE_NOT_AVAILABLE,
			"Device not available"),
	std::pair<cl_int, const std::string>(CL_COMPILER_NOT_AVAILABLE,
			"Compiler not available"),
	std::pair<cl_int, const std::string>(CL_MEM_OBJECT_ALLOCATION_FAILURE,
			"Memory object allocation error"),
	std::pair<cl_int, const std::string>(CL_OUT_OF_RESOURCES,
			"Out of resources"),
	std::pair<cl_int, const std::string>(CL_OUT_OF_HOST_MEMORY,
			"Out of host memory"),
	std::pair<cl_int, const std::string>(CL_PROFILING_INFO_NOT_AVAILABLE,
			"Profiling info not available"),
	std::pair<cl_int, const std::string>(CL_MEM_COPY_OVERLAP,
			"Memory copy overlap"),
	std::pair<cl_int, const std::string>(CL_IMAGE_FORMAT_MISMATCH,
			"Image format mismatch"),
	std::pair<cl_int, const std::string>(CL_IMAGE_FORMAT_NOT_SUPPORTED,
			"Image format not supported"),
	std::pair<cl_int, const std::string>(CL_BUILD_PROGRAM_FAILURE,
			"Program build failure"),
	std::pair<cl_int, const std::string>(CL_MAP_FAILURE, "Map failure"),
	std::pair<cl_int, const std::string>(CL_MISALIGNED_SUB_BUFFER_OFFSET,
			"Misaligned sub-buffer offset"),
	std::pair<cl_int, const std::string>(
			CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
			"Execution status error for events in wait list"),
	std::pair<cl_int, const std::string>(CL_INVALID_VALUE, "Invalid value"),
	std::pair<cl_int, const std::string>(CL_INVALID_DEVICE_TYPE,
			"Invalid device type"),
	std::pair<cl_int, const std::string>(CL_INVALID_PLATFORM,
			"Invalid platform"),
	std::pair<cl_int, const std::string>(CL_INVALID_DEVICE, "Invalid device"),
	std::pair<cl_int, const std::string>(CL_INVALID_CONTEXT,
			"Invalid context"),
	std::pair<cl_int, const std::string>(CL_INVALID_QUEUE_PROPERTIES,
			"Invalid command queue properties"),
	std::pair<cl_int, const std::string>(CL_INVALID_COMMAND_QUEUE,
			"Invalid command queue"),
	std::pair<cl_int, const std::string>(CL_INVALID_HOST_PTR,
			"Invalid host memory pointer"),
	std::pair<cl_int, const std::string>(CL_INVALID_MEM_OBJECT,
			"Invalid device memory object"),
	std::pair<cl_int, const std::string>(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
			"Invalid image format descriptor"),
	std::pair<cl_int, const std::string>(CL_INVALID_IMAGE_SIZE,
			"Invalid image size"),
	std::pair<cl_int, const std::string>(CL_INVALID_SAMPLER,
			"Invalid image sampler"),
	std::pair<cl_int, const std::string>(CL_INVALID_BINARY,
			"Invalid program binary"),
	std::pair<cl_int, const std::string>(CL_INVALID_BUILD_OPTIONS,
			"Invalid program build options"),
	std::pair<cl_int, const std::string>(CL_INVALID_PROGRAM,
			"Invalid program object"),
	std::pair<cl_int, const std::string>(CL_INVALID_PROGRAM_EXECUTABLE,
			"Invalid program executable"),
	std::pair<cl_int, const std::string>(CL_INVALID_KERNEL_NAME,
			"Invalid kernel name"),
	std::pair<cl_int, const std::string>(CL_INVALID_KERNEL_DEFINITION,
			"Invalid kernel definition"),
	std::pair<cl_int, const std::string>(CL_INVALID_KERNEL, "Invalid kernel"),
	std::pair<cl_int, const std::string>(CL_INVALID_ARG_INDEX,
			"Invalid kernel argument index"),
	std::pair<cl_int, const std::string>(CL_INVALID_ARG_VALUE,
			"Invalid kernel argument value"),
	std::pair<cl_int, const std::string>(CL_INVALID_ARG_SIZE,
			"Invalid kernel argument size"),
	std::pair<cl_int, const std::string>(CL_INVALID_KERNEL_ARGS,
			"Invalid kernel arguments"),
	std::pair<cl_int, const std::string>(CL_INVALID_WORK_DIMENSION,
			"Invalid work dimensions"),
	std::pair<cl_int, const std::string>(CL_INVALID_WORK_GROUP_SIZE,
			"Invalid work group size"),
	std::pair<cl_int, const std::string>(CL_INVALID_WORK_ITEM_SIZE,
			"Invalid work item size"),
	std::pair<cl_int, const std::string>(CL_INVALID_GLOBAL_OFFSET,
			"Invalid global offset"),
	std::pair<cl_int, const std::string>(CL_INVALID_EVENT_WAIT_LIST,
			"Invalid event wait list"),
	std::pair<cl_int, const std::string>(CL_INVALID_EVENT, "Invalid event"),
	std::pair<cl_int, const std::string>(CL_INVALID_OPERATION,
			"Invalid operation"),
	std::pair<cl_int, const std::string>(CL_INVALID_GL_OBJECT,
			"Invalid OpenGL object"),
	std::pair<cl_int, const std::string>(CL_INVALID_BUFFER_SIZE,
			"Invalid buffer size"),
	std::pair<cl_int, const std::string>(CL_INVALID_MIP_LEVEL,
			"Invalid MIP level"),
	std::pair<cl_int, const std::string>(CL_INVALID_GLOBAL_WORK_SIZE,
			"Invalid global work size"),
	std::pair<cl_int, const std::string>(CL_INVALID_PROPERTY,
			"Invalid property")
#ifdef CL_PLATFORM_NOT_FOUND_KHR
	,std::pair<cl_int, const std::string>(CL_PLATFORM_NOT_FOUND_KHR,
			"Platform not found KHR")
#endif
	};

std::size_t pscrCL::getVarSize(const PscrCLMode &mode) {
	if(mode.numComplex())
		return mode.precDouble() ? sizeof(std::complex<double>) :
			sizeof(std::complex<float>);
	else
		return mode.precDouble() ? sizeof(double) : sizeof(float);
}

int pscrCL::getDSize(const PscrCLMode &mode) {
	if(mode.numComplex())
		return mode.precDouble() ? 1 : 2;
	else
		return mode.precDouble() ? 2 : 4;
}

std::string pscrCL::CLErrorMessage(cl_int code) {
	std::string errorMessage = "Unknown error";

	for(size_t i = 0;
			i < sizeof(messages)/sizeof(std::pair<cl_int, const std::string>);
			i++) {
		if(messages[i].first == code) {
			errorMessage = messages[i].second;
			break;
		}
	}

	return "OpenCL error: " + errorMessage + " (Error code: " +
			toString(code) + ").";
}
