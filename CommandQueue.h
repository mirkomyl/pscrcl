/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_COMMANDQUEUE
#define PSCRCL_COMMANDQUEUE

#include <map>
#include <CL/cl.hpp>

#include "common.h"
#include "CArray.h"
#include "KernelLaunch.h"

namespace pscrCL {

// A modified OpenCL command queue. Enables easier profiling.
class CommandQueue : public cl::CommandQueue {
public:
	typedef std::vector<KernelLaunch> LauchContainer;

	CommandQueue(const cl::CommandQueue &queue);
    CommandQueue(const CommandQueue &queue);
    CommandQueue& operator=(const CommandQueue &queue);
	~CommandQueue();
	
	// Creates a new modified OpenCL command queue. If profiling is true,
	// then CL_QUEUE_PROFILING_ENABLE flag is enabled.
	CommandQueue(
			const cl::Context& context,
			const cl::Device& device,
			bool profiling);

	// Enables more detailed profiling using KernelLaunch object
	void enableFullProfiling();

	// Disables the more detailed profiling
	void disableFullProfiling();

	// Overloaded
	cl_int enqueueNDRangeKernel(
			const std::string& name,
			const cl::Kernel& kernel,
			const cl::NDRange& offset,
			const cl::NDRange& global,
			const cl::NDRange& local = cl::NullRange,
			const VECTOR_CLASS<cl::Event>* events = 0,
			cl::Event* event = 0,
			CLKernelLaunchInfo info = NoLaunchInfo);

	// Overloaded
	cl_int enqueueReadBuffer(
	        const cl::Buffer& buffer,
	        cl_bool blocking,
	        size_t offset,
	        size_t size,
	        void* ptr,
	        const VECTOR_CLASS<cl::Event>* events = 0,
	        cl::Event* event = 0,
	        CLKernelLaunchInfo info = NoLaunchInfo);

	// Overloaded
	cl_int enqueueWriteBuffer(
	        const cl::Buffer& buffer,
	        cl_bool blocking,
	        size_t offset,
	        size_t size,
	        const void* ptr,
	        const VECTOR_CLASS<cl::Event>* events = 0,
	        cl::Event* event = 0,
	        CLKernelLaunchInfo info = NoLaunchInfo);

	// Returns a container containing the kernel launch information
	const LauchContainer& getLaunches() const;

	// Outputs some information
	std::string printLauchesInfo() const;
	
	void clearLaunchInfo();

private:
	std::pair<int, LauchContainer> *launches;

	bool fullProfiling;
	bool basicProfiling;
};

}


#endif /* PSCRCL_COMMANDQUEUE */
