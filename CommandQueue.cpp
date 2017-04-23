/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <iostream>
#include "CL/cl.hpp"

#include "CommandQueue.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

CommandQueue::CommandQueue(const cl::CommandQueue& _queue) : 
	cl::CommandQueue(_queue) {
	cl_int err;

	// Check if the CL_QUEUE_PROFILING_ENABLE is set
	cl_command_queue_properties properties;
	err = _queue.getInfo(CL_QUEUE_PROPERTIES, &properties);

	if(err != CL_SUCCESS) {
		std::cerr <<
			errorLocation << "CommandQueue::CommandQueue: " \
			"Cannot query command queue properties. " <<
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	if(!(properties & CL_QUEUE_PROFILING_ENABLE))
		basicProfiling = true;
	else
		basicProfiling = false;
	
	launches = new std::pair<int, LauchContainer>();
	launches->first = 1;
	
#if DEBUG
	std::cout << debugLocation <<
			"CommandQueue::CommandQueue: CL_QUEUE_PROFILING_ENABLE enabled."
			<< std::endl;
#endif
}

CommandQueue::CommandQueue(const CommandQueue& _queue) : 
	cl::CommandQueue(_queue) {
		
	launches = _queue.launches;
	launches->first++;

	fullProfiling = _queue.fullProfiling;
	basicProfiling = _queue.basicProfiling;	
	
}

CommandQueue& CommandQueue::operator=(const CommandQueue& _queue) {
	if(this == &_queue)
		return *this;
	
	launches = _queue.launches;
	launches->first++;

	fullProfiling = _queue.fullProfiling;
	basicProfiling = _queue.basicProfiling;	
	
	return *this;
}

CommandQueue::~CommandQueue() {
	launches->first--;
	if(launches->first < 1)
		delete launches;
}


CommandQueue::CommandQueue(
		const cl::Context& _context,
		const cl::Device& _device,
		bool _profiling) : cl::CommandQueue(
				_context,
				_device,
				_profiling ? CL_QUEUE_PROFILING_ENABLE : 0, 0) {

	// FIXME: Somehow handle the cl::CommandQueue errors

	basicProfiling = _profiling;
	
	launches = new std::pair<int, LauchContainer>();
	launches->first = 1;
	
#if DEBUG
	std::cout << debugLocation <<
			"CommandQueue::CommandQueue: CL_QUEUE_PROFILING_ENABLE enabled."
			<< std::endl;
#endif

}

void CommandQueue::enableFullProfiling(){
	if(!basicProfiling) {
		std::cerr <<
			errorLocation << "CommandQueue::enableFullProfiling: " \
			"Command queue profiling is disabled." << std::endl;
		throw OpenCLError("CommandQueue::enableFullProfiling: " \
				"Command queue profiling is disabled.");
	}
	fullProfiling = true;
#if DEBUG
	std::cout << debugLocation <<
			"CommandQueue::enableFullProfiling: Full profiling enabled."
			<< std::endl;
#endif
}

void CommandQueue::disableFullProfiling(){
	fullProfiling = false;
#if DEBUG
	std::cout << debugLocation <<
			"CommandQueue::enableFullProfiling: Full profiling disabled."
			<< std::endl;
#endif
}

cl_int CommandQueue::enqueueNDRangeKernel(
		const std::string& _name,
		const cl::Kernel& _kernel,
		const cl::NDRange& _offset,
		const cl::NDRange& _global,
		const cl::NDRange& _local,
		const VECTOR_CLASS<cl::Event>* _events,
		cl::Event* _event,
		CLKernelLaunchInfo _info) {

	cl_int err;

	cl::Event tmp_event, *event;

	if(fullProfiling && _event == 0)
		event = &tmp_event;
	else
		event = _event;

	err = cl::CommandQueue::enqueueNDRangeKernel(
			_kernel, _offset, _global, _local, _events, event);

	if(fullProfiling && err == CL_SUCCESS) {
		// Generate some additional information
		std::string infoString = "Work items = (" +
				toString(_global[0]/_local[0]) + "*" +
				toString(_local[0]) + ", " +
				toString(_global[1]/_local[1]) + "*" +
				toString(_local[1]) + ", " +
				toString(_global[2]/_local[2]) + "*" +
				toString(_local[2]) + ")";

		if(_info != "")
			infoString += ", " + _info;
	
		launches->second.push_back(KernelLaunch(_name, *event, infoString));
	}

	return err;

}

cl_int CommandQueue::enqueueReadBuffer(
        const cl::Buffer& _buffer,
        cl_bool _blocking,
        size_t _offset,
        size_t _size,
        void* _ptr,
        const VECTOR_CLASS<cl::Event>* _events,
        cl::Event* _event,
        CLKernelLaunchInfo _info) {

	cl_int err;

	cl::Event tmp_event, *event;

	if(fullProfiling && _event == 0)
		event = &tmp_event;
	else
		event = _event;

	err = cl::CommandQueue::enqueueReadBuffer(_buffer, _blocking,
			_offset, _size, _ptr, _events, event);

	if(fullProfiling && err == CL_SUCCESS) {
		if(err != CL_SUCCESS) {
			std::cerr <<
				errorLocation << "CommandQueue::enqueueReadBuffer: " \
				"Cannot queue buffer read. " <<
				CLErrorMessage(err) << std::endl;
			throw OpenCLError(err);
		}

		launches->second.push_back(KernelLaunch("<read buffer>", *event, _info));
	}

	return err;
}

cl_int CommandQueue::enqueueWriteBuffer(
        const cl::Buffer& _buffer,
        cl_bool _blocking,
        size_t _offset,
        size_t _size,
        const void* _ptr,
        const VECTOR_CLASS<cl::Event>* _events,
        cl::Event* _event,
        CLKernelLaunchInfo _info) {

	cl_int err;

	cl::Event tmp_event, *event;

	if(fullProfiling && _event == 0)
		event = &tmp_event;
	else
		event = _event;

	err = cl::CommandQueue::enqueueWriteBuffer(_buffer, _blocking, _offset,
			_size, _ptr, _events, event);

	if(fullProfiling && err == CL_SUCCESS) {
		if(err != CL_SUCCESS) {
			std::cerr <<
				errorLocation << "CommandQueue::enqueueWriteBuffer: " \
				"Cannot queue buffer write. " <<
				CLErrorMessage(err) << std::endl;
			throw OpenCLError(err);
		}

		launches->second.push_back(KernelLaunch("<write buffer>", *event, _info));
	}

	return err;
}

const std::vector<KernelLaunch>& pscrCL::CommandQueue::getLaunches() const {
	return launches->second;
}

std::string CommandQueue::printLauchesInfo() const {

	std::string msg;

	if(!fullProfiling) {
		return "Cannot print kernel launch information. " \
		"Full profiling is disabled.\n";
	}

	msg += "Full kernel launch information:\n";

	double globalStartTime = 1.0/0.0;
	double globalStopTime = 0.0;
	double totalTime = 0.0;

	typedef std::pair<int, double> info;
	std::map<std::string, info> timeTable;

	std::vector<KernelLaunch>::const_iterator it;
	for(it = getLaunches().begin(); it != getLaunches().end(); it++) {
		totalTime += it->getTotalRuntime();
		globalStartTime = MIN(globalStartTime, it->getStartTime());
		globalStopTime = MAX(globalStopTime, it->getStopTime());
	}

	double realTotalTime = globalStopTime - globalStartTime;

	for(it = getLaunches().begin(); it != getLaunches().end(); it++) {
		double time = it->getTotalRuntime();
		const std::string& name = it->getName();

		std::map<std::string, info>::iterator kIt = timeTable.find(name);
		if(kIt == timeTable.end()) {
			timeTable.insert(std::pair<std::string, info>(name, info(1, time)));
		} else {
			kIt->second.first++;
			kIt->second.second += time;
		}

		msg += toString(time/realTotalTime) + " * 100%, " + toString(time) +
				"s, name: " + name + ", info: " + it->getInfo() + "\n";
	}

	msg += "Gathered kernel launch information:\n";

	std::map<std::string, info>::iterator it2;
	for(it2 = timeTable.begin(); it2 != timeTable.end(); it2++) {
		int count = it2->second.first;
		double time = it2->second.second;
		msg += toString(time/realTotalTime) + " * 100%, " + toString(time) +
				"s, " + toString(count) + " launches, name: " + it2->first +
				"\n";
	}

	msg += toString((realTotalTime-totalTime)/realTotalTime) + " * 100%, " +
			toString(realTotalTime-totalTime) +
			"s, x launches, name: <uncounted>\n";
				
	msg += "Total runtime: " + toString(realTotalTime) + "s\n";

	return msg;
}

void CommandQueue::clearLaunchInfo() {
	launches->second.clear();
}
