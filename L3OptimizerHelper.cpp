/*
 *  Created on: Feb 10, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "L3OptimizerHelper.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

L3OptimizerHelper::L3OptimizerHelper(
	cl::Context &_context, 
	cl::Device  &_device,
	int          _n3, 
	int          _l3Ldf1, 
	int          _l3Ldf2, 
	int          _l3MatLdf,
	int          _count1,
	int          _count2,
	PscrCLMode   _mode) : 
	context(_context),
	device(_device),
	mode(_mode) {

	initialized = false;
	prepared = false;
	
	queue <<= new CommandQueue(context, device, false);
	
	deviceContext = 0;
	optValues = 0;
	
	n3 = _n3;
	l3Ldf1 = _l3Ldf1;
	l3Ldf2 = _l3Ldf2;
	l3MatLdf = _l3MatLdf;
	count1 = _count1;
	count2 = _count2;
}

void L3OptimizerHelper::initialize() {

	if(initialized) 
		return;
	
	cl_int err;
	
	// Allocate memory
	devCoefMatrix = cl::Buffer(context, CL_MEM_READ_WRITE, 
		4*l3MatLdf*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L3OptimizerHelper::initialize: " <<
			"Cannot allocate devCoefMatrix. " << 
			pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	devLambda1 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		count1*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L3OptimizerHelper::initialize: " <<
		"Cannot allocate devLambda1. " << 
		pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	devLambda2 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		count2*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L3OptimizerHelper::initialize: " <<
		"Cannot allocate devLambda2. " << 
		pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	initialized = true;
}

bool L3OptimizerHelper::prepare(const OptValues& _values) {
	
	prepared = false;
	deviceContext = 0;
	devTmp = cl::Buffer();
	
	if(!L3DeviceContext::checkOptValues(device, _values, n3, mode, 0))
		return false;

	try {
	
		optValues <<= new OptValues(_values);
		deviceContext <<= new L3DeviceContext(
			context, device, _values, n3, l3Ldf1, l3Ldf2, l3MatLdf, false, mode);
		
		cl_int err;
		
		devF = cl::Buffer(context, CL_MEM_READ_WRITE, 
			l3Ldf1*l3Ldf2*deviceContext->getL3Ldf3()*getVarSize(mode), 0, &err);
		
		if(err != CL_SUCCESS) {
			std::cerr << "(error) pscrCL / L3OptimizerHelper::prepare: " <<
			"Cannot allocate devF. " << pscrCL::CLErrorMessage(err) << 
			std::endl;
			throw OpenCLError(err);
		}

		devTmp = cl::Buffer(context, CL_MEM_READ_WRITE, 
			l3Ldf1*l3Ldf2*deviceContext->getRequiredTmpSizePerSystem(), 0, &err);
		
		if(err != CL_SUCCESS) {
			std::cerr << "(error) pscrCL / L3OptimizerHelper::prepare: " <<
			"Cannot allocate devTmp. " << pscrCL::CLErrorMessage(err) << 
			std::endl;
			throw OpenCLError(err);
		}
		
		deviceContext->setArgs(
			devF, devTmp, devLambda1, devLambda2, devCoefMatrix);
		
	} catch(...) {
		deviceContext = 0;
		devTmp = cl::Buffer();
		recover();
		return false;
	}
	
	prepared = true;
	
	return true;
}

void L3OptimizerHelper::recover() {
	std::vector<cl::Device> devices;
	context.getInfo(CL_CONTEXT_DEVICES, &devices);
	queue <<= new CommandQueue(context, devices[0], false);
}

void L3OptimizerHelper::run() {
	std::complex<double> ch = 0.0;
	deviceContext->run(*queue, 0, 0, count1, count2, &ch);
}
	

void L3OptimizerHelper::finalize() {
	cl_int err = queue->finish();

	if(err != CL_SUCCESS) {
		std::cerr <<
			errorLocation << "L1OptimizerHelper::finalize: " \
			"Cannot finish command queue. " <<
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}	
}

void L3OptimizerHelper::release() {

	devCoefMatrix = cl::Buffer();
	devLambda1 = cl::Buffer();
	devLambda2 = cl::Buffer();
	devF = cl::Buffer();
	devTmp = cl::Buffer();
	
	deviceContext = 0;
	optValues = 0;
	
	initialized = false;
	prepared = false;
}

