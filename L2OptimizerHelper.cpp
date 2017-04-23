/*
 *  Created on: Feb 25, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "L2OptimizerHelper.h"
#include "DeviceInformation.h"
#include "eigen/EigenContainer.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

L2OptimizerHelper::L2OptimizerHelper(
	cl::Context &_context, 
	cl::Device  &_device,
	int          _n2,
	int          _n3,
	int          _l2Ldf2, 
	int          _l2Ldf3,
	int          _l2MatLdf,
	int          _l3MatLdf,
	int          _upperBound,
	int          _lowerBound,
	PscrCLMode   _mode) : 
	context(_context), 
	device(_device),
	mode(_mode) {

	initialized = false;
	prepared = false;
	
	queue <<= new CommandQueue(context, device, false);
	
	deviceContext = 0;
	optValues = 0;
	
	n2 = _n2;
	n3 = _n3;
	l2Ldf2 = _l2Ldf2;
	l2Ldf3 = _l2Ldf3;
	
	const int wBSize = DeviceInformation::getWBSize(device);
	l2MatLdf = 0 < _l2MatLdf ? _l2MatLdf : DIVCEIL(n2, wBSize) * wBSize;
	l3MatLdf = 0 < _l3MatLdf ? _l3MatLdf : DIVCEIL(n3, wBSize) * wBSize;
	
	upperBound = _upperBound;
	lowerBound = _lowerBound;
	
	gSize = 0;
	tmpSize = 0;
}

void L2OptimizerHelper::initialize() {

	if(initialized) 
		return;
	
	cl_int err;
	
	// Allocate memory
	devCoefMatrix2 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		4*l2MatLdf*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L2OptimizerHelper::initialize: " <<
			"Cannot allocate devCoefMatrix2. " << 
			pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	devCoefMatrix3 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		4*l3MatLdf*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L2OptimizerHelper::initialize: " <<
			"Cannot allocate devCoefMatrix3. " << 
			pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	devLambda1 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L2OptimizerHelper::initialize: " <<
		"Cannot allocate devLambda1. " << 
		pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	devLambda2 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		K_4(n2)*(lowerBound-upperBound)*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L2OptimizerHelper::initialize: " <<
		"Cannot allocate devLambda2. " << 
		pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	devF = cl::Buffer(context, CL_MEM_READ_WRITE, 
		 (lowerBound-upperBound)* l2Ldf3 *getVarSize(mode), 0, &err);
	
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L2OptimizerHelper::initialize: " <<
		"Cannot allocate devF. " << pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	initialized = true;
}

bool L2OptimizerHelper::prepare(const OptValues& _values) {
	
	prepared = false;
	deviceContext = 0;
	
	if(!L2DeviceContext::checkOptParams(device, _values, l2Ldf3, n3, mode, 0))
		return false;

	try { 
	
		Boundaries bounds(n2);
		std::vector<std::pair<int, int> > deviceBounds;
		deviceBounds.push_back(std::pair<int,int>(upperBound, lowerBound));
		
		MatrixContainer l2Matrix(0, 0, 0, 0, n2, l2MatLdf, mode, false); 
		
		EigenContainer eigenContainer(l2Matrix, bounds, deviceBounds, true);
	
		optValues <<= new OptValues(_values);
		deviceContext <<= new L2DeviceContext(
			context,
			device,
			*optValues,
			bounds,
			eigenContainer.getSection(0),
			upperBound,
			lowerBound,
			0,
			l2MatLdf,
			l3MatLdf,
			l2Ldf2,
			l2Ldf3,
			1,						// l3Ldf1
			lowerBound-upperBound,	// l3Ldf2
			n2,
			n3,
			false,
			mode);
			
		cl_int err;
	
		int newGSize = (lowerBound-upperBound) * 
			deviceContext->getL3DeviceContext().getL3Ldf3()*getVarSize(mode);
		
		if(gSize < newGSize) {
			
			devG = cl::Buffer(context, CL_MEM_READ_WRITE, newGSize, 0, &err);
			
			if(err != CL_SUCCESS) {
				std::cerr << "(error) pscrCL / L2OptimizerHelper::prepare: " <<
				"Cannot allocate devG. " << pscrCL::CLErrorMessage(err) << 
				std::endl;
				throw OpenCLError(err);
			}
			
			gSize = newGSize;
		}
		
		int newTmpSize = deviceContext->getRequiredTmpSizePerSystem();

		if(tmpSize < newTmpSize) {
		
			devTmp = cl::Buffer(
				context, CL_MEM_READ_WRITE, newTmpSize, 0, &err);
			
			if(err != CL_SUCCESS) {
				std::cerr << "(error) pscrCL / L2OptimizerHelper::prepare: " <<
				"Cannot allocate devTmp. " << pscrCL::CLErrorMessage(err) << 
				std::endl;
				throw OpenCLError(err);
			}
			
			tmpSize = newTmpSize;
		}
		
		deviceContext->allocate(*queue);
		
		deviceContext->setArgs(
			devF, devG, devTmp, devCoefMatrix2, devCoefMatrix3, 
			devLambda1, devLambda2);
	} catch(...) {
		devG = cl::Buffer();
		devTmp = cl::Buffer();
		gSize = tmpSize = 0;
		deviceContext = 0;
		recover();
		return false;
	}
	
	prepared = true;
	
	return true;
}

void L2OptimizerHelper::recover() {
	std::vector<cl::Device> devices;
	context.getInfo(CL_CONTEXT_DEVICES, &devices);
	queue <<= new CommandQueue(context, devices[0], false);
}

void L2OptimizerHelper::run() {
	std::complex<double> ch = 0.0;
	deviceContext->runSolver(*queue, 0, 1, &ch);
}
	

void L2OptimizerHelper::finalize() {
	cl_int err = queue->finish();

	if(err != CL_SUCCESS) {
		std::cerr <<
			errorLocation << "L1OptimizerHelper::finalize: " \
			"Cannot finish command queue. " <<
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}	
}

void L2OptimizerHelper::release() {

	devCoefMatrix2 = cl::Buffer();
	devCoefMatrix3 = cl::Buffer();
	devLambda1 = cl::Buffer();
	devLambda2 = cl::Buffer();
	devF = cl::Buffer();
	devG = cl::Buffer();
	devTmp = cl::Buffer();
	gSize = tmpSize = 0;

	deviceContext = 0;;
	optValues = 0;
	
	initialized = false;
	prepared = false;
}

