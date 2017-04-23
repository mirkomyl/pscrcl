/*
 *  Created on: Mar 18, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "L1OptimizerHelper.h"
#include "DeviceInformation.h"
#include "eigen/EigenContainer.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

L1OptimizerHelper::L1OptimizerHelper(
	cl::Context &_context,
	cl::Device  &_device,
	int          _n1,
	int          _n2,
	int          _n3,
	int          _l1Ldf2, 
	int          _l1Ldf3,
	int          _l2Ldf3,
	int          _l1MatLdf,
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
	
	n1 = _n1;
	n2 = _n2;
	n3 = _n3;
	l1Ldf2 = _l1Ldf2;
	l1Ldf3 = _l1Ldf3;
	l2Ldf3 = _l2Ldf3;
	
	const int wBSize = DeviceInformation::getWBSize(device);
	l1MatLdf = 0 < _l1MatLdf ? _l1MatLdf : DIVCEIL(n1, wBSize) * wBSize;
	l2MatLdf = 0 < _l2MatLdf ? _l2MatLdf : DIVCEIL(n2, wBSize) * wBSize;
	l3MatLdf = 0 < _l3MatLdf ? _l3MatLdf : DIVCEIL(n3, wBSize) * wBSize;
	
	upperBound = _upperBound;
	lowerBound = _lowerBound;
	
	gSize = tmpSize = 0;
}

void L1OptimizerHelper::initialize() {
	
	if(initialized) 
		return;
	
	cl_int err;
	
	// Allocate memory
	
	devCoefMatrix1 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		4*l1MatLdf*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L1OptimizerHelper::initialize: " <<
			"Cannot allocate devCoefMatrix1. " << 
			pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	devCoefMatrix2 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		4*l2MatLdf*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L1OptimizerHelper::initialize: " <<
			"Cannot allocate devCoefMatrix2. " << 
			pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	devCoefMatrix3 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		4*l3MatLdf*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L1OptimizerHelper::initialize: " <<
			"Cannot allocate devCoefMatrix3. " << 
			pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	devLambda1 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		K_4(n1)*(lowerBound-upperBound)*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L1OptimizerHelper::initialize: " <<
		"Cannot allocate devLambda1. " << 
		pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}

	devLambda2 = cl::Buffer(context, CL_MEM_READ_WRITE, 
		K_4(n2)*n2*getVarSize(mode), 0, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L1OptimizerHelper::initialize: " <<
		"Cannot allocate devLambda2. " << 
		pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	devF = cl::Buffer(context, CL_MEM_READ_WRITE, 
		 (lowerBound-upperBound)*l1Ldf2*l1Ldf3 *getVarSize(mode), 0, &err);
	
	if(err != CL_SUCCESS) {
		std::cerr << "(error) pscrCL / L1OptimizerHelper::initialize: " <<
		"Cannot allocate devF. " << pscrCL::CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	initialized = true;
}

void L1OptimizerHelper::recover() {
	std::vector<cl::Device> devices;
	devices.push_back(device);
	queue <<= new CommandQueue(context, devices[0], false);
}


bool L1OptimizerHelper::prepare(const OptValues& _values) {
	
	prepared = false;
	deviceContext = 0;
	devG = cl::Buffer();
	devTmp = cl::Buffer();
	
	if(!L1DeviceContext::checkOptParams(device, _values, l1Ldf3, l2Ldf3, n3, mode, 0))
		return false;
	
	try {
		const int wBSize = DeviceInformation::getWBSize(device);
		const int l2Ldf3 = DIVCEIL(n3, wBSize) * wBSize;
		
		if(!L1DeviceContext::checkOptParams(
			device, _values, l1Ldf3, l2Ldf3, n3, mode, 0))
			return false;

		Boundaries bounds1(n1);
		Boundaries bounds2(n2);
		
		std::vector<std::pair<int, int> > deviceBounds1;
		deviceBounds1.push_back(std::pair<int,int>(upperBound, lowerBound));
		
		std::vector<std::pair<int, int> > deviceBounds2;
		deviceBounds2.push_back(std::pair<int,int>(0, n2));
		
		MatrixContainer l1Matrix(0, 0, 0, 0, n1, l1MatLdf, mode, false); 
		MatrixContainer l2Matrix(0, 0, 0, 0, n2, l2MatLdf, mode, false); 
		
		EigenContainer eigenContainer1(l1Matrix, bounds1, deviceBounds1, true);
		EigenContainer eigenContainer2(l2Matrix, bounds2, deviceBounds2, true);
	
		optValues <<= new OptValues(_values);
		
		deviceContext <<= new L1DeviceContext(
			context,
			device,
			_values,
			bounds1,
			bounds2,
			eigenContainer1.getSection(0),
			eigenContainer2.getSection(0),
			deviceBounds1[0].first,
			deviceBounds1[0].second,
			l1MatLdf,
			l2MatLdf,
			l3MatLdf,
			l1Ldf2,
			l1Ldf3,
			n2,									// l2Ldf2
			l2Ldf3,
			n1,									// l3Ldf1
			n2, 								// l3Ldf2
			n1,
			n2,
			n3,
			mode);
		
		cl_int err;
		
		int newGSize = (lowerBound-upperBound) * n2 * l2Ldf3 *getVarSize(mode);
		
		if(gSize < newGSize) {
		
			devG = cl::Buffer(context, CL_MEM_READ_WRITE, newGSize, 0, &err);
			
			if(err != CL_SUCCESS) {
				std::cerr << "(error) pscrCL / L1OptimizerHelper::prepare: " <<
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
				std::cerr << "(error) pscrCL / L1OptimizerHelper::prepare: " <<
				"Cannot allocate devTmp. " << pscrCL::CLErrorMessage(err) << 
				std::endl;
				throw OpenCLError(err);
			}
			
			tmpSize = newTmpSize;
		}
			
		deviceContext->allocate(*queue);
		
		deviceContext->setArgs(
			devF, devG, devTmp, devCoefMatrix1, devCoefMatrix2, devCoefMatrix3, 
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

void L1OptimizerHelper::run() {
	std::complex<double> ch = 0.0;
	deviceContext->runSolver(*queue, &ch);
}
	

void L1OptimizerHelper::finalize() {
	cl_int err = queue->finish();

	if(err != CL_SUCCESS) {
		std::cerr <<
			errorLocation << "L1OptimizerHelper::finalize: " \
			"Cannot finish command queue. " <<
			CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}	
}

void L1OptimizerHelper::release() {

	devCoefMatrix1 = cl::Buffer();
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

