/*
 *  Created on: Feb 24, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "DeviceInformation.h"
#include "L1SolverContext.h"
#include "L1OptimizerHelper.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

L1SolverContext::L1SolverContext(
	std::vector< cl::Context > &_contexts, 
	std::vector<cl::Device> &_devices,
	const std::vector<OptValues> &_optValues,
	const void *_a1Diag, 
	const void *_a1Offdiag, 
	const void *_m1Diag, 
	const void *_m1Offdiag, 
	const void *_a2Diag, 
	const void *_a2Offdiag, 
	const void *_m2Diag, 
	const void *_m2Offdiag, 
	const void *_a3Diag, 
	const void *_a3Offdiag, 
	const void *_m3Diag, 
	const void *_m3Offdiag,
	int _n1,
	int _n2, 
	int _n3, 
	int _ldf2,
	int _ldf3, 
	const PscrCLMode& _mode) : mode(_mode) {
	
	// TODO: Implement support for multible GPUs
	if(_contexts.size() != 1) {
		std::cerr << errorLocation <<
			"pscrCL::L2Solver::L2Solver: Multi-GPU support is not yet implemented." << 
			std::endl;
		throw InvalidArgs();
	}
	
	// TODO: Context vs Devices test
	
	if(_contexts.size() != _optValues.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L1SolverContext::L1SolverContext: The number of cl::contexts and pscrCL::OptValues " <<
			"does not match." << std::endl;
		throw InvalidArgs();
	}

	const int wBsize = DeviceInformation::getWBSize(_devices.at(0));
	
	l1Matrix = MatrixContainer(
		_a1Diag,
		_a1Offdiag,
		_m1Diag,
		_m1Offdiag,
		_n1,
		DIVCEIL(_n1, wBsize) * wBsize,
		_mode,
		false);
	
	l2Matrix = MatrixContainer(
		_a2Diag,
		_a2Offdiag,
		_m2Diag,
		_m2Offdiag,
		_n2,
		DIVCEIL(_n2, wBsize) * wBsize,
		_mode,
		false);
	
	l3Matrix = MatrixContainer(
		_a3Diag,
		_a3Offdiag,
		_m3Diag,
		_m3Offdiag,
		_n3,
		DIVCEIL(_n3, wBsize) * wBsize,
		_mode,
		true);
	
	ldf2 = _ldf2;
	ldf3 = _ldf3;
	
	Boundaries bounds1(_n1);
	Boundaries bounds2(_n2);
	
#if FULL_DEBUG
	std::cout << debugLocation <<
		"pscrCL::L1SolverContext:L1SolverContext: Solving level 1 eigenproblems..." << std::endl;
#endif
	
	deviceBounds.push_back(std::pair<int,int>(0, _n1));
	eigenContainer1 = EigenContainer(l1Matrix, bounds1, deviceBounds);
	
#if FULL_DEBUG
	std::cout << debugLocation <<
		"pscrCL::L1SolverContext:L1SolverContext: Solving level 2 eigenproblems..." << std::endl;
#endif
	
	std::vector<std::pair<int, int> > deviceBounds2;
	deviceBounds2.push_back(std::pair<int,int>(0, _n2));
	eigenContainer2 = EigenContainer(l2Matrix, bounds2, deviceBounds2);
	
	for(int i = 0; i < (int) deviceBounds.size(); i++) {
		
#if DEBUG
		std::cout << debugLocation << 
			"pscrCL::L1SolverContext::L1SolverContext: Adding a pscrCL::L1DeviceContext object for " << 
			"bounds = [" << deviceBounds.at(i).first << "," <<
			deviceBounds.at(i).second << "]..." << std::endl;
#endif
	
		const int wBSize = DeviceInformation::getWBSize(_devices.at(i));
		
		deviceContexts.push_back(L1DeviceContext(
			_contexts.at(i),
			_devices.at(i),
			_optValues.at(i),
			bounds1,
			bounds2,
			eigenContainer1.getSection(i),
			eigenContainer2.getSection(0),
			deviceBounds.at(i).first,
			deviceBounds.at(i).second,
			l1Matrix.getLdf(),
			l2Matrix.getLdf(),
			l3Matrix.getLdf(),
			ldf2,								// l1Ldf2
			ldf3,								// l1Ldf3
			_n2,									// l2Ldf2
			DIVCEIL(_n3, wBSize) * wBSize,		// l2Ldf3
			_n1,									// l3Ldf1
			_n2, 								// l3Ldf2
			_n1,
			_n2,
			_n3,
			mode));
	}
	
	isAllocated = false;
	tmpIsAllocated = false;
	
#if DEBUG
	std::cout << debugLocation << 
		"pscrCL::L1SolverContext::L1SolverContext: pscrCL::L1SolverContext object initialized." << std::endl;
#endif
}

void L1SolverContext::allocate(std::vector<CommandQueue> &queues) {
	
#if DEBUG
	std::cout << debugLocation << 
		"pscrCL::L1SolverContext::allocate: Allocating device memory..." << std::endl;
#endif
		
	if(deviceContexts.size() != queues.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L1SolverContext::allocate: The number of pscrCL::L1DeviceContexts and " <<
			"pscrCL::CommandQueues does not match." << std::endl;
		throw InvalidArgs();
	}
	
	try {
		for(int i = 0; i < (int) deviceContexts.size(); i++) {
			deviceContexts.at(i).allocate(queues.at(i));
	
			// Allocate memory for matrices
			
			cl_int err;
			
			devMemL1Matrices.push_back(cl::Buffer(
				deviceContexts.at(i).getContext(),
				CL_MEM_READ_WRITE,
				l1Matrix.getSize(),
				0, &err));
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot allocate memory for level 1 " <<
					"matrix. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
			
			err = queues.at(i).enqueueWriteBuffer(
				devMemL1Matrices.at(i),
				true,
				0,
				l1Matrix.getSize(),
				l1Matrix.getPointer(),
				0,
				0,
				"l1Matrix");
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot write into the level 1 " <<
					"matrix buffer. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
			
			devMemL2Matrices.push_back(cl::Buffer(
				deviceContexts.at(i).getContext(),
				CL_MEM_READ_WRITE,
				l2Matrix.getSize(),
				0, &err));
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot allocate memory for level 2 " <<
					"matrix. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
			
			err = queues.at(i).enqueueWriteBuffer(
				devMemL2Matrices.at(i),
				true,
				0,
				l2Matrix.getSize(),
				l2Matrix.getPointer(),
				0,
				0,
				"l2Matrix");
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot write into the level 2 " <<
					"matrix buffer. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
			
			devMemL3Matrices.push_back(cl::Buffer(
				deviceContexts.at(i).getContext(),
				CL_MEM_READ_WRITE,
				l3Matrix.getSize(),
				0, &err));
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot allocate memory for level 3 " <<
					"matrix. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
			
			err = queues.at(i).enqueueWriteBuffer(
				devMemL3Matrices.at(i),
				true,
				0,
				l3Matrix.getSize(),
				l3Matrix.getPointer(),
				0,
				0,
				"l3Matrix");
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot write into the level 3 " <<
					"matrix buffer. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
				
			devMemLambda2.push_back(cl::Buffer(
				deviceContexts.at(i).getContext(),
				CL_MEM_READ_WRITE,
				eigenContainer1.getSection(i).getEigenValuesSize(),
				0, &err));
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot allocate memory for level 1 " <<
					"eigenvalues. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
			
			err = queues.at(i).enqueueWriteBuffer(
				devMemLambda2.at(i),
				true,
				0,
				eigenContainer1.getSection(i).getEigenValuesSize(),
				eigenContainer1.getSection(i).getEigenvalues(),
				0,
				0,
				"lambda2");
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot write into the level 1 " <<
					"eigenvalues buffer. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
			
			devMemLambda3.push_back(cl::Buffer(
				deviceContexts.at(i).getContext(),
				CL_MEM_READ_WRITE,
				eigenContainer2.getSection(0).getEigenValuesSize(),
				0, &err));
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot allocate memory for level 2 " <<
					"eigenvalues. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
			
			err = queues.at(i).enqueueWriteBuffer(
				devMemLambda3.at(i),
				true,
				0,
				eigenContainer2.getSection(0).getEigenValuesSize(),
				eigenContainer2.getSection(0).getEigenvalues(),
				0,
				0,
				"lambda3");
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L1SolverContext::allocate: Cannot write into the level 2 " <<
					"eigenvalues buffer. " << CLErrorMessage(err) << std::endl;
				free();
				throw OpenCLError(err);
			}
		}
			
	} catch(...) {
		free();
		throw;
	}
	
	isAllocated = true;
}

void L1SolverContext::free() {
	
#if DEBUG
	std::cout << debugLocation << 
		"pscrCL::L1SolverContext::free: Freeing device memory..." << std::endl;
#endif
	
	devMemL1Matrices.clear();
	devMemL2Matrices.clear();
	devMemL3Matrices.clear();
	
	typename std::vector<L1DeviceContext>::iterator it;
	for(it = deviceContexts.begin(); it != deviceContexts.end(); it++)
		it->free();
	
	isAllocated = false;
}

std::vector<size_t> L1SolverContext::getRequiredTmpSizes() {
	std::vector<size_t> tmp;
	const PscrCLMode mode = l1Matrix.getMode();
	
	for(int i = 0; i < (int) deviceContexts.size(); i++) {
		const int pn = deviceBounds.at(i).second - deviceBounds.at(i).first;
		const int l2Ldf2 = deviceContexts.at(i).getL2DeviceContext().getL2Ldf2();
		const int l2Ldf3 = deviceContexts.at(i).getL2DeviceContext().getL2Ldf3();

		tmp.push_back(
			pn*l2Ldf2*l2Ldf3*getVarSize(mode) + 
			deviceContexts.at(i).getRequiredTmpSizePerSystem());
	}
	return tmp;
}

void L1SolverContext::allocateTmp() {
	std::vector<cl::Buffer> buffers;
	
	std::vector<size_t> requiredTmpSizes = getRequiredTmpSizes();
	
	std::vector<L1DeviceContext>::iterator it;
	for(int i = 0; i < (int) deviceContexts.size(); i++) {
		cl_int err;
		buffers.push_back(cl::Buffer(deviceContexts.at(i).getContext(), 
			CL_MEM_READ_WRITE, requiredTmpSizes.at(i), 0, &err));
		
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"pscrCL::L1SolverContext::allocateTmp: Cannot allocate memory. " << 
				CLErrorMessage(err) << std::endl;
			throw OpenCLError(err);
		}
	}
	setTmp(buffers);
}

void L1SolverContext::setTmp(std::vector<cl::Buffer> &_devMemTmp) {
	
	if(deviceContexts.size() != _devMemTmp.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L1SolverContext::setTmp: The number of L2DeviceContexts and " <<
			"temporary memory buffers do not match." << std::endl;
		throw InvalidArgs();
	}
	
	std::vector<size_t> requiredTmpSizes = getRequiredTmpSizes();
	
	for(int i = 0; i < (int) _devMemTmp.size(); i++) {
		
		cl_int err;
	
		const int pn = deviceBounds.at(i).second - deviceBounds.at(i).first;
		const int l2Ldf2 = deviceContexts.at(i).getL2DeviceContext().getL2Ldf2();
		const int l2Ldf3 = deviceContexts.at(i).getL2DeviceContext().getL2Ldf3();
		
		size_t devMemTmpSize;
		err = _devMemTmp.at(i).getInfo(CL_MEM_SIZE, &devMemTmpSize);
		
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"pscrCL::L2Solver::setTmp: Cannot raed temporary buffer size." << 
				std::endl;
			freeTmp();
			throw InvalidArgs();
		}
		
		if(devMemTmpSize < requiredTmpSizes.at(i)) {
			std::cerr << errorLocation <<
				"pscrCL::L2Solver::setTmp: The temporary buffer is too small." << 
				std::endl;
			freeTmp();
			throw InvalidArgs();
		}

		cl_buffer_region gRegion, tmpRegion;
		gRegion.origin = 0;
		gRegion.size = pn*l2Ldf2*l2Ldf3*getVarSize(mode);
		tmpRegion.origin = pn*l2Ldf2*l2Ldf3*getVarSize(mode);
		tmpRegion.size = deviceContexts.at(i).getRequiredTmpSizePerSystem();
		
		devMemG.push_back(_devMemTmp.at(i).createSubBuffer(
			CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &gRegion, &err));
		
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"pscrCL::L2Solver::setTmp: Cannot create a sub-buffer for the " <<
				"g-vector. " << CLErrorMessage(err) << std::endl;
			freeTmp();
			throw OpenCLError(err);
		}
		
		devMemTmp.push_back(_devMemTmp.at(i).createSubBuffer(
			CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &tmpRegion, &err));
		
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"pscrCL::L2Solver::setTmp: Cannot create a sub-buffer for the " <<
				"temporatery buffer. " << CLErrorMessage(err) << std::endl;
			freeTmp();
			throw OpenCLError(err);
		}
	}
	
	tmpIsAllocated = true;
}

void L1SolverContext::freeTmp() {
	devMemG.clear();
	devMemTmp.clear();
	tmpIsAllocated = false;
}

void L1SolverContext::run(
	std::vector<CommandQueue> &_queues,
	std::vector<cl::Buffer>   &_devMemF,
	const void                *_ch) {
	// TODO: Implement support for multible GPUs
	
#if DEBUG
	std::cout << debugLocation << 
		"pscrCL::L1SolverContext::run: Starging the solution process..." << std::endl;
#endif
	
	if(deviceContexts.size() != _queues.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L1SolverContext::run: The number of pscrCL::L1DeviceContexts and " <<
			"pscrCL::CommandQueues do not match." << std::endl;
		throw InvalidArgs();
	}
	
	if(deviceContexts.size() != _devMemF.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L1SolverContext::run: The number of pscrCL::L1DeviceContexts and " <<
			"right hand side vector sections do not match." << std::endl;
		throw InvalidArgs();
	}
	
	if(!isAllocated) {
		std::cerr << errorLocation <<
			"pscrCL::L1SolverContext::run: The solver is not ready. Call " <<
			"pscrCL::L1SolverContext::allocate()." << std::endl;
		throw UnknownError();
	}
	
	if(!tmpIsAllocated) {
		std::cerr << errorLocation <<
			"pscrCL::L1SolverContext::run: The solver is not ready. Call " <<
			"pscrCL::L1SolverContext::tmpIsAllocated()." << std::endl;
		throw UnknownError();
	}
	
	const int pn = deviceBounds[0].second - deviceBounds[0].first;
	
	cl_int err;
	size_t devMemFSize;
	err = _devMemF[0].getInfo(CL_MEM_SIZE, &devMemFSize);
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
			"pscrCL::L1SolverContext::run: Cannot read f-buffer size." << std::endl;
		throw InvalidArgs();
	}
	
	if(devMemFSize < pn*ldf2*ldf3*getVarSize(mode)) {
		std::cerr << errorLocation <<
			"pscrCL::L1SolverContext::run: The f-buffer is too small." << std::endl;
		throw InvalidArgs();
	}
	
	deviceContexts[0].setArgs(
		_devMemF[0],
		devMemG[0],
		devMemTmp[0],
		devMemL1Matrices[0],
		devMemL2Matrices[0],
		devMemL3Matrices[0],
		devMemLambda2[0],
		devMemLambda3[0]);
	
	deviceContexts[0].runSolver(_queues[0], _ch);
	
}

std::vector<Optimizer> L1SolverContext::createOptimizer(
	const std::vector<cl::Device> &_devices,
	int                             _n3,
	const PscrCLMode               &_mode) {
	
	std::vector<Optimizer> tmp;
	for(int i = 0; i < (int) _devices.size(); i++) {
		Optimizer optimizer = 
			L1DeviceContext::createOptimizer(_devices.at(i), _n3, _mode);
		
		tmp.push_back(optimizer);
	}
	
	return tmp;
}

std::vector<OptValues> L1SolverContext::getDefaultValues(
	const std::vector<cl::Device> &_devices,
	int                             _n3,
	const PscrCLMode               &_mode) {

	std::vector<pscrCL::OptValues> optValues;
	
	for(int i = 0; i < (int) _devices.size(); i++) {
		Optimizer optimizer = L1DeviceContext::createOptimizer(_devices.at(i), _n3, _mode);
		optValues.push_back(optimizer.getDefaultValues());
	}
	
	return optValues;
}

std::vector<OptValues> L1SolverContext::getOptimizedValues(
	std::vector<cl::Context> &_contexts,
	std::vector<cl::Device> &_devices,
	int                       _n1,
	int                       _n2,
	int                       _n3,
	int                       _ldf2,
	int                       _ldf3,
	const PscrCLMode         &_mode) {

	std::vector<pscrCL::OptValues> optValues;
	
	for(int i = 0; i < (int) _contexts.size(); i++) {
		Optimizer optimizer = L1DeviceContext::createOptimizer(_devices.at(i), _n3, _mode);
		L1OptimizerHelper optimizerHelper(
			_contexts.at(i), _devices.at(i), _n1, _n2, _n3, _ldf2, _ldf3, _ldf3, -1, -1, -1, 0, _n1, _mode);
		optValues.push_back(optimizer.getOptimizedValues(optimizerHelper));
	}
	
	return optValues;
}