/*
 *  Created on: Dec 19, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "DeviceInformation.h"
#include "L2OptimizerHelper.h"
#include "L2SolverContext.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

L2SolverContext::L2SolverContext(
	std::vector< cl::Context >   &_contexts, 
	std::vector<cl::Device>      &_devices,
	const std::vector<OptValues> &_optValues, 
	const void                   *_a2Diag, 
	const void                   *_a2Offdiag, 
	const void                   *_m2Diag, 
	const void                   *_m2Offdiag, 
	const void                   *_a3Diag, 
	const void                   *_a3Offdiag, 
	const void                   *_m3Diag, 
	const void                   *_m3Offdiag, 
	int                           _n2, 
	int                           _n3, 
	int                           _ldf2,
	int                           _ldf3, 
	const PscrCLMode&             _mode) : mode(_mode) {
	
#if DEBUG
	std::cout << debugLocation << 
		"pscrCL::L2SolverContext::L2SolverContext: Initializing a " \
		"pscrCL::L2SolverContext object..." << std::endl;
#endif
		
	// TODO: Iimplement support form multible GPUs
	if(_contexts.size() != 1) {
		std::cerr << errorLocation <<
			"pscrCL::L2SolverContext::L2SolverContext: Multi-GPU support is " \
			"not implemented yet." << std::endl;
		throw InvalidArgs();
	}
	
	if(_contexts.size() != _optValues.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L2SolverContext::L2SolverContext: The number of " \
			"cl::contexts and pscrCL::OptValues do not match." << std::endl;
		throw InvalidArgs();
	}
	
	if(_devices.size() != _optValues.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L2SolverContext::L2SolverContext: The number of " \
			"cl::devices and pscrCL::OptValues do not match." << std::endl;
		throw InvalidArgs();
	}
	
	ldf3 = _ldf3;
	const int wBsize = DeviceInformation::getWBSize(_devices[0]);

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
	
	//
	// Calculate partial solution boundaries
	//
	Boundaries bounds(_n2);
	deviceBounds.push_back(std::pair<int,int>(0, _n2));
	
	//
	// Solve the generalized eigenvalue problems
	//
	eigenContainer = EigenContainer(l2Matrix, bounds, deviceBounds);
	
	//
	// Create pscrCL::deviceContexts
	//
	for(int i = 0; i < (int) deviceBounds.size(); i++) {
		
#if DEBUG
		std::cout << debugLocation << 
			"pscrCL::L2SolverContext::L2SolverContext: Adding a " \
			"pscrCL::L2DeviceContext object for bounds = [" << 
			deviceBounds.at(i).first << "," <<
			deviceBounds.at(i).second << "]..." << std::endl;
#endif
	
		deviceContexts.push_back(L2DeviceContext(
			_contexts.at(i),
			_devices.at(i),
			_optValues.at(i),
			bounds,
			eigenContainer.getSection(i),
			deviceBounds.at(i).first,
			deviceBounds.at(i).second,
			0, 		// "l1Matrix.getLdf()"
			l2Matrix.getLdf(),
			l3Matrix.getLdf(),
			_ldf2,	// l2Ldf2
			_ldf3, 	// l2Ldf3
			1, 		// l3Ldf1
			_n2,	// l3Ldf2
			_n2,	
			_n3,
			false,	// multibleLambda
			mode));
	}
	
	isAllocated = false;
	tmpIsAllocated = false;
	
#if DEBUG
	std::cout << debugLocation << 
		"pscrCL::L2SolverContext::L2SolverContext: pscrCL::L2SolverContext " \
		"object initialized." << std::endl;
#endif
}

void L2SolverContext::allocate(std::vector<CommandQueue> &queues) {
	
#if DEBUG
	std::cout << debugLocation << 
		"pscrCL::L2SolverContext::allocate: Allocating device memory..." << 
		std::endl;
#endif
		
	if(deviceContexts.size() != queues.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L2SolverContext::allocate: The number of " \
			"pscrCL::L2DeviceContexts and pscrCL::CommandQueues do not " \
			"match." << std::endl;
		throw InvalidArgs();
	}
	
	//
	// Goes through all devices; allocates device buffers for the factor 
	// matrices and eigenvalues and writes them into the device memory.
	//
	try {
		for(int i = 0; i < (int) deviceContexts.size(); i++) {
			deviceContexts.at(i).allocate(queues.at(i));
	
			//
			// Allocate device buffers for the factor matrices and write them 
			// into the device memory.
			//
			
			cl_int err;
			
			devMemL2Matrices.push_back(cl::Buffer(
				deviceContexts.at(i).getContext(),
				CL_MEM_READ_WRITE,
				l2Matrix.getSize(),
				0, &err));

			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L2SolverContext::allocate: Cannot allocate " \
					"memory for level 2 factor matrix. " << CLErrorMessage(err);
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
					"pscrCL::L2SolverContext::allocate: Cannot write into " \
					"the level 2 factor matrix buffer. " << CLErrorMessage(err);
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
					"pscrCL::L2SolverContext::allocate: Cannot allocate " \
					"memory for level 3 factor matrix. " << CLErrorMessage(err);
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
					"pscrCL::L2SolverContext::allocate: Cannot write into " \
					"the level 3 factor matrix buffer. " << CLErrorMessage(err);
				free();
				throw OpenCLError(err);
			}
				
			devMemLambda3.push_back(cl::Buffer(
				deviceContexts.at(i).getContext(),
				CL_MEM_READ_WRITE,
				eigenContainer.getSection(i).getEigenValuesSize(),
				0, &err));
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L2SolverContext::allocate: Cannot allocate " \
					"memory for level 2 eigenvalues. " << CLErrorMessage(err);
				free();
				throw OpenCLError(err);
			}
			
			err = queues.at(i).enqueueWriteBuffer(
				devMemLambda3.at(i),
				true,
				0,
				eigenContainer.getSection(i).getEigenValuesSize(),
				eigenContainer.getSection(i).getEigenvalues(),
				0,
				0,
				"lambda3");
			
			if(err != CL_SUCCESS) {
				std::cerr << errorLocation <<
					"pscrCL::L2SolverContext::allocate: Cannot write into " \
					"the level 2 eigenvalues buffer. " << CLErrorMessage(err);
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

void L2SolverContext::free() {
	
#if DEBUG
	std::cout << debugLocation << 
		"pscrCL::L2SolverContext::free: Freeing device memory..." << std::endl;
#endif
	
	devMemL2Matrices.clear();
	devMemL3Matrices.clear();
	devMemLambda3.clear();
	
	typename std::vector<L2DeviceContext>::iterator it;
	for(it = deviceContexts.begin(); it != deviceContexts.end(); it++)
		it->free();
	
	isAllocated = false;
}

std::vector<size_t> L2SolverContext::getRequiredTmpSizes() {
	std::vector<size_t> tmp;
	
	for(int i = 0; i < (int) deviceContexts.size(); i++) {
		const int pn = deviceBounds.at(i).second - deviceBounds.at(i).first;
		const int l3Ldf3 = 
			deviceContexts.at(i).getL3DeviceContext().getL3Ldf3();

		//
		// Shared buffer for the right-hand side vectors and temporary data.
		//
		tmp.push_back(
			pn*l3Ldf3*getVarSize(mode) + 
			deviceContexts.at(i).getRequiredTmpSizePerSystem());
	}
	return tmp;
}

void L2SolverContext::allocateTmp() {
	std::vector<cl::Buffer> buffers;
	
	std::vector<size_t> requiredTmpSizes = getRequiredTmpSizes();
	
	std::vector<L2DeviceContext>::iterator it;
	for(int i = 0; i < (int) deviceContexts.size(); i++) {
		cl_int err;
		buffers.push_back(cl::Buffer(deviceContexts.at(i).getContext(), 
			CL_MEM_READ_WRITE, requiredTmpSizes.at(i), 0, &err));
		
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"pscrCL::L2SolverContext::allocateTmp: Cannot allocate " \
				"temporary memory. " << CLErrorMessage(err) << std::endl;
			throw OpenCLError(err);
		}
		
#if FULL_DEBUG
		std::cout << debugLocation << 
			"pscrCL::L2SolverContext::allocateTmp: Allocated " << 
			requiredTmpSizes.at(i) << " bytes for device #" << i << "." << 
			std::endl;
#endif
	}
	setTmp(buffers);
}

void L2SolverContext::setTmp(std::vector<cl::Buffer> &_devMemTmp) {
	
	if(deviceContexts.size() != _devMemTmp.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L2SolverContext::setTmp: The number of " \
			"pscrCL::L2DeviceContexts and temporary memory buffers do " \
			"not match." << std::endl;
		throw InvalidArgs();
	}
	
	std::vector<size_t> requiredTmpSizes = getRequiredTmpSizes();
	
	for(int i = 0; i < (int) _devMemTmp.size(); i++) {
	
		cl_int err;
		
		const int pn = deviceBounds.at(i).second - deviceBounds.at(i).first;
		const int l3Ldf3 = 
			deviceContexts.at(i).getL3DeviceContext().getL3Ldf3();
		
		//
		// Get global memory size
		//
		size_t devMemSize;
		err = _devMemTmp.at(i).getInfo(CL_MEM_SIZE, &devMemSize);
		
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"pscrCL::L2SolverContext::setTmp: Cannot read device memory " \
				"size for device #" << i << "." << std::endl;
			freeTmp();
			throw OpenCLError();
		}
		
		if(devMemSize < requiredTmpSizes.at(i)) {
			std::cerr << errorLocation <<
				"pscrCL::L2SolverContext::setTmp: Device memory is too " \
				"small for device #" << i << "." << std::endl;
			freeTmp();
			throw OpenCLError();
		}

		cl_buffer_region gRegion, tmpRegion;
		gRegion.origin = 0;
		gRegion.size = pn*l3Ldf3*getVarSize(mode);
		tmpRegion.origin = pn*l3Ldf3*getVarSize(mode);
		tmpRegion.size = deviceContexts.at(i).getRequiredTmpSizePerSystem();
		
		//
		// Create sub-buffer for the right-hand side vectors.
		//
		devMemG.push_back(_devMemTmp.at(i).createSubBuffer(
			CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &gRegion, &err));
		
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"pscrCL::L2SolverContext::setTmp: Cannot create a sub-buffer " \
				"for the g-vector. " << CLErrorMessage(err) << std::endl;
			freeTmp();
			throw OpenCLError(err);
		}
		
		//
		// Create sub-buffer for temporary data.
		//
		devMemTmp.push_back(_devMemTmp.at(i).createSubBuffer(
			CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &tmpRegion, &err));
		
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"pscrCL::L2SolverContext::setTmp: Cannot create a sub-buffer " \
				"for the temporatery buffer. " << CLErrorMessage(err) << 
				std::endl;
			freeTmp();
			throw OpenCLError(err);
		}
	}
	
	tmpIsAllocated = true;
}

void L2SolverContext::freeTmp() {
	devMemG.clear();
	devMemTmp.clear();
	tmpIsAllocated = false;
}

void L2SolverContext::run(
	std::vector<CommandQueue> &_queues,
	std::vector<cl::Buffer>   &_devMemF,
	int                        count,
	const void                *_ch) {
	
#if DEBUG
	std::cout << debugLocation << 
		"pscrCL::L2SolverContext::run: Starging the solution process..." 
		<< std::endl;
#endif
	
	if(deviceContexts.size() != _queues.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L2SolverContext::run: The number of " \
			"pscrCL::L2DeviceContexts and pscrCL::CommandQueues do not " \
			"match." << std::endl;
		throw InvalidArgs();
	}
	
	if(deviceContexts.size() != _devMemF.size()) {
		std::cerr << errorLocation <<
			"pscrCL::L2SolverContext::run: The number of " \
			"pscrCL::L2DeviceContexts and right hand side vector sections do " \
			"not match." << std::endl;
		throw InvalidArgs();
	}
	
	if(!isAllocated) {
		std::cerr << errorLocation <<
			"pscrCL::L2SolverContext::run: The solver is not ready. Call " \
			"pscrCL::L2SolverContext::allocate()." << std::endl;
		throw UnknownError();
	}
	
	if(!tmpIsAllocated) {
		std::cerr << errorLocation <<
			"pscrCL::L2SolverContext::run: The solver is not ready. Call " \
			"pscrCL::L2SolverContext::setTmp()." << std::endl;
		throw UnknownError();
	}
	
	for(int i = 0; i < (int) deviceContexts.size(); i++) {
	
		const int pn = deviceBounds.at(i).second - deviceBounds.at(i).first;
		
		cl_int err;
		size_t devMemFSize;
		err = _devMemF.at(i).getInfo(CL_MEM_SIZE, &devMemFSize);
		
		if(err != CL_SUCCESS) {
			std::cerr << errorLocation <<
				"pscrCL::L2SolverContext::run: Cannot device memory size for " \
				"device #" << i << "." << std::endl;
			throw InvalidArgs();
		}
		
		if(devMemFSize < pn*ldf3*getVarSize(mode)) {
			std::cerr << errorLocation <<
				"pscrCL::L2SolverContext::run: device memory size is too " \
				"small for device #" << i << "." << std::endl;
			throw InvalidArgs();
		}
		
		deviceContexts.at(i).setArgs(
			_devMemF.at(i),
			devMemG.at(i),
			devMemTmp.at(i),
			devMemL2Matrices.at(i),
			devMemL3Matrices.at(i),
			devMemTmp.at(i), // This should be "null"
			devMemLambda3.at(i));
		
	}
	
	// TODO: Implement support for multible GPUs
	deviceContexts[0].runSolver(
		_queues[0], 0, count, _ch);
}

std::vector<Optimizer> L2SolverContext::createOptimizer(
	const std::vector<cl::Device> &_device,
	int                             _n3,
	const PscrCLMode               &_mode) {
	
	std::vector<Optimizer> tmp;
	typename std::vector<cl::Device>::const_iterator it;
	for(it = _device.begin(); it != _device.end(); it++) {
		Optimizer optimizer = 
			L2DeviceContext::createOptimizer(*it, _n3, _mode);
		
		tmp.push_back(optimizer);
	}
	
	return tmp;
}

std::vector<OptValues> L2SolverContext::getDefaultValues(
	const std::vector<cl::Device> &_devices,
	int                             _n3,
	const PscrCLMode               &_mode) {

	std::vector<pscrCL::OptValues> optValues;
	
	std::vector<cl::Device>::const_iterator it;
	for(it = _devices.begin(); it < _devices.end(); it++) {
		Optimizer optimizer = L2DeviceContext::createOptimizer(*it, _n3, _mode);
		optValues.push_back(optimizer.getDefaultValues());
	}
	
	return optValues;
}

std::vector<OptValues> L2SolverContext::getOptimizedValues(
	std::vector<cl::Context> &_contexts,
	std::vector<cl::Device>  &_devices,
	int                       _n2,
	int                       _n3,
	int                       _ldf3,
	const PscrCLMode         &_mode) {

	std::vector<pscrCL::OptValues> optValues;
	
	for(int i = 0; i < (int) _contexts.size(); i++) {
		Optimizer optimizer = 
			L2DeviceContext::createOptimizer(_devices.at(i), _n3, _mode);
		L2OptimizerHelper optimizerHelper(
			_contexts.at(i), _devices.at(i), _n2, _n3, _n2, _ldf3, -1, -1, 0, 
				_n2, _mode);
		optValues.push_back(optimizer.getOptimizedValues(optimizerHelper));
	}
	
	return optValues;
}