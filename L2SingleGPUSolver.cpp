/*
 *  Created on: Jan 23, 2015
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "L2SolverContext.h"

#include "pscrCL.hpp" 

using namespace pscrCL;

L2SingleGPUSolver::L2SingleGPUSolver(
	cl_context    _context,
	cl_device_id  _device,
	opt_params    *_optParams,
	const void   *_a2Diag,
	const void   *_a2Offdiag,
	const void   *_m2Diag,
	const void   *_m2Offdiag,
	const void   *_a3Diag,
	const void   *_a3Offdiag,
	const void   *_m3Diag,
	const void   *_m3Offdiag,
	int           _n2,
	int           _n3,
	int           _ldf2,
	int           _ldf3,
	mode_flag     _mode) {
	
	std::vector<cl::Context> contexts;
	contexts.push_back(cl::Context(_context));
	
	std::vector<cl::Device> devices;
	devices.push_back(cl::Device(_device));
	
	const std::vector<OptValues> optValues;
	
	if(_optParams) {
		std::vector<OptValues> defaultOptValues = 
			L2SolverContext::getDefaultValues(
				cl::Device(_device), _n3, PscrCLMode(_mode));
		
		opt_params::const_iterator it;
		for(it = _optParams->begin(); it != _optParams->end(); it++) {
			defaultOptValues.front()
		}
	} else {
		optValues.push_back(
			L2SolverContext::getDefaultValues(_device, _n3, _mode));
	}
	
	ptr = new RefPtr;
	ptr->ref = 1;
	ptr->ptr = new L2SolverContext(
		contexts,
		devices,
		optValues,
		_a2Diag, 
		_a2Offdiag, 
		_m2Diag, 
		_m2Offdiag, 
		_a3Diag, 
		_a3Offdiag, 
		_m3Diag, 
		_m3Offdiag,
		_n2, 
		_n3,
		_ldf2,
		_ldf3, 
		_mode);
}

L2SingleGPUSolver::L2SingleGPUSolver(const L2SingleGPUSolver &_a) {
	ptr = _a.ptr;
	ptr->ref++;
}

L2SingleGPUSolver& L2SingleGPUSolver::operator=(const L2SingleGPUSolver & _a) {
	if(&_a == *this)
		return this;
	
	ptr->ref--;
	if(ptr->ref == 0) {
		delete ptr->ptr;
		delete ptr;
	}
	
	ptr = _a.ptr;
	ptr->ref++;
	
	return *this;
}

L2SingleGPUSolver::~L2SingleGPUSolver() {
	ptr->ref--;
	if(ptr->ref == 0) {
		delete ptr->ptr;
		delete ptr;
	}
}

void L2SingleGPUSolver::allocate(cl_command_queue queue) {
	((L2SolverContext*) ptr->ptr)->allocate(CommandQueue(queue));
}

void L2SingleGPUSolver::free() {
	((L2SolverContext*) ptr->ptr)->free();
}

size_t L2SingleGPUSolver::getRequiredTmpSize() {
	((L2SolverContext*) ptr->ptr)->getRequiredTmpSizes()[0];
}

void L2SingleGPUSolver::allocateTmp() {
	((L2SolverContext*) ptr->ptr)->allocateTmp();
}

void L2SingleGPUSolver::setTmp(cl_mem devMemTmp) {
	((L2SolverContext*) ptr->ptr)->setTmp(cl::Buffer(devMemTmp));
}

void L2SingleGPUSolver::freeTmp() {
	((L2SolverContext*) ptr->ptr)->freeTmp();
}

void L2SingleGPUSolver::run(
	cl_command_queue  queue,
	cl_mem            devMem,
	int               count,
	const void       *ch) {
	
}

static opt_params getOptimalParams(
	cl_context   contexts,
	cl_device_id devices,
	int          n2,
	int          n3,
	int          ldf2,
	int          ldf3,
	mode_flag    mode);