#include "L2DeviceContext.h"
#include "common.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

L2DeviceContext::L2DeviceContext(
		cl::Context                 &_context,
		cl::Device                  &_device,
		const OptValues             &_optValues,
		const Boundaries            &_bounds,
		const EigenSectionInterface &_eigenSection,
		int                          _upperBound,
		int                          _lowerBound,
		int                          _l1MatLdf,
		int                          _l2MatLdf,
		int                          _l3MatLdf,
		int                          _l2Ldf2,
		int                          _l2Ldf3,
		int                          _l3Ldf1,
		int                          _l3Ldf2,
		int                          _n2,
		int                          _n3,
		bool                         _multibleLambda,
		const PscrCLMode            &_mode) :
		l3DeviceContext(_context, _device, _optValues, _n3, 
			_l3Ldf1, _l3Ldf2, _l3MatLdf, _multibleLambda, _mode),
		helper(_context, _device, _optValues, _bounds, _eigenSection, 
			_upperBound, _lowerBound,
			_l1MatLdf, _l2MatLdf, _l3MatLdf, 
			_l2Ldf2, _l2Ldf3, _l3Ldf2, l3DeviceContext.getL3Ldf3(),
			_n2, _n3, _l3Ldf1, _n3, _mode, DeviceContextHelper::level2, 
			_multibleLambda) {
			
	upperBound = _upperBound;
	lowerBound = _lowerBound;
	n2 = _n2;
	l2Ldf2 = _l2Ldf2;
	l2Ldf3 = _l2Ldf3;
	
}

void L2DeviceContext::setArgs(
	cl::Buffer &_devMemF,
	cl::Buffer &_devMemG,
	cl::Buffer &_devMemTmp,
	cl::Buffer &_devMemCoefMat2,
	cl::Buffer &_devMemCoefMat3,
	cl::Buffer &_devMemLambda2,
	cl::Buffer &_devMemLambda3) {
	
	helper.setArgs(
		_devMemF, _devMemG, _devMemTmp, 
		_devMemCoefMat2, _devMemCoefMat3, _devMemLambda2);
	
	l3DeviceContext.setArgs(
		_devMemG,
		_devMemTmp,
		_devMemLambda2,
		_devMemLambda3,
		_devMemCoefMat3);
}

void L2DeviceContext::allocate(CommandQueue &_queue) {
	helper.allocate(_queue);
}

void L2DeviceContext::free() {
	helper.free();
}

size_t L2DeviceContext::getRequiredTmpSizePerSystem() {
	return MAX(
		helper.getRequiredTmpSizePerSystem(),
		(lowerBound-upperBound)*l3DeviceContext.getRequiredTmpSizePerSystem());
}

const L3DeviceContext& L2DeviceContext::getL3DeviceContext() const {
	return l3DeviceContext;
}


cl::Context& L2DeviceContext::getContext() {
	return helper.getContext();
}

int L2DeviceContext::getL2Ldf2() const {
	return l2Ldf2;
}

int L2DeviceContext::getL2Ldf3() const {
	return l2Ldf3;
}

void L2DeviceContext::runReductionStepBegin(
	CommandQueue &_queue, 
	int           _i, 
	int           _lambdaStride2, 
	int           _count2, 
	const void   *_ch) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"L2DeviceContext::runReductionStepBegin: Queueing step i = " << _i << "..." 
		<< std::endl;
#endif
	
	helper.setCh(_ch);

	helper.queueStage11(_queue, _count2, _i);

	l3DeviceContext.run(
		_queue, 
		_lambdaStride2, 
		(_i-1)*(lowerBound-upperBound),
		_count2,
		helper.getSystemCount(_i-1),
		_ch);

	helper.queueStage12a(_queue, _count2, _i);
	
}
void L2DeviceContext::runReductionStepEnd(
	CommandQueue &_queue, 
	int           _i, 
	int           _lambdaStride2, 
	int           _count2, 
	const void   *_ch) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"L2DeviceContext::runReductionStepEnd: Queueing step i = " << _i << "..." 
		<< std::endl;
#endif
	
	helper.setCh(_ch);

	helper.queueStage12b(_queue, _count2, _i, _lambdaStride2);
	
#if FULL_DEBUG
	std::cout << std::endl;
#endif
	
}

void L2DeviceContext::runBackSubstitutionStepBegin(
	CommandQueue &_queue, 
	int           _i, 
	int           _lambdaStride2, 
	int           _count2, 
	const void   *_ch) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"L2DeviceContext::runBackSubstitutionStepBegin: Queueing step i = " << _i << "..." 
		<< std::endl;
#endif
	
	helper.setCh(_ch);
	
	helper.queueStage21(_queue, _count2, _i, _lambdaStride2);
	
	l3DeviceContext.run(
		_queue, 
		_lambdaStride2, 
		_i*(lowerBound-upperBound),
		_count2,
		helper.getSystemCount(_i),
		_ch);

	helper.queueStage22a(_queue, _count2, _i);

}
void L2DeviceContext::runBackSubstitutionStepEnd(
	CommandQueue &_queue, 
	int           _i, 
	int           _count2, 
	const void   *_ch) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"L2DeviceContext::runBackSubstitutionStepEnd: Queueing step i = " << _i << "..." 
		<< std::endl;
#endif
	
	helper.setCh(_ch);
	
	helper.queueStage22b(_queue, _count2, _i);
	
#if FULL_DEBUG
	std::cout << std::endl;
#endif
}

int L2DeviceContext::getK() const {
	return LOG4(n2) + 1;
}

void L2DeviceContext::runSolver(
	CommandQueue &_queue, 
	int           _lambdaStride2,
	int	          _count2,
	const void   *_ch) {
	
#if FULL_DEBUG
	std::cout << std::endl;
#endif
	
	const int k = getK();

	for(int i = 1; i <= k-1; i++) {
		runReductionStepBegin(_queue, i, _lambdaStride2, _count2, _ch);
		runReductionStepEnd(_queue, i, _lambdaStride2, _count2, _ch);
	}

	for(int i = k-1; 0 <= i; i--) {
		runBackSubstitutionStepBegin(_queue, i, _lambdaStride2, _count2, _ch);
		runBackSubstitutionStepEnd(_queue, i, _count2, _ch);
	}
}

Optimizer L2DeviceContext::createOptimizer(
	const cl::Device &_device, 
	int               _n3, 
	const PscrCLMode &_mode) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"L2DeviceContext::createOptimizer: Creating an Optimizer object..." 
		<< std::endl;
#endif

	Optimizer optimizer2 = DeviceContextHelper::createOptimizer(
		_device, _n3, _mode, DeviceContextHelper::level2);
		
	Optimizer optimizer3 = L3DeviceContext::createOptimizer(
		_device, _n3, _mode);
		
	return optimizer2 + optimizer3;
	
}

bool L2DeviceContext::checkOptParams(
	const cl::Device &_device, 
	const OptValues  &_optValues, 
	int               _l2Ldf3, 
	int               _n3, 
	const PscrCLMode &_mode,
	std::string      *_err) {
	
	if(!DeviceContextHelper::checkOptParams(_device, _optValues, _l2Ldf3,
		_n3, _mode, DeviceContextHelper::level2, _err))
		return false;
	
	if(!L3DeviceContext::checkOptValues(_device, _optValues, _n3, _mode, _err))
		return false;
	
	return true;
	
}

