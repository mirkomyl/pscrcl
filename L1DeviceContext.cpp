#include "L1DeviceContext.h"
#include "common.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

L1DeviceContext::L1DeviceContext(
		cl::Context                 &_context,
		cl::Device                  &_device,
		const OptValues             &_optValues,
		const Boundaries            &_bounds1,
		const Boundaries            &_bounds2,
		const EigenSectionInterface &_eigenSection1,
		const EigenSectionInterface &_eigenSection2,
		int                          _upperBound,
		int                          _lowerBound,
		int                          _l1MatLdf,
		int                          _l2MatLdf,
		int                          _l3MatLdf,
		int                          _l1Ldf2,
		int                          _l1Ldf3,
		int                          _l2Ldf2,
		int                          _l2Ldf3,
		int                          _l3Ldf1,
		int                          _l3Ldf2,
		int                          _n1,
		int                          _n2,
		int                          _n3,
		const PscrCLMode            &_mode) :
		l2DeviceContext(_context, _device, _optValues, _bounds2, _eigenSection2,
			0, _n2, _l1MatLdf, _l2MatLdf, _l3MatLdf,
			_l2Ldf2, _l2Ldf3, _l3Ldf1, _l3Ldf2, _n2, _n3, true, _mode),
		helper(_context, _device, _optValues, _bounds1, _eigenSection1, 
			_upperBound, _lowerBound,
			_l1MatLdf, _l2MatLdf, _l3MatLdf, 
			_l1Ldf2, _l1Ldf3, _l2Ldf2, _l2Ldf3,
			_n1, _n2, _l3Ldf1, _n3, _mode, DeviceContextHelper::level1, false),
		mode(_mode) {
			
	upperBound = _upperBound;
	lowerBound = _lowerBound;
	n1 = _n1;
	n2 = _n2;
	l3Ldf2 = _l3Ldf2;
}

void L1DeviceContext::setArgs(
	cl::Buffer &_devMemF,
	cl::Buffer &_devMemG,
	cl::Buffer &_devMemTmp,
	cl::Buffer &_devMemCoefMat1,
	cl::Buffer &_devMemCoefMat2,
	cl::Buffer &_devMemCoefMat3,
	cl::Buffer &_devMemLambda2,
	cl::Buffer &_devMemLambda3) {
	
	cl_int err;
	
	size_t devMemTmpSize;
	err = _devMemTmp.getInfo(CL_MEM_SIZE, &devMemTmpSize);
	
	if(err != CL_SUCCESS)  {
		std::cerr << errorLocation <<
			"L1DeviceContext::setArgs: Cannot read temporary buffer size." << 
			std::endl;
		throw InvalidArgs();
	}
		
	if(devMemTmpSize < getRequiredTmpSizePerSystem()) {
		std::cerr << errorLocation <<
			"L1DeviceContext::setArgs: The temporary buffer is too small." << 
			std::endl;
		throw InvalidArgs();
	}
	
	helper.setArgs(
		_devMemF, _devMemG, _devMemTmp, 
		_devMemCoefMat1, _devMemCoefMat2, _devMemCoefMat3);
	
	cl_buffer_region gRegion, tmpRegion;
	gRegion.origin = 0;
	gRegion.size = (lowerBound-upperBound) * l3Ldf2 *
		l2DeviceContext.getL3DeviceContext().getL3Ldf3()*getVarSize(mode);
	tmpRegion.origin = gRegion.size;
	tmpRegion.size = (lowerBound-upperBound) * 
		l2DeviceContext.getRequiredTmpSizePerSystem();
	
	devMemG2 = _devMemTmp.createSubBuffer(
		CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &gRegion, &err);
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
			"L1DeviceContext::setArgs: Cannot create a sub-buffer for " << 
			"the level 2 g-vector. " << CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	devMemTmp2 = _devMemTmp.createSubBuffer(
		CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &tmpRegion, &err);
	
	if(err != CL_SUCCESS) {
		std::cerr << errorLocation <<
			"L1DeviceContext::setArgs: Cannot create a sub-buffer for " << 
			"the level 2 tmp-vector. " << CLErrorMessage(err) << std::endl;
		throw OpenCLError(err);
	}
	
	l2DeviceContext.setArgs(
		_devMemG, devMemG2, devMemTmp2, 
		_devMemCoefMat2, _devMemCoefMat3, _devMemLambda2, _devMemLambda3);
		
}

void L1DeviceContext::allocate(CommandQueue &_queue) {
	helper.allocate(_queue);
	l2DeviceContext.allocate(_queue);
}

void L1DeviceContext::free() {
	devMemG2 = cl::Buffer();
	devMemTmp2 = cl::Buffer();
	helper.free();
	l2DeviceContext.free();
}

size_t L1DeviceContext::getRequiredTmpSizePerSystem() {
	return MAX(
		helper.getRequiredTmpSizePerSystem(),
		(lowerBound-upperBound)*(
			l3Ldf2*l2DeviceContext.getL3DeviceContext().getL3Ldf3()*getVarSize(mode) + 
			l2DeviceContext.getRequiredTmpSizePerSystem()));
}

const L2DeviceContext& L1DeviceContext::getL2DeviceContext() const {
	return l2DeviceContext;
}


cl::Context& L1DeviceContext::getContext() {
	return helper.getContext();
}

void L1DeviceContext::runReductionStep(
	CommandQueue &_queue, 
	int           _i, 
	const void   *_ch) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"L1DeviceContext::runReductionStep: Queueing step i = " << _i << "..." 
		<< std::endl;
#endif
	
	helper.setCh(_ch);
	
	helper.queueStage11(_queue, n2, _i);

	l2DeviceContext.runSolver(
		_queue, 
		(_i-1)*(lowerBound-upperBound),
		helper.getSystemCount(_i-1),
		_ch);

	helper.queueStage12a(_queue, n2, _i);
	helper.queueStage12b(_queue, n2, _i);
	
#if FULL_DEBUG
	std::cout << std::endl;
#endif
}
void L1DeviceContext::runBackSubstitutionStep(
	CommandQueue &_queue, 
	int           _i, 
	const void   *_ch) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"L1DeviceContext::runBackSubstitutionStep: Queueing step i = " << _i << "..." 
		<< std::endl;
#endif
	
	helper.setCh(_ch);
	
	helper.queueStage21(_queue, n2, _i);
	
	l2DeviceContext.runSolver(
		_queue, 
		_i*(lowerBound-upperBound),
		helper.getSystemCount(_i),
		_ch);

	helper.queueStage22a(_queue, n2, _i);
	helper.queueStage22b(_queue, n2, _i);
	
#if FULL_DEBUG
	std::cout << std::endl;
#endif
}

int L1DeviceContext::getK() const {
	return LOG4(n1) + 1;
}

void L1DeviceContext::runSolver(
	CommandQueue &_queue, 
	const void   *_ch) {
	
#if FULL_DEBUG
	std::cout << std::endl;
#endif
	
	const int k = LOG4(n1) + 1;

	for(int i = 1; i <= k-1; i++)
		runReductionStep(_queue, i, _ch);

	for(int i = k-1; 0 <= i; i--)
		runBackSubstitutionStep(_queue, i, _ch);
	
}

Optimizer L1DeviceContext::createOptimizer(
	const cl::Device &_device, 
	int               _n3, 
	const PscrCLMode &_mode) {
	
#if FULL_DEBUG
	std::cout << debugLocation << 
		"L1DeviceContext::createOptimizer: Creating an Optimizer object..." 
		<< std::endl;
#endif

	Optimizer optimizer1 = DeviceContextHelper::createOptimizer(
		_device, _n3, _mode, DeviceContextHelper::level1);
	
	Optimizer optimizer2 = L2DeviceContext::createOptimizer(
		_device, _n3, _mode);
		
	return optimizer1 + optimizer2;
	
}

bool L1DeviceContext::checkOptParams(
	const cl::Device &_device, 
	const OptValues  &_optValues, 
	int               _l1Ldf3, 
	int               _l2Ldf3, 
	int               _n3, 
	const PscrCLMode &_mode,
	std::string      *_err) {
	
	if(!DeviceContextHelper::checkOptParams(_device, _optValues, _l1Ldf3,
		_n3, _mode, DeviceContextHelper::level1, _err))
		return false;
	
	if(!L2DeviceContext::checkOptParams(_device, _optValues, _l2Ldf3, 
		_n3, _mode, _err))
		return false;
	
	return true;
	
}