/*
 *  Created on: Feb 24, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_L1_DEVICE_CONTEXT
#define PSCRCL_L1_DEVICE_CONTEXT

#include "DeviceContextHelper.h"
#include "L2DeviceContext.h"

namespace pscrCL {

/*
 * Device context for level 1 solver. 
 */
class L1DeviceContext {
public:
	L1DeviceContext(
		cl::Context                 &context,
		cl::Device                  &device,
		const OptValues             &optValues,
		const Boundaries            &bounds1,
		const Boundaries            &bounds2,
		const EigenSectionInterface &eigenSection1,
		const EigenSectionInterface &eigenSection2,
		int                          upperBound,
		int                          lowerBound,
		int                          l1MatLdf,
		int                          l2MatLdf,
		int                          l3MatLdf,
		int                          l1Ldf2,
		int                          l1Ldf3,
		int                          l2Ldf2,
		int                          l2Ldf3,
		int                          l3Ldf1,
		int                          l3Ldf2,
		int                          n1,
		int                          n2,
		int                          n3,
		const PscrCLMode            &mode);
	
	// Sets the initial kernel arguments
	void setArgs(
		cl::Buffer &devMemF,
		cl::Buffer &devMemG,
		cl::Buffer &devMemTmp,
		cl::Buffer &devMemCoefMat1,
		cl::Buffer &devMemCoefMat2,
		cl::Buffer &devMemCoefMat3,
		cl::Buffer &devMemLambda2,
		cl::Buffer &devMemLambda3);
	
	// Allocate memory for guiding information and eigendata
	void allocate(CommandQueue &queue);
	
	// Frees the space allocated for guiding information and eigendata
	void free();
	
	size_t getRequiredTmpSizePerSystem();
	
	const L2DeviceContext& getL2DeviceContext() const;
	
	// Lauches a reduction step for single L2 system
	void runReductionStep(CommandQueue &queue, int i, const void *ch);
	
	// Lauches a back substitution step for single L2 system
	void runBackSubstitutionStep(CommandQueue &queue, int i, const void *ch);
	
	int getK() const;
	
	cl::Context& getContext();
	
	void runSolver(CommandQueue &queue, const void *ch);
	
	// Creates an Optimizer-object
	static Optimizer createOptimizer(
		const cl::Device &device,
		int               n3,
		const PscrCLMode &mode);
	
	static bool checkOptParams(
		const cl::Device &device,
		const OptValues  &optValues,
		int               l1Ldf3,
		int               l2Ldf3,
		int               n3,
		const PscrCLMode &mode,
		std::string      *err);
	
private:
	cl::Buffer devMemG2;	// g-vector for level 2 problems
	cl::Buffer devMemTmp2;	// Temporary buffer for level 2 problems
	
	L2DeviceContext l2DeviceContext;
	DeviceContextHelper helper;
	PscrCLMode mode;
	int upperBound;
	int lowerBound;
	int n1;
	int n2;
	int l3Ldf2;
};

}

#endif