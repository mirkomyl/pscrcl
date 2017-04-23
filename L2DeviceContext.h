/*
 *  Created on: Dec 13, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_L2_DEVICE_CONTEXT
#define PSCRCL_L2_DEVICE_CONTEXT

#include "DeviceContextHelper.h"
#include "L3DeviceContext.h"

namespace pscrCL {

/*
 * Device context for level 2 solver. 
 */
class L2DeviceContext {
public:
	L2DeviceContext(
		cl::Context                 &context,
		cl::Device                  &device,
		const OptValues             &optValues,
		const Boundaries            &bounds,
		const EigenSectionInterface &eigenSection,
		int                          upperBound,
		int                          lowerBound,
		int                          l1MatLdf,
		int                          l2MatLdf,
		int                          l3MatLdf,
		int                          l2Ldf2,
		int                          l2Ldf3,
		int                          l3Ldf1,
		int                          l3Ldf2,
		int                          n2,
		int                          n3,
		bool                         multibleLambda,
		const PscrCLMode            &mode
	);
	
	// Sets the initial kernel arguments
	void setArgs(
		cl::Buffer &devMemF,
		cl::Buffer &devMemG,
		cl::Buffer &devMemTmp,
		cl::Buffer &devMemCoefMat2,
		cl::Buffer &devMemCoefMat3,
		cl::Buffer &devMemLambda2,
		cl::Buffer &devMemLambda3);
	
	// Allocates memory for guiding information and eigendata
	void allocate(CommandQueue &queue);
	
	// Frees the space allocated for guiding information and eigendata
	void free();
	
	// Returns the amount of temporary memory required by one level 2 problem
	size_t getRequiredTmpSizePerSystem();
	
	// Returns the level 3 device context associated with this level 2 device
	// context.
	const L3DeviceContext& getL3DeviceContext() const;
	
	// Lauches a reduction step for multible L2 systems, part A
	void runReductionStepBegin(
		CommandQueue &queue, 
		int           i, 
		int           lambdaStride2, 
		int           count2, 
		const void   *ch);
	
	// Lauches a reduction step for multible L2 systems, part B
	void runReductionStepEnd(
		CommandQueue &queue, 
		int           i, 
		int           lambdaStride2, 
		int           count2, 
		const void   *ch);
	
	// Lauches a back substitution step for multible L2 systems, part A
	void runBackSubstitutionStepBegin(
		CommandQueue &queue, 
		int           i, 
		int           lambdaStride2, 
		int           count2, 
		const void   *ch);
	
	// Lauches a back substitution step for multible L2 systems, part B
	void runBackSubstitutionStepEnd(
		CommandQueue &queue, 
		int           i, 
		int           count2, 
		const void   *ch);
	
	int getK() const;
	
	cl::Context& getContext();
	
	int getL2Ldf2() const;
	int getL2Ldf3() const;
	
	void runSolver(
		CommandQueue &queue, 
		int           lambdaStride2,
		int	          count2,
		const void   *ch);
	
	// Creates an Optimizer-object
	static Optimizer createOptimizer(
		const cl::Device &device,
		int               n3,
		const PscrCLMode &mode);
	
	static bool checkOptParams(
		const cl::Device &device,
		const OptValues  &optValues,
		int               l2Ldf3,
		int               n3,
		const PscrCLMode &mode,
		std::string      *err);
	
private:
	L3DeviceContext l3DeviceContext;
	DeviceContextHelper helper;
	int upperBound;
	int lowerBound;
	int n2;
	int l2Ldf2;
	int l2Ldf3;
};

}

#endif