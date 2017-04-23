/*
 *  Created on: Nov 28, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef LXDEVICECONTEXTHELPER_H_
#define LXDEVICECONTEXTHELPER_H_

#include "common.h"
#include "KernelHelper.h"
#include "Guide.h"
#include "eigen/EigenSectionInterface.h"

namespace pscrCL {

/*
 * Level 1 and 2 DeviceContext helper class. Contains code which is common for
 * both levels. 
 */
class DeviceContextHelper {
public:
	enum LevelType { level1 = 1, level2 = 2 };
	
	DeviceContextHelper(
		cl::Context      &context,		// OpenCL context
		cl::Device       &device,		// OpenCL device
		const OptValues  &optValues,	// Parameter value container
		const Boundaries &bounds,		// Partial solution boundaries
		const EigenSectionInterface &eigenSection, // Eigenvalues and -vectors
		int               upperBound,	// The upper boundary of the section 
		int               lowerBound,	// The lower boundary of the section
		int               l1MatLdf,		// Ldf for level 1 matrix
		int               l2MatLdf,		// Ldf for level 2 matrix
		int               l3MatLdf,		// Ldf for level 3 matrix
		int               lxLdf2,		// Internal ldf2 for the current level
		int               lxLdf3,		// Internal ldf3 for the current level
		int               lxp1Ldf2,		// Internel ldf2 for the upper level
		int               lxp1Ldf3,		// Internal ldf3 for the upper level
		int               nx,			// Size of the current level
		int               nxp1,			// Size of the upper level
		int               l3Ldf1,		//
		int               n3,			// Size of the third level
		const PscrCLMode &mode,			// Solver mode
		LevelType         level,		// Current solver level. The upper level
										// is current level + 1
		bool              multibleLambda
	);
	
	// Sets the initial kernel arguments
	void setArgs(
		cl::Buffer &devMemF,
		cl::Buffer &devMemG,
		cl::Buffer &devMemTmp,
		cl::Buffer &devMemMatX,
		cl::Buffer &devMemMatXp1,
		cl::Buffer &devMisc);
	
	// Sets the constant c
	void setCh(const void *ch);
	
	// Allocate memory for guiding information and eigendata
	void allocate(CommandQueue &queue);
	
	// Frees the space allocated for guiding information and eigendata
	void free();
	
	// Calculates the amount of temporaty memory required for the solution of
	// a level X problem
	size_t getRequiredTmpSizePerSystem();
	
	// Returns the number of partial solutions per recursion level
	int getSystemCount(int i) const;
	
	// Returns the OpenCL context assiciated with the DeviceContext
	cl::Context& getContext();
	
	// Enqueues the lx_stage11 kernel
	void queueStage11(
		CommandQueue &queue,
		int           secondaryIndexSpaceSize,
		int           i);
	
	// Enqueues the lx_stage12 and lx_stagey2b kernels
	void queueStage12a(
		CommandQueue &queue,
		int           secondaryIndexSpaceSize,
		int           i);
	
	// Enqueues the lx_stage12c kernel
	void queueStage12b(
		CommandQueue &queue,
		int           secondaryIndexSpaceSize,
		int           i,
		int           lambdaStride = 0);
	
	// Enqueues the lx_stage21 kernel
	void queueStage21(
		CommandQueue &queue,
		int           secondaryIndexSpaceSize,
		int           i,
		int           lambdaStride = 0);
		
	// Enqueues the lx_stage22 and lx_stagey2b kernels
	void queueStage22a(
		CommandQueue &queue,
		int           secondaryIndexSpaceSize,
		int           i);
	
	// Enqueues the lx_stage22c kernel
	void queueStage22b(
		CommandQueue &queue,
		int           secondaryIndexSpaceSize,
		int           i);
	
	// Creates an Optimizer-object
	static Optimizer createOptimizer(
		const cl::Device &device,
		int               n3,
		const PscrCLMode &mode,
		LevelType         level
		);
	
	static bool checkOptParams(
		const cl::Device &device,
		const OptValues  &optValues,
		int               lxLdf3,
		int               n3,
		const PscrCLMode &mode,
		LevelType         level,
		std::string      *err);
		
private:
	// Return true if the initial kernel arguments are set and temporaty memory
	// is allocated. 
	bool checkStatus(bool throwException = true);
	
	// lx_stageXYZ => l1_stageXYZ / l2_stageXYZ
	std::string getAlias(const std::string &name) const;
	
	cl::Context context;
	KernelHelper kernelHelper;
	Optimizer optimizer;
	OptValues optValues;
	Guide guide;
	ClonablePointer<EigenSectionInterface> eigenSection;
	
	cl::Buffer stageY1Guide1;
	cl::Buffer stageY1Guide2;
	
	cl::Buffer stage12AGuide;
	cl::Buffer stage12BGuide;
	cl::Buffer stage12CGuide;
	
	cl::Buffer stage22AGuide;
	cl::Buffer stage22BGuide;
	cl::Buffer stage22CGuide;
	
	cl::Buffer eigenVectors;
	
	bool argsAreSet;
	bool chIsSet;
	bool isAllocated;
	
	PscrCLMode mode;
	LevelType level;
	bool multibleLambda;
	int pn;
	
	int lxLdf3;
	
	int nx;
	int nxp1;
	int n3;
	
	int maxSumSizeA;
	int maxSumSizeB;
};
	
}

#endif