/*
 *  Created on: Feb 24, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_L1_SOLVER_CONTEXT
#define PSCRCL_L1_SOLVER_CONTEXT

#include <vector>
#include "L1DeviceContext.h"
#include "MatrixContainer.h"
#include "eigen/EigenContainer.h"

namespace pscrCL {

/*
 * Level 1 solver
 */
class L1SolverContext {
public:
	L1SolverContext(
	  std::vector< cl::Context > &contexts, 
	  std::vector<cl::Device> &devices,
	  const std::vector<OptValues> &optValues,
	  const void *a1Diag, 
	  const void *a1Offdiag, 
	  const void *m1Diag, 
	  const void *m1Offdiag, 
	  const void *a2Diag, 
	  const void *a2Offdiag, 
	  const void *m2Diag, 
	  const void *m2Offdiag, 
	  const void *a3Diag, 
	  const void *a3Offdiag, 
	  const void *m3Diag, 
	  const void *m3Offdiag,
	  int n1,
	  int n2, 
	  int n3, 
	  int ldf2,
	  int ldf3, 
	  const PscrCLMode& mode
	);
	
	void allocate(std::vector<CommandQueue> &queues);
	
	void free();
	
	// Calculates the amount of temporaty memory required for the solution of
	// a problem
	std::vector<size_t> getRequiredTmpSizes();
	
	void allocateTmp();
	
	void setTmp(std::vector<cl::Buffer> &devMemTmp);
	
	void freeTmp();
	
	void run(
		std::vector<CommandQueue> &queues,
		std::vector<cl::Buffer>   &devMemf,
		const void                *ch
	);
	
	// Creates an Optimizer-object
	static std::vector<Optimizer> createOptimizer(
		const std::vector<cl::Device>  &devices,
		int                             n3,
		const PscrCLMode               &mode);
	
	// Returns the default parameters
	static std::vector<OptValues> getDefaultValues(
		const std::vector<cl::Device>  &devices,
		int                             n3,
		const PscrCLMode               &mode);
	
	// Return the optimized parameters
	static std::vector<OptValues> getOptimizedValues(
		std::vector<cl::Context> &contexts,
		std::vector<cl::Device>  &devices,
		int                       n1,
		int                       n2,
		int                       n3,
		int                       ldf2,
		int                       ldf3,
		const PscrCLMode         &mode);
	
private:
	bool isAllocated;
	bool tmpIsAllocated;
	
	EigenContainer eigenContainer1;
	EigenContainer eigenContainer2;
	std::vector<std::pair<int, int> > deviceBounds;
	std::vector<L1DeviceContext> deviceContexts;
	MatrixContainer l1Matrix;
	MatrixContainer l2Matrix;
	MatrixContainer l3Matrix;
	PscrCLMode mode;
	
	std::vector<cl::Buffer> devMemG;
	std::vector<cl::Buffer> devMemTmp;
	std::vector<cl::Buffer> devMemL1Matrices;
	std::vector<cl::Buffer> devMemL2Matrices;
	std::vector<cl::Buffer> devMemL3Matrices;
	std::vector<cl::Buffer> devMemLambda2;
	std::vector<cl::Buffer> devMemLambda3;
	
	int ldf2;
	int ldf3;
};
	
}

#endif
