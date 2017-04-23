/*
 *  Created on: Feb 25, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_L2_OPTIMIZER_HELPER
#define PSCRCL_L2_OPTIMIZER_HELPER

#include "optimizer/AbstractOptimizerHelper.h"
#include "L2DeviceContext.h"
#include "CopyPointer.h"

namespace pscrCL {

class L2OptimizerHelper : public AbstractOptimizerHelper {
public:
	L2OptimizerHelper(
	  cl::Context &context,
	  cl::Device  &device,
	  int          n2,
	  int          n3,
	  int          l2Ldf2, 
	  int          l2Ldf3,
	  int          l2MatLdf,
	  int          l3MatLdf,
	  int          upperBound,
	  int          lowerBound,
	  PscrCLMode   mode);
	// Tells the helper to allocate all necessary resources. Should be called
	// before the first prepare/evaluate.
	virtual void initialize();

	// Tells tehe helper to recover from invalid Commanqueue situation
	virtual void recover();
	
	// Tells helper to prepare for run with the given OptValues.
	virtual bool prepare(const OptValues& values);

	// Runs the the solver using the OptValues given earlier through prepare()
	// member function.
	virtual void run();

	// Tells the helper to finalize the OpenCL command queue. Blocking.
	virtual void finalize();

	// Tells helper to release the unnecessary resources. Should be called only
	// when all desired evaluations have been carried out.
	virtual void release();	

private:
	cl::Context context;
	cl::Device device;
	int n2;
	int n3;
	int l2Ldf2;
	int l2Ldf3;
	int l2MatLdf;
	int l3MatLdf;
	int upperBound;
	int lowerBound;
	PscrCLMode mode;
	
	CopyPointer<CommandQueue> queue;
	CopyPointer<OptValues> optValues;
	CopyPointer<L2DeviceContext> deviceContext;
	
	cl::Buffer devCoefMatrix2;
	cl::Buffer devCoefMatrix3;
	cl::Buffer devLambda1;
	cl::Buffer devLambda2;
	cl::Buffer devF;
	
	cl::Buffer devG;
	int gSize;
	cl::Buffer devTmp;
	int tmpSize;
	
	bool initialized;
	bool prepared;
};
	
}

#endif