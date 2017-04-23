/*
 *  Created on: Feb 10, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_L3_OPTIMIZER_HELPER
#define PSCRCL_L3_OPTIMIZER_HELPER

#include "optimizer/AbstractOptimizerHelper.h"
#include "L3DeviceContext.h"
#include "CopyPointer.h"

namespace pscrCL {

class L3OptimizerHelper : public AbstractOptimizerHelper {
public:
	L3OptimizerHelper(
		cl::Context &context,
		cl::Device  &device,
		int          n3,
		int          l3Ldf1,
		int          l3Ldf2,
		int          l3MatLdf,
		int          count1,
		int          count2,
		PscrCLMode   mode);
	
	virtual ~L3OptimizerHelper() {};

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
	int n3;
	int l3Ldf1;
	int l3Ldf2;
	int l3MatLdf;
	int count1;
	int count2;
	PscrCLMode mode;
	
	CopyPointer<CommandQueue> queue;
	CopyPointer<OptValues> optValues;
	CopyPointer<L3DeviceContext> deviceContext;
	
	cl::Buffer devCoefMatrix;
	cl::Buffer devLambda1;
	cl::Buffer devLambda2;
	cl::Buffer devF;
	cl::Buffer devTmp;
	
	bool initialized;
	bool prepared;
};
	
}

#endif