/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_L3_DEVICE_CONTEXT
#define PSCRCL_L3_DEVICE_CONTEXT

#include "KernelHelper.h"

namespace pscrCL {

class L3DeviceContext {
public:
	L3DeviceContext(
		cl::Context      &context,
		cl::Device       &device,
		const OptValues  &optValues,
		int               n3,
		int               l3Ldf1,
		int               l3Ldf2,
		int               l3MatLdf,
		bool              multibleLambda,
		const PscrCLMode &mode);

	void setArgs(
		const cl::Buffer &devMemF,
		const cl::Buffer &devMemTmp,
		const cl::Buffer &devMemLambda2,
		const cl::Buffer &devMemLambda3,
		const cl::Buffer &devMemCoefMatrix);

	// Returns the amount of global memory required to solve one tridiagonal
	// system
	int getL3Ldf3() const;

	size_t getRequiredTmpSizePerSystem();
	
	bool stageAEnabled() const;

	bool stageBEnabled() const;

	void run(
		CommandQueue &queue,
		int         lambdaStride2,
		int         lambdaStride3,
		int	        count2,
		int         count3,
		const void *ch);

	int getN3() const;
	
	static Optimizer createOptimizer(
			const cl::Device &device,
			int               n3,
			const PscrCLMode &mode);
	
	static bool checkOptValues(
		const cl::Device &device,
		const OptValues  &optValues,
		int               n3,
		const PscrCLMode &mode,
		std::string      *err);

private:
	Optimizer optimizer;
	OptValues optValues;
	KernelHelper helper;
	PscrCLMode mode;
	int n3;
	int l3Ldf3;
	bool argsAreSet;
};

}

#endif // L3DEVICECONTEXT_H_
