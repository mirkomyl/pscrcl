/*
 *  Created on: Nov 20, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_GUIDE
#define PSCRCL_GUIDE

#include <vector>
#include <CL/cl.hpp>

#include "Boundaries.h"
#include "CArray.h"

namespace pscrCL {

// A class for storing all the necessary guiding information for the host and
// the device.
class Guide {
public:

	// Host side guiding element.
    struct HostGuideElement {
        int partialSolutionCount;	// Number of partial solutions per recursion
									// step. TODO: Obsolete?
        int totalSystemCount;		// Number of L2/L3 subproblems per per 
									// recursion step.
        int stage1UpdateCount;		// Number of update-able f-vector blocks 
									// during the first stage.
        int stage2UpdateCount;		// Number of update-able u-vector blocks 
        							// during the second stage.
        int maxSumSize;
        int stage1SumCount;
		int stage2SumCount;
		int stage1BSumCount;
		int stage2BSumCount;
    };
	
    Guide(const Boundaries &bounds, int stageY2AMaxSumSize, int stageY2BMaxSumSize, int begin, int end);
	
	const HostGuideElement& getHostGuide(int i) const;
	
	const void* getStageY1GuidePointer() const;
	size_t getStageY1GuideSize() const;
	
	const void* getStageY1SharedGuidePointer() const;
	size_t getStageY1SharedGuideSize() const;
	
	const void* getStage12AGuidePointer() const;
	size_t getStage12AGuideSize() const;
	
	const void* getStage12BGuidePointer() const;
	size_t getStage12BGuideSize() const;
	
	const void* getStage12CGuidePointer() const;
	size_t getStage12CGuideSize() const;
	
	const void* getStage22AGuidePointer() const;
	size_t getStage22AGuideSize() const;
	
	const void* getStage22BGuidePointer() const;
	size_t getStage22BGuideSize() const;
	
	const void* getStage22CGuidePointer() const;
	size_t getStage22CGuideSize() const;
	
private:
	
	// Device side guiding element for kernel lx_stage_11
	struct StageY1GuideElement {
		cl_int guideLoc;		// Location of the shared guiding information
		cl_int sId;				// Subproblem index
	};
	
	// Shared device side guiding element for kernel lx_stage_y1
	struct StageY1SharedGuideElement {
		cl_int elem1Loc;		// Location of the 1st element
		cl_int elem2Loc;		// Location of the 2nd element
		cl_int elem3Loc;		// Location of the 3rd element
		cl_int eigenLoc;		// Location of the first eigenvector
		cl_int elemULoc;		// Location of the upper element
		cl_int elemLLoc;		// Location of the lower element
		cl_int matElemULoc;		// Location of the upper matrix element
		cl_int matElemLLoc;		// Location of the lower matrix element
	};
	
	// Device side guiding element for kernel lx_stage_12a
	struct Stage12AGuideElement {
		cl_int upperBound;		// Location of the first vector
		cl_int lowerBound;		// Location of the last vector + 1
		cl_int eigenLoc;		// Location of the first eigenvector
		cl_int sumSlct;			// Upper (0) / Lower (1) sum
	};
	
	// Device side guiding element for kernel lx_stage_y2b
	struct StageY2BGuideElement {
		cl_int upperBound;		// Location of the first vector
		cl_int lowerBound;		// Location of the last vector + 1
	};
	
	// Device side guiding element for kernel lx_stage_12c
	struct Stage12CGuideElement {
		cl_int updateLoc;		// Location of the updateble f-vector block
		cl_int upperSumLoc;		// Location of the upper sum
		cl_int lowerSumLoc;		// Location of the lower sum
	};
	
	// Device side guiding element for kernel lx_stage_22a
	struct Stage22AGuideElement {
		cl_int upperBound;		// Location of the first vector
		cl_int lowerBound;		// Location of the last vector + 1
		cl_int eigenLoc;		// Location of the first eigenvector
		cl_int eigenComId;		// Index of the eigenvector component
	};
	
	// Device side guiding element for kernel lx_stage_22c
	struct Stage22CGuideElement {
		cl_int updateLoc;		// Location of the updateble u-vector block
		cl_int sumLoc;			// Location of the sum
	};
	
	int n;
	int pn;
	int k;
	
	size_t stageY2AMaxSumSize;
	size_t stageY2BMaxSumSize;
	
    std::vector<HostGuideElement> cpuGuide;
	
	CArray<StageY1GuideElement> stageY1Guide;
	CArray<StageY1SharedGuideElement> stageY1SharedGuide;
	
	CArray<Stage12AGuideElement> stage12AGuide;
	CArray<StageY2BGuideElement> stage12BGuide;
	CArray<Stage12CGuideElement> stage12CGuide;
	
	CArray<Stage22AGuideElement> stage22AGuide;
	CArray<StageY2BGuideElement> stage22BGuide;
	CArray<Stage22CGuideElement> stage22CGuide;
};

}

#endif
