/*
 *  Created on: Dec 02, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <iostream>
#include "Guide.h"
#include "common.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

Guide::Guide(
	const Boundaries &_bounds,
	int               _stageY2AMaxSumSize,
	int               _stageY2BMaxSumSize,
	int               _begin, 
	int               _end) {
	
	// TODO: Implement support for multible OpenCL devices
	
	n = _bounds.getN();
	pn = _end-_begin;
	k = LOG4(n) + 1;
	
	stageY2AMaxSumSize = _stageY2AMaxSumSize;
	stageY2BMaxSumSize = _stageY2BMaxSumSize;
	
	cpuGuide = std::vector<HostGuideElement>(k);
	
	stageY1Guide = CArray<StageY1GuideElement>(k*pn);
	stageY1SharedGuide = CArray<StageY1SharedGuideElement>(n);
	stage12AGuide = CArray<Stage12AGuideElement>(2*k*(pn/2));
	stage12BGuide = CArray<StageY2BGuideElement>(3*k*(pn/2));
	stage12CGuide = CArray<Stage12CGuideElement>(k*(pn/2));
	stage22AGuide = CArray<Stage22AGuideElement>(3*k*(pn/2));
	stage22BGuide = CArray<StageY2BGuideElement>(3*k*(pn/2));
	stage22CGuide = CArray<Stage22CGuideElement>(k*pn);
	
	// Counter for stageY1SharedGuide
	int sgc = 0;
	
	for(int i = 1; i <= k; i++) {	
		
		// Counter for stageY1Guide
		int y1gc = (i-1)*pn;
		
		// Counter for stage12AGuide
		int s12agc = 2*(i-1)*(pn/2);
		
		// Counter for stage12BGuide
		int s12bgc = 3*(i-1)*(pn/2);
		
		// Counter for stage12CGuide
		int s12cgc = (i-1)*(pn/2);
		
		// Counter for stage22AGuide
		int s22agc = 3*(i-1)*(pn/2);
		
		// Counter for stage22BGuide
		int s22bgc = 3*(i-1)*(pn/2);
		
		// Counter for stage22CGuide
		int s22cgc = (i-1)*pn;
		
		// These variables keep track where each partial vector sum is located
		int stageY2ASumLocation = 0;
		int stage12BSumLocation = 0;
		int stage22BSumLocation = 0;
		
		// Counter for stageY2BGuide
		
		HostGuideElement &hostGuide = cpuGuide[i-1];
		hostGuide.partialSolutionCount = 0;
		hostGuide.totalSystemCount = 0;
		hostGuide.maxSumSize = 0;
		hostGuide.stage1UpdateCount = 0;
		hostGuide.stage2UpdateCount = 0;
		hostGuide.stage1SumCount = 0;
		hostGuide.stage2SumCount = 0;
		hostGuide.stage1BSumCount = 0;
		hostGuide.stage2BSumCount = 0;
		
		for(int j = 0; j < POW4(k-i); j++) {
			
			const int upperBound = _bounds.getArrayForStep(i)[j];
			const int lowerBound = _bounds.getArrayForStep(i)[j+1];

			hostGuide.maxSumSize = MAX(
				hostGuide.maxSumSize, lowerBound - upperBound - 1);
		}
		
		for(int j = 0; j < POW4(k-i); j++) {
			
			const int upperBound = _bounds.getArrayForStep(i)[j];
			const int middleBound = _bounds.getArrayForStep(i)[j+1];
			const int lowerBound = j < POW4(k-i)-1 ? 
				_bounds.getArrayForStep(i)[j+2] : n;
				
			const int upperSize = middleBound - upperBound - 1;
			const int lowerSize = lowerBound - middleBound - 1;
			
			const int upperNsSize = MIN(upperSize/4, 3);
			const int lowerNsSize = MIN(lowerSize/4, 3);
			
			// TODO: If neither partial solution does not belong to this device, 
			// skip them.
			
			if(middleBound < n && ((i == 1 && 0 < upperSize) || 
				(i != 1 && (0 < upperNsSize || 0 < lowerNsSize)))) {
				
				hostGuide.stage1UpdateCount++;
			
				// Generate guiding information for stage12c
			
				stage12CGuide[s12cgc].updateLoc = middleBound;
			
				bool hasLowerSum =
					i == 1 || (i != 1 && 0 < lowerNsSize);
			
				if(j == 0) {
					stage12CGuide[s12cgc].upperSumLoc = 0;
					stage12CGuide[s12cgc].lowerSumLoc = hasLowerSum ? 
						DIVCEIL(hostGuide.maxSumSize, _stageY2AMaxSumSize) : -1;
				} else {
					stage12CGuide[s12cgc].upperSumLoc = stage12BSumLocation +
							DIVCEIL(hostGuide.maxSumSize, _stageY2AMaxSumSize);
					stage12CGuide[s12cgc].lowerSumLoc = 
						hasLowerSum ? stage12BSumLocation + 
							2 * DIVCEIL(hostGuide.maxSumSize, 
								_stageY2AMaxSumSize) : -1;
				}
										
				s12cgc++;
			}
			
			if((i == 1 && 0 < upperSize) || (i != 1 && 0 < upperNsSize)) {
				
				hostGuide.partialSolutionCount++;
				hostGuide.totalSystemCount += upperSize;

				// Generate shared guiding information for stage11 and stage21
				
				if(i == 1) {
					stageY1SharedGuide[sgc].elem1Loc = 
						1 <= upperSize ? upperBound + 1 : -1;
					stageY1SharedGuide[sgc].elem2Loc = 
						2 <= upperSize ? upperBound + 2 : -1;
					stageY1SharedGuide[sgc].elem3Loc = 
						3 <= upperSize ? upperBound + 3 : -1;
						
					hostGuide.stage2UpdateCount += upperSize;
				} else {
					stageY1SharedGuide[sgc].elem1Loc = 1 <= upperNsSize ? 
						_bounds.getArrayForStep(i-1)[4*j+1] : -1;
					stageY1SharedGuide[sgc].elem2Loc = 2 <= upperNsSize ? 
						_bounds.getArrayForStep(i-1)[4*j+2] : -1;
					stageY1SharedGuide[sgc].elem3Loc = 3 <= upperNsSize ? 
						_bounds.getArrayForStep(i-1)[4*j+3] : -1;
						
					hostGuide.stage2UpdateCount += upperNsSize;
				}
				
				stageY1SharedGuide[sgc].elemULoc = 
					0 <= upperBound ? upperBound : -1;
				stageY1SharedGuide[sgc].elemLLoc = 
					middleBound < n ? middleBound : -1;
					
				// TODO: This assumes that only one GPU is used. Fix when
				// implementing multi-GPU support.
				stageY1SharedGuide[sgc].matElemULoc = 
					0 <= upperBound ? upperBound+1 : -1;
				stageY1SharedGuide[sgc].matElemLLoc = 
					middleBound < n ? middleBound-1 : -1;
					
				stageY1SharedGuide[sgc].eigenLoc = (i-1)*pn + upperBound + 1;
				
				// Generate work group specific guiding information for stage11
				// and stage21
				
				for(int d = 0; d < upperSize; d++) {
					stageY1Guide[y1gc].guideLoc = sgc;
					stageY1Guide[y1gc].sId = d;
					y1gc++;
				}
				
				sgc++;
				
				// Generate guiding information for stage12a and stage12b

				if(0 <= upperBound) {
					hostGuide.stage1SumCount++;
					stage12AGuide[s12agc].upperBound = stageY2ASumLocation;
					stage12AGuide[s12agc].lowerBound = stageY2ASumLocation +
						upperSize;
					stage12AGuide[s12agc].eigenLoc = (i-1)*pn + upperBound + 1;
					stage12AGuide[s12agc].sumSlct = 0;
					
					if(_stageY2AMaxSumSize < upperSize) {
						hostGuide.stage1BSumCount++;
						
						stage12BGuide[s12bgc].upperBound = stage12BSumLocation;
						stage12BGuide[s12bgc].lowerBound = stage12BSumLocation + 
							DIVCEIL(upperSize, _stageY2AMaxSumSize);

						s12bgc++;
					}
					
					stage12BSumLocation += 
						DIVCEIL(hostGuide.maxSumSize, _stageY2AMaxSumSize);
					
					s12agc++;
				}
				
				if(middleBound < n) {
					hostGuide.stage1SumCount++;
					stage12AGuide[s12agc].upperBound = stageY2ASumLocation;
					stage12AGuide[s12agc].lowerBound = stageY2ASumLocation + 
						upperSize;
					stage12AGuide[s12agc].eigenLoc = (i-1)*pn + upperBound + 1;
					stage12AGuide[s12agc].sumSlct = 1;
					
					if(_stageY2AMaxSumSize < upperSize) {
						hostGuide.stage1BSumCount++;
						
						stage12BGuide[s12bgc].upperBound = stage12BSumLocation;
						stage12BGuide[s12bgc].lowerBound = stage12BSumLocation + 
							DIVCEIL(upperSize, _stageY2AMaxSumSize);
							
						s12bgc++;
					}
					
					stage12BSumLocation += 
						DIVCEIL(hostGuide.maxSumSize, _stageY2AMaxSumSize);
					
					s12agc++;
				}
				
				// Generate guiding information for stage22a, stage22b and
				// stage22c
				
				if((i == 1 && 1 <= upperSize) || (i != 1 && 1 <= upperNsSize)) {
					hostGuide.stage2SumCount++;
					stage22AGuide[s22agc].upperBound = stageY2ASumLocation;
					stage22AGuide[s22agc].lowerBound = stageY2ASumLocation +
						upperSize;
					stage22AGuide[s22agc].eigenLoc = (i-1)*pn + upperBound + 1;
					stage22AGuide[s22agc].eigenComId = 1;
					
					if(_stageY2AMaxSumSize < upperSize) {
						hostGuide.stage2BSumCount++;
						
						stage22BGuide[s22bgc].upperBound = stage22BSumLocation;
						stage22BGuide[s22bgc].lowerBound = stage22BSumLocation + 
							DIVCEIL(upperSize, _stageY2AMaxSumSize);

						s22bgc++;
					}
					
					if(i == 1) {
						stage22CGuide[s22cgc].updateLoc = upperBound + 1;
						stage22CGuide[s22cgc].sumLoc = stage22BSumLocation;
					} else {
						stage22CGuide[s22cgc].updateLoc = 
							_bounds.getArrayForStep(i-1)[4*j+1];
						stage22CGuide[s22cgc].sumLoc = stage22BSumLocation;
					}
					
					stage22BSumLocation += 
						DIVCEIL(hostGuide.maxSumSize, _stageY2AMaxSumSize);
					
					s22agc++;
					s22cgc++;
				}
				
				if((i == 1 && 2 <= upperSize) || (i != 1 && 2 <= upperNsSize)) {
					hostGuide.stage2SumCount++;
					stage22AGuide[s22agc].upperBound = stageY2ASumLocation;
					stage22AGuide[s22agc].lowerBound = stageY2ASumLocation +
						upperSize;
					stage22AGuide[s22agc].eigenLoc = (i-1)*pn + upperBound + 1;
					stage22AGuide[s22agc].eigenComId = 2;
					
					if(_stageY2AMaxSumSize < upperSize) {
						hostGuide.stage2BSumCount++;
						
						stage22BGuide[s22bgc].upperBound = stage22BSumLocation;
						stage22BGuide[s22bgc].lowerBound = stage22BSumLocation + 
							DIVCEIL(upperSize, _stageY2AMaxSumSize);
							
						s22bgc++;
					}
					
					if(i == 1) {
						stage22CGuide[s22cgc].updateLoc = upperBound + 2;
						stage22CGuide[s22cgc].sumLoc = stage22BSumLocation;
					} else {
						stage22CGuide[s22cgc].updateLoc = 
							_bounds.getArrayForStep(i-1)[4*j+2];
						stage22CGuide[s22cgc].sumLoc = stage22BSumLocation;
					}
					
					stage22BSumLocation += 
						DIVCEIL(hostGuide.maxSumSize, _stageY2AMaxSumSize);
					
					s22agc++;
					s22cgc++;
				}
				
				if((i == 1 && 3 <= upperSize) || (i != 1 && 3 <= upperNsSize)) {
					hostGuide.stage2SumCount++;
					stage22AGuide[s22agc].upperBound = stageY2ASumLocation;
					stage22AGuide[s22agc].lowerBound = stageY2ASumLocation +
						upperSize;
					stage22AGuide[s22agc].eigenLoc = (i-1)*pn + upperBound + 1;
					stage22AGuide[s22agc].eigenComId = 3;
					
					if(_stageY2AMaxSumSize < upperSize) {
						hostGuide.stage2BSumCount++;
						
						stage22BGuide[s22bgc].upperBound = stage22BSumLocation;
						stage22BGuide[s22bgc].lowerBound = stage22BSumLocation + 
							DIVCEIL(upperSize, _stageY2AMaxSumSize);
							
						s22bgc++;
					}
					
					if(i == 1) {
						stage22CGuide[s22cgc].updateLoc = upperBound + 3;
						stage22CGuide[s22cgc].sumLoc = stage22BSumLocation;
					} else {
						stage22CGuide[s22cgc].updateLoc = 
							_bounds.getArrayForStep(i-1)[4*j+3];
						stage22CGuide[s22cgc].sumLoc = stage22BSumLocation;
					}
					
					stage22BSumLocation += 
						DIVCEIL(hostGuide.maxSumSize, _stageY2AMaxSumSize);
					
					s22agc++;
					s22cgc++;
				}
				
				stageY2ASumLocation += upperSize;
				
			}
			
		}
	}
}

const Guide::HostGuideElement& Guide::getHostGuide(int i) const {
	return cpuGuide[i];
}

const void* Guide::getStageY1GuidePointer() const {
	return stageY1Guide.getPointer();
}

size_t Guide::getStageY1GuideSize() const {
	return stageY1Guide.getSizeInBytes();
}

const void* Guide::getStageY1SharedGuidePointer() const {
	return stageY1SharedGuide.getPointer();
}

size_t Guide::getStageY1SharedGuideSize() const {
	return stageY1SharedGuide.getSizeInBytes();
}

const void* Guide::getStage12AGuidePointer() const {
	return stage12AGuide.getPointer();
}

size_t Guide::getStage12AGuideSize() const {
	return stage12AGuide.getSizeInBytes();
}

const void* Guide::getStage12BGuidePointer() const {
	return stage12BGuide.getPointer();
}

size_t Guide::getStage12BGuideSize() const {
	return stage12BGuide.getSizeInBytes();
}

const void* Guide::getStage12CGuidePointer() const {
	return stage12CGuide.getPointer();
}

size_t Guide::getStage12CGuideSize() const {
	return stage12CGuide.getSizeInBytes();
}

const void* Guide::getStage22AGuidePointer() const {
	return stage22AGuide.getPointer();
}

size_t Guide::getStage22AGuideSize() const {
	return stage22AGuide.getSizeInBytes();
}

const void* Guide::getStage22BGuidePointer() const {
	return stage22BGuide.getPointer();
}

size_t Guide::getStage22BGuideSize() const {
	return stage22BGuide.getSizeInBytes();
}

const void* Guide::getStage22CGuidePointer() const {
	return stage22CGuide.getPointer();
}

size_t Guide::getStage22CGuideSize() const {
	return stage22CGuide.getSizeInBytes();
}