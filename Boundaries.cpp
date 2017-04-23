/*
 *  Created on: Dec 02, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "Boundaries.h"
#include "common.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

// Helper function which is used when accessing the partial solution boundary 
// data
int slb(int _i, int _k) {
	return (POW4((_i)-1)-1)*POW4(_k-(_i)+1)/3+(_i)-1;
}

Boundaries::Boundaries(int _n) {
	const int k = LOG4(_n) + 1;
	n = _n;

	bounds = CArray<int>(slb(k,k)+2);
	
	bounds[slb(k,k)] = -1;
	bounds[slb(k,k)+1] = n;

	for(int i = k-1; i >= 1; i--) {
		bounds[slb(i,k)] = bounds[slb(i+1,k)];

		for(int j = 0; j < POW4(k-i-1); j++) {
			int last_begin = bounds[slb(i+1,k)+j];
			int last_end = bounds[slb(i+1,k)+j+1];
			int length = last_end-last_begin;
			int kk = MIN((length-1)/4 + 1, 4);
			int id = length/kk;
			int im = length%kk;

			for(int ii = 0; ii < 4; ii++) {
				kk = bounds[slb(i,k)+j*4+ii] + id + MIN(im,1);
				bounds[slb(i,k)+j*4+ii+1] = MIN(kk,last_end);
				im = MAX(im-1,0);
			}
		}
	}
}

int Boundaries::getN() const {
	return n;
}

int Boundaries::getK() const {
	return LOG4(getN()) + 1;
}


const int* Boundaries::getArrayForStep(int _i) const {
	const int k = LOG4(getN()) + 1;
	return bounds.getPointer() + slb(_i, k);
}

int Boundaries::getArrayForStepSize(int _i) const {
	const int k = LOG4(getN()) + 1;
	return POW4(k-_i) + 1;
}

const int* Boundaries::getArray() const {
	return bounds.getPointer();
}

int Boundaries::getArraySize() const {
	const int k = LOG4(getN()) + 1;
	return slb(k,k)+2;
}