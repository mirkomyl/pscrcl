/*
 *  Created on: Jan 13, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_EIGEN_SECTION
#define PSCRCL_EIGEN_SECTION

#include <vector>
#include "EigenSectionInterface.h"
#include "../CArray.h"

namespace pscrCL {

template <typename T>
class EigenSection : public EigenSectionInterface {
	const static int spacing = 5;
	
public:
	EigenSection(int _begin, int _end, int _k) :
		eigenvalues(_k*(_end-_begin)), 
		eigenVectors(_k*spacing*(_end-_begin)), 
		eigenCount(_k) {

		k = _k;
		begin = _begin;
		end = _end;
		
		for(int i = 0; i < k; i++)
			eigenCount[i] = 0;
	}
	virtual const void* getEigenvalues() const {
		return eigenvalues.getPointer();
	}
	virtual const void* getEigenVectors() const {
		return eigenVectors.getPointer();
	}
	virtual int getEigenValuesSize() const {
		return eigenvalues.getSizeInBytes();
	}
	
	virtual int getEigenVectorsSize() const {
		return eigenVectors.getSizeInBytes();
	}
	virtual Clonable* clone() const {
		return new EigenSection<T>(*this);
	}
	int getBegin() const {
		return begin;
	}
	int getEnd() const {
		return end;
	}
	void setEigen(
		T _eigenvalue,
		T _eigenVectorComp0,
		T _eigenVectorComp1,
		T _eigenVectorComp2,
		T _eigenVectorComp3,
		T _eigenVectorComp4,
		int _loc,
		int _level) {

		if(k <= _level || end-begin <= _loc || 
			end-begin <= eigenCount[_level] )
			throw UnknownError();

		eigenvalues[_level*(end-begin)+eigenCount[_level]] = _eigenvalue;
		eigenCount[_level]++;
		
		eigenVectors[(_level*(end-begin)+_loc)*spacing]   = _eigenVectorComp0;
		eigenVectors[(_level*(end-begin)+_loc)*spacing+1] = _eigenVectorComp1;
		eigenVectors[(_level*(end-begin)+_loc)*spacing+2] = _eigenVectorComp2;
		eigenVectors[(_level*(end-begin)+_loc)*spacing+3] = _eigenVectorComp3;
		eigenVectors[(_level*(end-begin)+_loc)*spacing+4] = _eigenVectorComp4;
	}
		
private:
	int k;
	int begin;
	int end;
	CArray<T> eigenvalues;
	CArray<T> eigenVectors;
	CArray<int> eigenCount;
};

}

#endif