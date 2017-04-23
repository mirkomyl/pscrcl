/*
 *  Created on: Jan 13, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_EIGEN_SECTION_INTERFACE
#define PSCRCL_EIGEN_SECTION_INTERFACE

#include "../Clonable.h"

namespace pscrCL {

class EigenSectionInterface : public Clonable {
public:
	virtual ~EigenSectionInterface() {};
	virtual const void* getEigenvalues() const = 0;
	virtual const void* getEigenVectors() const = 0;
	virtual int getEigenValuesSize() const = 0;
	virtual int getEigenVectorsSize() const = 0;
};

}

#endif