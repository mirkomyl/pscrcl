/*
 *  Created on: Jan 13, 2014
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_EIGEN_CONTAINER_HELPER_INTERFACE
#define PSCRCL_EIGEN_CONTAINER_HELPER_INTERFACE

#include "../Clonable.h"
#include "EigenSectionInterface.h"

namespace pscrCL {

class EigenContainerHelperInterface : public Clonable {
public:
	virtual ~EigenContainerHelperInterface() {}
	virtual const EigenSectionInterface& getSection(int sectionId) const = 0;
};

}

#endif