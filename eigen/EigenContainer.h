/*
 *  Created on: Dec 17, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_EIGEN_CONTAINER
#define PSCRCL_EIGEN_CONTAINER

#include <vector>
#include "../MatrixContainer.h"
#include "../Boundaries.h"
#include "../ClonablePointer.h"
#include "EigenContainerHelperInterface.h"

namespace pscrCL {

/*
 * EigenContainer is used to store the eigenvalues and eigenvector components
 * accosiated with the partial solutions. 
 */
class EigenContainer {
public:
	typedef std::pair<int, int> SectionBounds;

	EigenContainer();
    EigenContainer(
        const MatrixContainer            &matrix,
        const Boundaries                 &bounds,
		const std::vector<SectionBounds> &sections,
		bool                              fake = false);

    const EigenSectionInterface& getSection(int sectionId) const;
	
private:
	ClonablePointer<EigenContainerHelperInterface> eigenContainerHelper;
};

}

#endif