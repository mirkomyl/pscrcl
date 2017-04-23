/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <complex>
#include "EigenContainer.h"
#include "EigenContainerHelper.h"
#include "../common.h"

using namespace pscrCL;

EigenContainer::EigenContainer() {
	eigenContainerHelper = 0;
}

EigenContainer::EigenContainer(
	const MatrixContainer            &_matrix,
	const Boundaries                 &_bounds,
	const std::vector<SectionBounds> &_sections,
	bool                              _fake) {
	
	if(_matrix.getMode().numComplex()) {
		if(_matrix.getMode().precDouble()) {
			eigenContainerHelper <<= 
				new EigenContainerHelper<std::complex<double>, double>(
					static_cast<const std::complex<double>*>(
						_matrix.getADiag()),
					static_cast<const std::complex<double>*>(
						_matrix.getAOffdiag()), 
					static_cast<const std::complex<double>*>(
						_matrix.getMDiag()),
					static_cast<const std::complex<double>*>(
						_matrix.getMOffdiag()), 
					_bounds, _sections, _fake);
		} else {
			eigenContainerHelper <<= 
				new EigenContainerHelper<std::complex<float>, float>(
					static_cast<const std::complex<float>*>(
						_matrix.getADiag()),
					static_cast<const std::complex<float>*>(
						_matrix.getAOffdiag()), 
					static_cast<const std::complex<float>*>(
						_matrix.getMDiag()),
					static_cast<const std::complex<float>*>(
						_matrix.getMOffdiag()), 
					_bounds, _sections, _fake);	
		}
	} else {
		if(_matrix.getMode().precDouble()) {
			eigenContainerHelper <<= 
				new EigenContainerHelper<double, double>(
					static_cast<const double*>(_matrix.getADiag()),
					static_cast<const double*>(_matrix.getAOffdiag()), 
					static_cast<const double*>(_matrix.getMDiag()),
					static_cast<const double*>(_matrix.getMOffdiag()), 
					_bounds, _sections, _fake);	
		} else {
			eigenContainerHelper <<= 
				new EigenContainerHelper<float, float>(
					static_cast<const float*>(_matrix.getADiag()),
					static_cast<const float*>(_matrix.getAOffdiag()), 
					static_cast<const float*>(_matrix.getMDiag()),
					static_cast<const float*>(_matrix.getMOffdiag()), 
					_bounds, _sections, _fake);	
		}
	}
}

const EigenSectionInterface& EigenContainer::getSection(int sectionId) const {
	return eigenContainerHelper->getSection(sectionId);
}
