/*
 *  Created on: Dec 17, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_EIGEN_CONTAINER_HELPER
#define PSCRCL_EIGEN_CONTAINER_HELPER

#include "../Boundaries.h"
#include "../common.h"
#include "EigenSolver.h"
#include "EigenContainerHelperInterface.h"
#include "EigenSection.h"

namespace pscrCL {

/*
 * A template helper class used by the Eigencontainer class. Solves the 
 * eigenvalue problems accosiated with the partial solutions. 
 */
template <typename T, typename PREC>
class EigenContainerHelper : public EigenContainerHelperInterface {
public:
	typedef std::pair<int, int> EigenSectionBounds;
	
    EigenContainerHelper(
        const T                               *_aDiag,
        const T                               *_aOffdiag,
        const T                               *_mDiag,
        const T                               *_mOffdiag,
        const Boundaries                      &_bounds,
		const std::vector<EigenSectionBounds> &_sections,
		bool                                   _fake = false) {

		const int n = _bounds.getN();
		const int k = LOG4(n) + 1;
		
		T *tmp = 0;
		
		try {
			tmp = new T[(9+n)*n];
			
			typename std::vector<EigenSectionBounds>::const_iterator it;
			for(it = _sections.begin(); it != _sections.end(); it++)
				sections.push_back(EigenSection<T>(it->first, it->second, k));
		
			if(_fake) {
				delete [] tmp;
				return;
			}
			
			// Go throug all partial solutions and solve the corresponding
			// eigenvalue problems
			
			for(int i = 1; i <= k; i++) {
				for(int j = 0; j < POW4(k-i); j++) { 
					const int upperBound = _bounds.getArrayForStep(i)[j];
					handlePartialSolution(_aDiag, _aOffdiag, _mDiag, _mOffdiag, tmp, 
						upperBound,
						_bounds.getArrayForStep(i)[j+1],
						i == 1 ? 0 : _bounds.getArrayForStep(i-1)[4*j+1] - upperBound - 1,
						i == 1 ? 1 : _bounds.getArrayForStep(i-1)[4*j+2] - upperBound - 1,
						i == 1 ? 2 : _bounds.getArrayForStep(i-1)[4*j+3] - upperBound - 1,
						i-1);
				}
			}
					
		} catch(...) {
			if(tmp) delete [] tmp;
			throw;
		}

		delete [] tmp;
	}
	
    virtual const EigenSectionInterface& getSection(int sectionId) const {		
		return sections[sectionId];
	}
	
	virtual Clonable* clone() const {
		return new EigenContainerHelper(*this);
	}
		
private:
	
	void handlePartialSolution(
        const T *_aDiag,
        const T *_aOffdiag,
        const T *_mDiag,
        const T *_mOffdiag,
		T       *_tmp,
		int      _upperBound,
		int      _lowerBound,
		int      _elem1,
		int      _elem2,
		int      _elem3,
		int      _level) {

		const int nn = _lowerBound - _upperBound - 1;
		
		if(nn < 1)
			return;
		
		const T *aDiag = _aDiag + _upperBound + 1;
		const T *aOffdiag = _aOffdiag + _upperBound + 1;
		const T *mDiag = _mDiag + _upperBound + 1;
		
		const T *mOffdiag = _mOffdiag ? _mOffdiag + _upperBound + 1 : 0;
		
		T *newDiag = _tmp;
		T *newOffdiag = _tmp + nn;
		T *eigenvalues = _tmp + 2*nn;
		T *eigenVectors = _tmp + 3*nn;
		T *tmp = _tmp + (3+nn)*nn;
		
		// Convert a generalized problem into a normal one
		EigenSolver<T, PREC>::crawford(newDiag,	newOffdiag,	tmp, 
			aDiag, aOffdiag, mDiag, mOffdiag, nn);
		
		// Solve eigenvalues
		EigenSolver<T, PREC>::lrWilkinson(
			eigenvalues, tmp, newDiag, newOffdiag, nn);

		// Solve eigenvectors
		EigenSolver<T, PREC>::inverseIteration(eigenVectors, tmp, 
			eigenvalues, aDiag, aOffdiag, mDiag, mOffdiag, nn);

		// Orthonormalize the eigenvector and copy the required components
		for(int i = 0; i < nn; i++) {

			T *eigenvector = eigenVectors + i*nn;
			
			// Calculate the orthonormalization coefficient
			T s;
			if(1 < nn && _mOffdiag) {
				T dot = 0;
				dot += eigenvector[0] * 
					(mDiag[0] * eigenvector[0] + mOffdiag[1] * eigenvector[1]);
				for(int j = 1; j < nn-1; j++)
					dot += eigenvector[j] * 
						(mOffdiag[j] * eigenvector[j-1] + 
						mDiag[j] * eigenvector[j] +
						mOffdiag[j+1] * eigenvector[j+1]);
				dot += eigenvector[nn-1] * 
					(mOffdiag[nn-1] * eigenvector[nn-2] + 
					mDiag[nn-1] * eigenvector[nn-1]);

				s = (T) 1.0/sqrt(dot);
			} else if(1 < nn) {
				T dot = 0.0;
				for(int j = 0; j < nn; j++)
					dot += eigenvector[j] * mDiag[j] * eigenvector[j];
				s = (T) 1.0/sqrt(dot);
			} else {
				s = (T) 1.0/sqrt(mDiag[0]*eigenvector[0]);
			}

			// Copy the eigenvalue and the required eigenvector components
			typename std::vector<EigenSection<T> >::iterator it;
			for(it = sections.begin(); it != sections.end(); it++) {
				if(it->getBegin() <= _upperBound+i+1 && 
					_upperBound+i+1 < it->getEnd()) {
					
					// These checks are neccessary when nn < 3
					T eigenVectorElem1 = _elem1 < nn ? eigenvector[_elem1] : 0;
					T eigenVectorElem2 = _elem2 < nn ? eigenvector[_elem2] : 0;
					T eigenVectorElem3 = _elem3 < nn ? eigenvector[_elem3] : 0;
						
					it->setEigen(
						eigenvalues[i],
						s*eigenvector[0],
						s*eigenVectorElem1,
						s*eigenVectorElem2,
						s*eigenVectorElem3,
						s*eigenvector[nn-1],
						_upperBound+i+1 - it->getBegin(), _level);

					break;
				}
			}
		}
	}
	std::vector<EigenSection<T> > sections;
};

}

#endif