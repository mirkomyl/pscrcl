/*
 *  Created on: Dec 30, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include "MatrixContainer.h"
#include "common.h"

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

/*
 * A helper class which is used when permutationg the level 3 matrix into a
 * form suitable for the tridiagonal solver.
 */
template <typename T>
class FlipHelper {
public:
	// Permutates the level 3 matrix into a form suitable for the tridiagonal 
	// solver.
	static char* flip(
		const std::complex<T> *_aDiag, 
		const std::complex<T> *_aOffdiag, 
		const std::complex<T> *_mDiag, 
		const std::complex<T> *_mOffdiag, 
		int                    _n,
		int                    _ldf) {
		
		T *real = 0;
		T *imag = 0;
		try {
			if(_mOffdiag) {
				real = new T[2*4*_ldf];
				imag = real + 4*_ldf;
			} else {
				real = new T[2*3*_ldf];
				imag = real + 3*_ldf;
			}
		} catch(...) {
			if(real) delete [] real;
			if(imag) delete [] imag;
			throw;
		}
		
		for(int i = 0; i < _n; i++) {
			real[       i] = _aDiag[i].real();
			imag[       i] = _aDiag[i].imag();
			real[1*_ldf+i] = _aOffdiag[i].real();
			imag[1*_ldf+i] = _aOffdiag[i].imag();
			real[2*_ldf+i] = _mDiag[i].real();
			imag[2*_ldf+i] = _mDiag[i].imag();
		}
		
		if(_mOffdiag) {
			for(int i = 0; i < _n; i++) {
				real[3*_ldf+i] = _mOffdiag[i].real();
				imag[3*_ldf+i] = _mOffdiag[i].imag();
			}
		}
		
		return static_cast<char*>((void*)real);
	}
};

MatrixContainer::MatrixContainer() : mode(0) {
	flipped = false;
	mTridiagonal = false;
	data = 0;
	n = 0;
	ldf = 0;
}

MatrixContainer::MatrixContainer(
	const void       *_aDiag, 
	const void       *_aOffdiag, 
	const void       *_mDiag, 
	const void       *_mOffdiag, 
	int               _n,
	int               _ldf,
	const PscrCLMode &_mode,
	bool              _flip) :
	mode(_mode) {
	
	if(_ldf < _n)
		throw UnknownError("MatrixContainer::MatrixContainer: Invalid ldf.");
		
	n = _n;
	ldf = _ldf;
	flipped = _flip;
	mTridiagonal = _mOffdiag != 0;
	
	if(_flip && _mode.numComplex()) {
		// Permutate the vectors if neccessary
		if(_mode.precDouble()) {
			data = FlipHelper<double>::flip(
				static_cast<const std::complex<double>*>(_aDiag),
				static_cast<const std::complex<double>*>(_aOffdiag),
				static_cast<const std::complex<double>*>(_mDiag),
				static_cast<const std::complex<double>*>(_mOffdiag),
				_n, _ldf);
		} else {
			data = FlipHelper<float>::flip(
				static_cast<const std::complex<float>*>(_aDiag),
				static_cast<const std::complex<float>*>(_aOffdiag),
				static_cast<const std::complex<float>*>(_mDiag),
				static_cast<const std::complex<float>*>(_mOffdiag),
				_n, _ldf);
		}
	} else {
		const size_t varSize = getVarSize(_mode);
		
		if(_mOffdiag)
			data = new char[4*_ldf*varSize];
		else
			data = new char[3*_ldf*varSize];
		
		char *aDiag = data;
		char *aOffdiag = data + _ldf*varSize;
		char *mDiag = data + 2*_ldf*varSize;
		char *mOffdiag = data + 3*_ldf*varSize;
		
		if(_aDiag)
			memcpy(aDiag, _aDiag, _n*varSize);
		if(_aOffdiag)
			memcpy(aOffdiag, _aOffdiag, _n*varSize);
		if(_mDiag)
			memcpy(mDiag, _mDiag, _n*varSize);
		if(_mOffdiag)
			memcpy(mOffdiag, _mOffdiag, _n*varSize);
	}
}

MatrixContainer::MatrixContainer(const MatrixContainer& old) : 
	mode(old.getMode()) {
	
	n = old.getN();
	ldf = old.getLdf();
	flipped = old.isFlipped();
	mTridiagonal = old.isMTridiagonal();
	
	data = new char[old.getSize()];
	memcpy(data, old.getPointer(), old.getSize());
}

MatrixContainer::~MatrixContainer() {
	if(data) delete [] data;
}

MatrixContainer& MatrixContainer::operator=(const MatrixContainer& a) {
	if(&a == this)
		return *this;
	
	n = a.getN();
	ldf = a.getLdf();
	flipped = a.isFlipped();
	mTridiagonal = a.isMTridiagonal();
	mode = a.getMode();

	if(data) delete [] data;
	
	data = new char[a.getSize()];
	memcpy(data, a.getPointer(), a.getSize());
	
	return *this;
}

const void* MatrixContainer::getPointer() const {
	return data;
}

int MatrixContainer::getSize() const {
	const size_t varSize = getVarSize(mode);
		
	if(isMTridiagonal())
		return 4*getLdf()*varSize;
	else
		return 3*getLdf()*varSize;
}

const void* MatrixContainer::getADiag() const {
	if(isFlipped())
		throw UnknownError();
	
	return data;
}

const void* MatrixContainer::getAOffdiag() const  {
	if(isFlipped())
		throw UnknownError();
	
	return data+getLdf()*getVarSize(mode);
}

const void* MatrixContainer::getMDiag() const   {
	if(isFlipped())
		throw UnknownError();
	
	return data+2*getLdf()*getVarSize(mode);;
}

const void* MatrixContainer::getMOffdiag() const {
	if(isFlipped())
		throw UnknownError();
	
	if(!isMTridiagonal())
		return 0;
	
	return data+3*getLdf()*getVarSize(mode);;
}

int MatrixContainer::getN() const {
	return n;
}

int MatrixContainer::getLdf() const {
	return ldf;
}

const PscrCLMode MatrixContainer::getMode() const {
	return mode;
}

bool MatrixContainer::isFlipped() const {
	return flipped;
}

bool MatrixContainer::isMTridiagonal() const {
	return mTridiagonal;
}
