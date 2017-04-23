/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_MATH
#define PSCRCL_MATH

namespace pscrCL {

// Mathematical functions

// x^2
template <typename T>
T mul2(const T &x) { return x * x; }

template <typename T>
T real(const T &value) { return value; }

template <typename T>
T imag(const T &value) { return 0; }

template <typename PREC>
PREC real(const std::complex<PREC> &value) { return value.real(); }

template <typename PREC>
PREC imag(const std::complex<PREC> &value) { return value.imag(); }

template <typename T>
T conjugate(const T &value) { return value; }

template <typename PREC>
std::complex<PREC> conjugate(const std::complex<PREC> &value) {
	return std::conj(value);
}

template <typename T>
T conmul2(const T& a) {
	return a * a;
}

template <typename PREC>
PREC conmul2(const std::complex<PREC>& a) {
	return mul2(a.real()) + mul2(a.imag());
}

template <typename T>
T dot(const T *a, const T *b, int n) {
	T dot = 0.0;
	for(int i = 0; i < n; i++)
		dot += a[i] * conjugate(b[i]);
	return dot;
}

template <typename T>
void axpy(T *result, T coef, const T *a, const T *b, int n) {
	for(int i = 0; i < n; i++)
		result[i] = coef * a[i] + b[i];
}

// <y,Ax>
template <typename T>
T matvecdot(const T *diag, const T *codiag, const T* x, const T *y, int n) {
	T tmp = 0.0;

	if(codiag && 1 < n) {
		tmp += (diag[0] * x[0] + codiag[1] * x[1]) * conjugate(y[0]);
		for(int i = 1; i < n-1; i++)
			tmp += (codiag[i] * x[i-1] + diag[i] * 
				x[i] + codiag[i+1] * x[i+1]) * conjugate(y[i]);
		tmp += (codiag[n-1] * x[n-2] + diag[n-1] * x[n-1]) * conjugate(y[n-1]);
	} else {
		for(int i = 0; i < n; i++)
			tmp += diag[i] * x[i] * conjugate(y[i]);
	}

	return tmp;
}

template <typename T>
T absVal(const T &value) {
	return fabs(value);
}

template <typename PREC>
PREC absVal(const std::complex<PREC> &value) {
	return sqrt(conmul2(value));
}

template <typename T>
T asum(const T *v, int n) {
	T tmp = 0.0;
	for(int i = 0; i < n; i++)
		tmp += absVal(v[i]);
	return tmp;
}

template <typename PREC>
PREC asum(const std::complex<PREC> *v, int n) {
	PREC tmp = 0.0;
	for(int i = 0; i < n; i++)
		tmp += absVal(v[i]);
	return tmp;
}

template <typename T>
void scal(const T &coef, T *v, int n) {
	for(int i = 0; i < n; i++)
		v[i] *= coef;
}

template <typename PREC>
void scal(const PREC &coef, std::complex<PREC> *v, int n) {
	for(int i = 0; i < n; i++)
		v[i] *= coef;
}

template <typename T>
T norm(const T *x, int n) {
	return sqrt(dot(x, x, n));
}

template <typename PREC>
PREC norm(const std::complex<PREC> *x, int n) {
	PREC dot = 0.0;
	for(int i = 0; i < n; i++)
		dot += conmul2(x[i]);
	return sqrt(dot);
}

}

#endif
