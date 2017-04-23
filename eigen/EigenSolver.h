/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_EIGENSOLVER
#define PSCRCL_EIGENSOLVER

#include "../common.h"
#include "math.h"

namespace pscrCL {
	
/*
 * A template class for the solution of the eigenvalue problems.
 * Public static member functions:
 *   lrWilkinson		A LR algorithm with Wilkinson's shift for the solution
 * 						of the eigenvectors.
 *   crawford			Crawford's algorithm. Used to preprocess the 
 * 						generalized eigenvalue problems into a form suitable
 * 						for the LR algorithm.
 *   inverseIteration	An Inverse iteration algorithm for the solution of the 
 * 						eigenvectors
 *   
 */
template <typename T, typename PREC>
class EigenSolver {
public:
	/*
	 * A LR algorithm with Wilkinson's shift
	 * Arguments:
	 *   eigenvalues	A buffer which will contain the requested eigenvectors
	 * 					after a successful function call
	 *   tmp			A temporary buffer of the size 3*n*sizeof(T)
	 *   diag			Diagonal elements of the coefficient matrix
	 *   offdiag		Off-diagonal elements of the coefficient matrix
	 * 	 n				Size of the eigenvalue problem
	 */
	static void lrWilkinson(
			T       *eigenvalues,
			T       *tmp,
			const T *diag,
			const T *codiag,
			int      n) {

		T *a = tmp;
		T *b = tmp + n;
		T *c = tmp + 2*n;

		// TODO: Add parameters for single precision

		const PREC eps1 = 1.0E-6;
		const PREC eps2 = 1.0E-20;
		const int maxIter = 30;

		if(n == 1) {
			eigenvalues[0] = diag[0];
			return;
		}

		// Form the matrix
		b[0] = diag[0];
		for(int i = 1; i < n; i++) {
			a[i] = codiag[i];
			b[i] = diag[i];
			c[i-1] = codiag[i];
		}

		for(int i = n-1; 0 < i; i--) {
			T s = 0.0;

			// Inner iteration
			int it;
			for(it = 0; it < maxIter; it++) {
				T ps = s;
				s = sqrt(mul2(b[i-1]) + (T) 4.0*c[i-1]*a[i] - 
					(T) 2.0*b[i-1]*b[i] + mul2(b[i]));
				T p = (T) 0.5*(b[i-1] + b[i] + s);
				T q = (T) 0.5*(b[i-1] + b[i] - s);

				if(absVal(p - q) < eps1*absVal(p))
					s = b[i];
				else if(absVal(p - b[i]) <= absVal(q - b[i]))
					s = p;
				else
					s = q;

				T shift = 0.0;
				if(absVal(s - ps) < 0.5*absVal(ps))
					shift = s;

				T ad = b[i];

				b[0] -= shift;
				for(int j = 1; j <= i; j++) {
					a[j] /= b[j-1];
					b[j] -= c[j-1]*a[j] + shift;
				}

				b[0] += a[1]*c[0] + shift;
				for(int j = 1; j <= i-1; j++) {
					a[j] *= b[j];
					b[j] += a[j+1]*c[j] + shift;
				}
				a[i] *= b[i];
				b[i] += shift;

				ad -= b[i];

				if(absVal((ad)) <= eps2*absVal(real(b[i])) && 
					absVal(imag(ad)) <= eps2*absVal(imag(b[i])))
					break;
			}
/*			
			if(it == maxIter) {
				std::cerr << errorMsgBegin <<
					" / EigenSolver::lrWilkinson: " \
					"Unable to solve the eigenvalues. Maximum number of " \
					"iterations exceeded. " << std::endl;
					
				throw UnknownError("Unable to solve the eigenvalues.");
			}
*/		}

		for(int i = 0; i < n; i++)
			eigenvalues[i] = b[i];

	}

	/*
	 * Crawford's algorithm
	 * Arguments:
	 *   newDiag		Diagonal elements of the new coefficient matrix
	 *   newOffdiag		Off-diagonal elements of the new coefficient matrix
	 *   tmp			A temporary buffer of the size 2*n*sizeof(T)
	 *   aDiag			Diagonal elements of the matrix A
	 *   aOffdiag		Off-diagonal elements of the matrix A
	 *   mDiag			Diagonal elements of the matrix M
	 *   mOffdiag		Off-diagonal elements of the matrix M
	 *   n				Size of the eigenvalue problems
	 */
	static void crawford(
			T       *newDiag,
			T       *newOffdiag,
			T       *tmp,
			const T *aDiag,
			const T *aOffdiag,
			const T *mDiag,
			const T *mOffdiag,
			int      n) {

		if(n == 1) {
			newDiag[0] = aDiag[0] / mDiag[0];
			return;
		}

		// No need for Crawford
		if(mOffdiag == 0) {
			newDiag[0] = aDiag[0]/mDiag[0];
			for(int i = 1; i < n; i++) {
				newDiag[i] = aDiag[i]/mDiag[i];
				newOffdiag[i] = aOffdiag[i]/sqrt(mDiag[i-1]*mDiag[i]);
			}
			return;
		}

		// Form Cholesky
		T *cDiag = tmp;
		T *cOffdiag = tmp + n;

		cDiag[0] = sqrt(mDiag[0]);
		for(int i = 1; i < n; i++) {
			cOffdiag[i] = mOffdiag[i] / cDiag[i-1];
			cDiag[i] = sqrt(mDiag[i] - mul2(cOffdiag[i]));
		}

		// Main loop

		T bc = aDiag[0];
		newDiag[0] = bc/mul2(cDiag[0]);
		newDiag[1] = aDiag[1] + cOffdiag[1]*(bc*cOffdiag[1]/cDiag[0] - (T) 2.0*aOffdiag[1])/cDiag[0];
		newOffdiag[1] = (aOffdiag[1] - bc*cOffdiag[1]/cDiag[0])/cDiag[0];

		for(int i = 1; i < n-1; i++) {
			bc = newDiag[i];
			T e = (T) -1.0*newOffdiag[i]*cOffdiag[i+1]/cDiag[i];
			newDiag[i] = bc/mul2(cDiag[i]);
			newOffdiag[i] /= cDiag[i];
			newDiag[i+1] = aDiag[i+1] + cOffdiag[i+1]*(bc*cOffdiag[i+1]/cDiag[i] - (T) 2.0*aOffdiag[i+1])/cDiag[i];
			newOffdiag[i+1] = (aOffdiag[i+1] - bc*cOffdiag[i+1]/cDiag[i])/cDiag[i];

			for(int j = i; 1 <= j; j--) {
				PREC t;
				T r, s;
				if(absVal(e*e) == 0.0) {
					t = 1.0;
					r = 1.0;
					s = 0.0;
//#if FULL_DEBUG
//					std::cout << warningMsgBegin <<
//							" / EigenSolver::crawford: Zero warning." << std::endl;
//#endif
				} else {
					t = absVal(e) + absVal(newOffdiag[j+1]);
					r = newOffdiag[j+1]/sqrt(mul2(e/t) + mul2(newOffdiag[j+1]/t));
					s = (T) -1.0*e/sqrt(mul2(e/t) + mul2(newOffdiag[j+1]/t));
				}

				newOffdiag[j+1] = (newOffdiag[j+1]*r - e*s)/t;
				if(1 < j) {
					e = (T) -1.0*s*newOffdiag[j-1]/t;
					newOffdiag[j-1] = newOffdiag[j-1]*r/t;
				}
				T bp = newDiag[j-1];
				bc = newDiag[j];
				newDiag[j-1] = (bp*r*r + (T) 2.0*newOffdiag[j]*r*s + bc*s*s)/(t*t);
				newDiag[j] = (bc*r*r - (T) 2.0*newOffdiag[j]*r*s + bp*s*s)/(t*t);
				newOffdiag[j] = (newOffdiag[j]*(r*r - s*s) + (bc - bp)*r*s)/(t*t);
			}
		}
		
		newDiag[n-1] /= mul2(cDiag[n-1]);
		newOffdiag[n-1] /= cDiag[n-1];

	}

	/*
	 * Inverse iteration algorithm for the solutions of the eigenvectors
	 * Arguments:
	 *   eigenVectors	A buffer which will contain the requested eigenvectors
	 * 					after a successful function call.
	 *   tmp			Temporary buffer of the size 6*n*sizeof(T)
	 *   eigenvalues	A buffer containing the eigenvalues
	 *   aDiag			Buffer containing the diagonal elements of the matrix A
	 *   aOffdiag		Buffer containing the off-diagonal elements of the 
	 * 					matrix A
	 *   mDiag			Buffer containing the diagonal elements of the matrix M
	 *   mOffdiag		Buffer containing the off-diagonal elements of the 
	 * 					matrix M
	 *   n				Size of the eigenvalue problem
	 */
	static void inverseIteration(
			T       *eigenVectors,
			T       *tmp,
			const T *eigenvalues,
			const T *aDiag,
			const T *aOffdiag,
			const T *mDiag,
			const T *mOffdiag,
			int      n) {
		
		if(n == 1) {
			eigenVectors[0] = 1.0;
			return;
		}

		T *diag = tmp;
		T *codiag = tmp + n;
		T *uDiag = tmp + 2*n;
		T *uOffdiag = tmp + 3*n;
		T *lOffdiag = tmp + 4*n;
		T *pivot = tmp + 5*n;

		const PREC eps = 1.0E-3;
		const int maxIter = 5;

		const PREC t = sqrt(1.0*n)*espmat(aDiag, aOffdiag, n);

		for(int j = 0; j < n; j++) {

			T *eigenVector = eigenVectors+j*n;
			
			// Generate the matrix
			for(int i = 0; i < n; i++)
				diag[i] = aDiag[i] - eigenvalues[j]*mDiag[i];
			if(mOffdiag)
				for(int i = 1; i < n; i++)
					codiag[i] = aOffdiag[i] - eigenvalues[j]*mOffdiag[i];
			else
				for(int i = 1; i < n; i++)
					codiag[i] = aOffdiag[i];

			// Form LU decomposition
			formLU(lOffdiag, uDiag, uOffdiag, pivot, diag, codiag, n);
			
			// Initial guess
			for(int i = 0; i < n; i++)
				eigenVector[i] = t;

			backwardLU(eigenVector, uDiag, uOffdiag, pivot, n);

			// Orthogonalization in the case of close eigenvalues
			for(int i = 0; i < j-1; i++) {
				if(absVal(eigenvalues[i] - eigenvalues[j]) < 
					eps*MAX(absVal(eigenvalues[i]), absVal(eigenvalues[j]))) {
					T dot = (T) -1.0*matvecdot(mDiag, mOffdiag, eigenVector, 
										   eigenVectors+i*n, n) /
							matvecdot(mDiag, mOffdiag, eigenVectors+i*n, 
											eigenVectors+i*n, n);
					axpy(eigenVector, dot, eigenVectors+i*n, eigenVector, n);
				}
			}

			PREC s = asum(eigenVector, n);

			// Main loop
			int it;
			for(it = 0; it < maxIter; it++) {
				// Scale the eigenvector
				scal(t/s, eigenVector, n);
				
				forwardLU(eigenVector, uDiag, lOffdiag, codiag, n);
				backwardLU(eigenVector, uDiag, uOffdiag, pivot, n);
				
				// Orthogonalization in the case of close eigenvalues
				for(int i = 0; i < j-1; i++) {
					if(absVal(eigenvalues[i] - eigenvalues[j]) < eps*MAX(absVal(eigenvalues[i]), absVal(eigenvalues[j]))) {
						T dot = (T) -1.0*matvecdot(diag, codiag, eigenVector, eigenVectors+i*n, n) /
								matvecdot(diag, codiag, eigenVectors+i*n, eigenVectors+i*n, n);
						axpy(eigenVector, dot, eigenVectors+i*n, eigenVector, n);
					}
				}

				s = asum(eigenVector, n);

				if(1.0 <= s)
					break;
			}

			// Scale the eigenvector
			scal((PREC)1.0/s, eigenVector, n);
		}
		
	}

private:
#ifdef __GNUC__
	static __attribute__((optimize("O0"))) PREC espmat(
#else
	static PREC espmat(
#endif
			const T *diag,
			const T *codiag,
			int      n) {

		PREC norm = absVal(diag[0]);

		for(int i = 1; i < n; i++)
			norm = MAX(absVal(diag[i]) + absVal(codiag[i]), norm);

		PREC a = 4.0/3.0;
		PREC c = a - 1.0;
		a = c + c + c;
		PREC eps = absVal(a - 1.0);
		
		if(eps == 0.0)
			throw UnknownError("EigenSolver::espmat failed.");

		if(eps*norm == 0.0)
			return eps;
		
		return eps*norm;

		// This should work (I think) but instead it causes NaNs
		// return norm * std::numeric_limits<PREC>::min();

	}

	static void formLU(
			T       *lOffdiag,
			T       *uDiag,
			T       *uOffdiag,
			T       *pivot,
			const T *diag,
			const T *codiag,
			int      n) {

		T u = diag[0];
		T v = 1 < n ? codiag[1] : 0.0;

		for(int i = 1; i < n; i++) {
			if(absVal(codiag[i]) < absVal(u)) {
				T xu = codiag[i] / u;
				lOffdiag[i] = xu;
				uDiag[i-1] = u;
				uOffdiag[i-1] = v;
				pivot[i-1] = 0.0;
				u = diag[i] - xu*v;
				if(i < n-1)
					v = codiag[i+1];
			} else {
				// Pivot
				T xu = u / codiag[i];
				lOffdiag[i] = xu;
				uDiag[i-1] = codiag[i];
				uOffdiag[i-1] = diag[i];
				if(i < n-1)
					pivot[i-1] = codiag[i+1];
				else
					pivot[i-1] = 0.0;
				u = v - xu*uOffdiag[i-1];
				v = (T) -1.0*xu*pivot[i-1];
			}
		}

		if(u == (T) 0.0)
			uDiag[n-1] = espmat(diag, codiag, n);
		else
			uDiag[n-1] = u;
	}

	static void forwardLU(
			T       *x,
			const T *uDiag,
			const T *lOffdiag,
			const T *codiag,
			int      n) {

		for(int i = 1; i < n; i++) {
			T u = x[i];
			if(uDiag[i-1] == codiag[i]) {
				u = x[i-1];
				x[i-1] = x[i];
			}
			x[i] = u - lOffdiag[i]*x[i-1];
		}

	}

	static void backwardLU(
			T       *x,
			T       *uDiag,
			T       *uOffdiag,
			T       *pivot,
			int      n) {

		if(uDiag[n-1] != (T) 0.0)
			x[n-1] /= uDiag[n-1];

		if(1 < n)
			x[n-2] = (x[n-2] - uOffdiag[n-2]*x[n-1])/uDiag[n-2];

		for(int i = n-3; 0 <= i; i--)
			x[i] = (x[i] - uOffdiag[i]*x[i+1] - pivot[i]*x[i+2])/uDiag[i];

	}

};

}

#endif