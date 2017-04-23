/*
 *  Created on: Dec 20, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef PSCRCL_MATRIX_CONTAINER
#define PSCRCL_MATRIX_CONTAINER

#include "common.h"

namespace pscrCL {

/*
 * A class which is used to store the vector presentations of the matrices
 * A_i and M_i. 
 */
class MatrixContainer {
public:
	MatrixContainer();
	MatrixContainer(
		const void       *aDiag,
		const void       *aOffdiag,
		const void       *mDiag,
		const void       *mOffdiag,
		int               n,
		int               ldf,
		const PscrCLMode &mode,
		bool              flip = false
   		);
	~MatrixContainer();
	MatrixContainer(const MatrixContainer &old);
	MatrixContainer& operator=(const MatrixContainer &a);
	
	const void* getPointer() const;
	int getSize() const;
	
	const void* getADiag() const;
	const void* getAOffdiag() const;
	const void* getMDiag() const;
	const void* getMOffdiag() const;
	
	int getN() const;
	int getLdf() const;
	const PscrCLMode getMode() const;
	bool isFlipped() const;
	bool isMTridiagonal() const;

private:	
	bool mTridiagonal;
	bool flipped;
	char *data;
	int n;
	int ldf;
	PscrCLMode mode;
};

}

#endif