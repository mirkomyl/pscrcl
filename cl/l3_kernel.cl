/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 *
 * 4-stage tridiagonal system solver: 3-stage cyclic reduction and parallel
 * cyclic reduction
 *
 * It is assumed that each tridiagonal linear system is of the form
 *   (A + (lambda1 + lambda2 + ch) * M) u = f,
 * where A is a tridiagonal matrix, M is a diagonal (or tridiagonal) matrix,
 * lambda1 is a eigenvalue from level 1 problem, lambda2 is a eigenvalue from
 * level 2 problem and ch is an additional coefficient.
 *
 * Stage A is performed only if the system (coefficient matrix and right-hand
 * side vector) does not fit into the local memory. The amount of allocated
 * local memory can be adjusted by parameter L3_LOCAL_MEM_SIZE. The coefficient
 * matrix and the right-hand side vector are stored into the global memory. The
 * local memory is used to share odd numbered rows between work-items. Each
 * reduction step requires a separate kernel launch and multiple work groups
 * can be mapped to one linear system. If multiple work groups are mapped to
 * one linear system, then the system is divided into segments, one for each
 * work group. The solution process itself is performed by dividing each
 * segment into multiple sections which are then processed sequentially. The
 * rows are permuted after each reduction step in the following manner:
 * (r=0)    (r=1)                           (r=2)
 * [ 1] <-> * 1*  -+                        | 1| -+
 * [ 2] <-> * 3*   |                        | 3|  |
 * [ 3] <-> * 5*   |                        | 5|  |
 * [ 4] <->	* 7*   |                        | 7|  |
 * [ 5] <->	[ 2]   |- Work group no. 0  <-> * 2*  |
 * [ 6] <->	[ 4]   |                    <-> * 6*  |
 * [ 7] <->	[ 6]   |                    <-> *10*  |
 * [_8] <-> [_8]  _|                    <-> *14*  |- Work group no. 0
 * [ 9] <->	* 9*   |                        | 9|  |
 * [10] <->	*11*   |                        |11|  |
 * [11] <->	*13*   |                        |13|  |
 * [12] <->	*15*   |- Work group no. 1      |15|  |
 * [13] <->	[10]   |                    <-> [ 4]  |
 * [14] <->	[12]   |                    <-> [ 6]  |
 * [15] <->	[14]   |                    <-> [12]  |
 * [16] <->	[16]  -+                    <-> [16] -+
 *
 * Stage B is performed only if the number of remaining even numbered rows is
 * greater that the used work group size. The coefficient matrix and the right
 * hand side vector are stored into the local memory. The remaining system is
 * divided into multiple sections which are then processed sequentially. The
 * rows are permuted before the first reduction step and after each reduction
 * step in the following manner:
 * (pre)    (r=0)    (r=1)
 * | 1| <-> [ 1]     | 1|  -+
 * | 2| <-> [ 3]     | 3|   |
 * | 3| <-> [ 5]     | 5|   |
 * | 4| <->	[ 7]     | 7|   |- First segment
 * | 5| <->	[ 2] <-> [ 2]   |
 * | 6| <->	[ 4] <-> [ 6]   |
 * | 7| <->	[ 6] <-> [10]   |
 * |_8| <-> [_8]_<->_[14]  _|
 * | 9| <->	[ 9]     | 9|   |
 * |10| <->	[11]     |11|   |
 * |11| <->	[13]     |13|   |
 * |12| <->	[15]     |15|   |- Second segment
 * |13| <->	[10] <-> [ 4]   |
 * |14| <->	[12] <-> [ 6]   |
 * |15| <->	[14] <-> [12]   |
 * |16| <->	[16] <-> [16]  -+
 *
 * Stage C is performed only if the number of remaining even numbered rows is
 * greater than L3_PCR_LIMIT. The coefficient matrix and the right-hand side
 * vector are stored into the local memory. The rows are permuted after each
 * reduction step in the following manner:

 *
 *
 *
 * ***** Kernels *****
 * l3_gen_glo_sys:      Forms the coefficient matrix to the global memory.
 * l3_a1:		        Performs stage A reduction step
 * l3_a2:		        Performs stage A back substitution step
 * l3_bcd_cpy_sys:	    Performs the steps B, C and D. Transfers the system 
 *                      from the global memory into the local memory.
 * l3_bcd_gen_sys:	    Performs the steps B, C and D. Forms the coefficient 
 *                      matrix to the local memory.
 *
 * If the stage A is enabled, then the coefficient matrices are first formed
 * into the global memory using l3_gen_glo_sys. Then the system size is reduced
 * using l3_a1. Once the system size is small enough, the remaining system is
 * copied into the local memory and solved using stages B, C and D. This is
 * achieved by calling the l3_bcd_cpy_sys. The remaining system is then solved
 * using l3_a2.
 *
 * If the stage A is disabled, then the whole solution process is handled by
 * l3_bcd_gen_sys.
 */

#define PCR_LIMIT MIN(L3_BCD_WG_SIZE, L3_PCR_LIMIT)

#if L3_LOCAL_MEM_SIZE < N3
#define L3_STAGE_A 1
#else
#define L3_STAGE_A 0
#endif

#if (L3_STAGE_A && L3_BCD_WG_SIZE < L3_A_WG_SIZE/2) || \
    (!L3_STAGE_A && L3_BCD_WG_SIZE < N3/2)
#define L3_STAGE_B 1
#else
#define L3_STAGE_B 0
#endif

#if L3_STAGE_B || \
	(L3_STAGE_A && PCR_LIMIT < L3_A_WG_SIZE) || \
	(!L3_STAGE_A && PCR_LIMIT < N3) 
#define L3_STAGE_C 1
#else
#define L3_STAGE_C 0
#endif

#if 0 < PCR_LIMIT
#define L3_STAGE_D 1
#else
#define L3_STAGE_D 0
#endif

#if L3_STAGE_D && 0 < (DIVBYPOW2(PCR_LIMIT, L3_PCR_STEPS))
#define L3_STAGE_E 1
#else
#define L3_STAGE_E 0
#endif

//
// Warnings
//

// Invalid work group sizes
#if L3_STAGE_A && L3_A_WG_SIZE < L3_BCD_WG_SIZE
#error "Invalid work group sizes."
#endif

#if L3_STAGE_B && (L3_LOCAL_MEM_SIZE % (4*L3_BCD_WG_SIZE) != 0)
#error "Invalid work group size or local mem. size"
#endif

/*
* Loads the eigenvalues from the global memory and computes the coefficient
* appearing in the level 3 linear system coefficient matrix.
* Arguments:
*   _lambda1	Eigenvalues for the level 1 problem
*   _lambda2	Eigenvalues for the level 2 problem
*   _idx1	    Eigenvalue index for level 1 problem. Set on -1 if not needed.
*   _idx2	    Eigenvalue index for level 2 problem
* Return value:
*   Requested coefficient
*/
_var get_lambda(
		const __global _var *_lambda1,
		const __global _var *_lambda2,
        int                  _idx1,
        int                  _idx2) {

#if L3_SHARED_ISOLATED_ACCESS

	ALLOC_LOCAL_MEM(work, tmp_work, 1);
	
	if(get_local_id(0) == 0) {
		_var tmp = _lambda2[_idx2];

#if MULTIBLE_LAMBDA
		if(1 < get_num_groups(0))
			tmp += _lambda1[_idx1];
#endif
		
		LSTORE(tmp, 0, work);
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	_var lambda = LLOAD(0, work);
	barrier(CLK_LOCAL_MEM_FENCE);

#else

	_var lambda;
#if MULTIBLE_LAMBDA
	if(1 < get_num_groups(0))
		lambda = _lambda1[_idx1] + _lambda2[_idx2];
	else
#endif
		lambda = _lambda2[_idx2];

#endif
	
	return lambda;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * Returns the coefficient matrix which corresponds to the given ids
 * Arguments:
 *   _matrix	A buffer which contains the coefficient matrices.
 *   _a			Returns a pointer to a buffer containing the lower off-diagonal
 *   			elements of the coefficient matrix.
 *   _b			Returns a pointer to a buffer containing the diagonal elements
 *   			of the coefficient matrix.
 *   _c			Returns a pointer to a buffer containing the upper off-diagonal
 *   			elements of the coefficient matrix.
 *   _id0		System index for level 1 problem
 *   _id1		System index for level 2 problems
 */
void get_coef_matrix(
		__global _var_t*   _matrix,
		_var_global_array* _a,
		_var_global_array* _b,
		_var_global_array* _c,
		int                _id1,
		int                _id2) {

	_var_global_array tmp = GLOBAL_ARRAY(_matrix, 3*L3_LDF1*L3_LDF2*L3_LDF3);

	// Lower off-diagonal
	*_a = SUB_GLOBAL_ARRAY(tmp, ((_id1*L3_LDF2+_id2)*3+0)*L3_LDF3);
	// Diagonal
	*_b = SUB_GLOBAL_ARRAY(tmp, ((_id1*L3_LDF2+_id2)*3+1)*L3_LDF3);
	// Upper off-diagonal
	*_c = SUB_GLOBAL_ARRAY(tmp, ((_id1*L3_LDF2+_id2)*3+2)*L3_LDF3);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * Returns the right-hand side vector which corresponds to the given ids
 * Arguments:
 *   _vector	A buffer which contains the right-hand side vectors.
 *   _id1		System index for level 1 problem
 *   _id2		System index for level 2 problems
 */
_var_global_array get_right_hand_side(
		__global _var_t* _vector,
		int              _id1,
		int              _id2) {

	_var_global_array tmp = GLOBAL_ARRAY(_vector, L3_LDF1*L3_LDF2*L3_LDF3);
	return SUB_GLOBAL_ARRAY(tmp, (_id1*L3_LDF2+_id2)*L3_LDF3);

}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#if L3_STAGE_A

/*
* Generates the coefficient matrix into the global memory. The vectors
* corresponding to the lower off-diagonals, diagonal and upper off-diagonal
* are stored sequentially.
* Arguments:
*   _tmp				The buffer in which the coefficient matrices are stored
*   					in.
*   _matrix_components	Data structure which defines the matrix A and M
*   _lambda1			Eigenvalues for the level 1 problem
*   _lambda2			Eigenvalues for the level 2 problems
*   _lambda1_stride		The index of the first level 1 eigenvalue
*   _lambda2_stride		The index of the first level 2 eigenvalue
*   _ch					Additional coefficient
* OpenCL parameters:
*   get_num_groups(0)	Number of level 1 problems
*   get_num_groups(1) 	Number of level 2 problems
*   get_num_groups(2)	Number of work groups allocated to each level 3
*   					problem. Should be the same as
*   					L3_GEN_GLO_SYS_WG_PER_SYS.
*/
__attribute__((reqd_work_group_size(L3_GEN_GLOBAL_SYS_WG_SIZE, 1, 1)))
__kernel void l3_gen_glo_sys(
		__global _var_t     *_tmp,
		__global _var_t     *_matrix_components,
		const __global _var *_lambda1,
		const __global _var *_lambda2,
        int                  _lambda1_stride,
        int                  _lambda2_stride,
        _var                 _ch
		) {

	const size_t local_id = get_local_id(0);
	const size_t gid0 = get_group_id(0);
	const size_t gid1 = get_group_id(1);

	// Get the coefficient
	const _var mul = get_lambda(_lambda1, _lambda2,
			_lambda1_stride+gid0, _lambda2_stride+gid1) + _ch;
	
	// Locate the matrices A and M
	_var_global_array a3_diag, a3_offdiag, m3_diag;
#if M3_TRIDIAG
	_var_global_array m3_offdiag;
	get_coef_matrix_components(
			_matrix_components, &a3_diag, &a3_offdiag, &m3_diag, &m3_offdiag);
#else
	get_coef_matrix_components(
			_matrix_components, &a3_diag, &a3_offdiag, &m3_diag, 0);
#endif

	// Get the location where the coefficient matrix is going to be stored
	_var_global_array a, b, c;
	get_coef_matrix(_tmp, &a, &b, &c, gid0, gid1);

#if L3_GEN_GLOBAL_SYS_WG_PER_SYS > 1

	// Calculate boundaries for the current segment
	const int begin =
			get_group_id(2) * L3_GEN_GLOBAL_SYS_WG_SIZE;
	const int jump = L3_GEN_GLOBAL_SYS_WG_PER_SYS * L3_GEN_GLOBAL_SYS_WG_SIZE;

	for(int l = begin+local_id; l < N3; l += jump) {

#else

	for(int l = local_id; l < N3; l += L3_GEN_GLOBAL_SYS_WG_SIZE) {

#endif

#if M3_TRIDIAG
		_var offdiag = GLOAD(l, a3_offdiag) + MUL(mul, GLOAD(l, m3_offdiag));
#else
		_var offdiag = GLOAD(l, a3_offdiag);
#endif
			
		GSTORE(offdiag, l, a);
		GSTORE(GLOAD(l, a3_diag) + MUL(mul, GLOAD(l, m3_diag)), l, b);
		if(0 < l)
			GSTORE(offdiag, l-1, c);

	}
	
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
* Performs a stage A reduction step.
* Arguments:
*   _f					Right hand side vector
*   _coef_matrix		Coefficient matrix
*   _r					Reductions step index
* OpenCL parameters:
*   get_num_groups(0)	Number of level 1 problems
*   get_num_groups(1) 	Number of level 2 problems
*   get_num_groups(2)   Number of work groups per system
*/
__attribute__((reqd_work_group_size(L3_A_WG_SIZE, 1, 1)))
__kernel void l3_a1(
		__global _var_t *_f,
		__global _var_t *_coef_matrix,
		int              _r
		) {
		
	const size_t local_id = get_local_id(0);
	const size_t half_local_id = local_id - L3_A_WG_SIZE/2;
	const size_t gid0 = get_group_id(0);
	const size_t gid1 = get_group_id(1);
		
	// Allocate local memory buffers
	ALLOC_LOCAL_MEM(loc_a, tmp_loc_a, L3_A_WG_SIZE);
	ALLOC_LOCAL_MEM(loc_b, tmp_loc_b, L3_A_WG_SIZE);
	ALLOC_LOCAL_MEM(loc_c, tmp_loc_c, L3_A_WG_SIZE);
	ALLOC_LOCAL_MEM(loc_f, tmp_loc_f, L3_A_WG_SIZE);
	
	// Get the right-hand side vector
	_var_global_array glo_f = get_right_hand_side(_f, gid0, gid1);

	// Get the coefficient matrix
	_var_global_array glo_a, glo_b, glo_c;
	get_coef_matrix(_coef_matrix, &glo_a, &glo_b, &glo_c, gid0, gid1);

	// Calculate the size and boundaries of the current segment
#if L3_PARALLEL_STAGE_A1
		// Calculates the boundaries for each segment
		const int segment_size = L3_LDF3 / NEXTPOW2(get_num_groups(2));
		const int begin = get_group_id(2) * segment_size;
		const int end = (get_group_id(2)+1) * segment_size;

		if(N3 <= begin)
			return;
#else
		const int segment_size = L3_LDF3;
		const int begin = 0;
		const int end = L3_LDF3;
#endif

#if L3_PARALLEL_STAGE_A1
	{ int r = _r;
#else
	// The main iteration is repeated as long as the number of remaining even
	// numbered rows is greater than the work group size
	for(int r = _r; L3_A_WG_SIZE <= DIVBYPOW2(segment_size, r); r++) {
#endif

		// The remaining system is divided into smaller sections which are then
		// processed in pairs starting from the last section pair.
		int first_part = end - (POW2(r-1)+1)*L3_A_WG_SIZE;
		int second_part = end - L3_A_WG_SIZE;

		const int part_count = DIVBYPOW2(segment_size, r)/L3_A_WG_SIZE;
		for(int i = part_count-1; 0 <= i; i--) {

			// If there is nothing to do, jump to the next sections pair
			if(DIVBYPOW2(N3-begin, r-1) <= 2*i*L3_A_WG_SIZE) {
				first_part -= POW2(r)*L3_A_WG_SIZE;
				second_part -= POW2(r)*L3_A_WG_SIZE;
				continue;
			}

			// Determine whether or not there exist a row-pair for this
			// work-item
			const int process =
					2*(i*L3_A_WG_SIZE + local_id) < DIVBYPOW2(N3-begin, r-1);

			// Odd numbered row above the current (even numbered) row and the
			// current (even numbered) row
			_var2  a, b, c, f;

			// Odd numbered row below the current (even numbered) row
			_var  l_a, l_b, l_c, l_f;

			if(process) {
				// Load rows
				if(local_id < L3_A_WG_SIZE/2) {
					a = GLOAD2(first_part/2+local_id, glo_a);
					b = GLOAD2(first_part/2+local_id, glo_b);
					c = GLOAD2(first_part/2+local_id, glo_c);
					f = GLOAD2(first_part/2+local_id, glo_f);
				} else {
					a = GLOAD2(second_part/2+half_local_id, glo_a);
					b = GLOAD2(second_part/2+half_local_id, glo_b);
					c = GLOAD2(second_part/2+half_local_id, glo_c);
					f = GLOAD2(second_part/2+half_local_id, glo_f);
				}

				if(local_id == L3_A_WG_SIZE - 1) {
#if !L3_PARALLEL_STAGE_A1
					// Reuse data from previous iteration
					l_a = LLOAD(0, loc_a);
					l_b = LLOAD(0, loc_b);
					l_c = LLOAD(0, loc_c);
					l_f = LLOAD(0, loc_f);
#else
					if(end < L3_LDF3) {
						l_a = GLOAD(first_part+POW2(r)*L3_A_WG_SIZE, glo_a);
						l_b = GLOAD(first_part+POW2(r)*L3_A_WG_SIZE, glo_b);
						l_c = GLOAD(first_part+POW2(r)*L3_A_WG_SIZE, glo_c);
						l_f = GLOAD(first_part+POW2(r)*L3_A_WG_SIZE, glo_f);
					}
#endif
				}

			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
			// Store odd numbered rows into the local memory
			if(process) {
				LSTORE(VAR2_S0(a), local_id, loc_a);
				LSTORE(VAR2_S0(b), local_id, loc_b);
				LSTORE(VAR2_S0(c), local_id, loc_c);
				LSTORE(VAR2_S0(f), local_id, loc_f);
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			if(process) {
				if(local_id < L3_A_WG_SIZE - 1) {
					l_a = LLOAD(local_id + 1, loc_a);
					l_b = LLOAD(local_id + 1, loc_b);
					l_c = LLOAD(local_id + 1, loc_c);
					l_f = LLOAD(local_id + 1, loc_f);
				}

				// Calculate coefficients. Make sure that the beta is
				// calculated correctly when handling the last row of the
				// system.
				const _var alpha = -1.0 * DIV(VAR2_S1(a), VAR2_S0(b));
				_var beta = 0.0;
				if(2*(i*L3_A_WG_SIZE+local_id)+2 < DIVBYPOW2(N3-begin, r-1)) {
					beta = -1.0 * DIV(VAR2_S1(c), l_b);
				} else {
					l_a = 0.0;
					l_b = 0.0;
					l_c = 0.0;
					l_f = 0.0;
				}
				
				// Process the even numbered rows. If the size of the remaining
				// system is an odd number, then one non-contributing row is
				// processed as a by-product. However, this has no effect on
				// the outcome or the performance.
				SET_VAR2_S1(MUL(alpha, VAR2_S0(a)), &a);
				SET_VAR2_S1(VAR2_S1(b) + MUL(alpha, VAR2_S0(c)) +
						MUL(beta, l_a), &b);
				SET_VAR2_S1(MUL(beta, l_c), &c);
				SET_VAR2_S1(VAR2_S1(f) + MUL(alpha, VAR2_S0(f)) +
						MUL(beta, l_f), &f);
						
			}

			if(process) {
	
				// Save the rows back to the global memory
				GSTORE(VAR2_S1(a), second_part + local_id, glo_a);
				GSTORE(VAR2_S1(b), second_part + local_id, glo_b);
				GSTORE(VAR2_S1(c), second_part + local_id, glo_c);
				GSTORE(VAR2_S1(f), second_part + local_id, glo_f);

				if(0 < local_id) { 
					GSTORE(VAR2_S0(a), first_part + local_id, glo_a);
					GSTORE(VAR2_S0(b), first_part + local_id, glo_b);
					GSTORE(VAR2_S0(c), first_part + local_id, glo_c);
					GSTORE(VAR2_S0(f), first_part + local_id, glo_f);
				}

			}

			barrier(CLK_GLOBAL_MEM_FENCE);

			first_part -= POW2(r)*L3_A_WG_SIZE;
			second_part -= POW2(r)*L3_A_WG_SIZE;
		}

	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
* Performs a stage A back substitution step.
* FIXME: -cl-opt-disable is required, fix.
* Arguments:
*   _f					Right hand side vector
*   _coef_matrix		Coefficient matrix
*   _r					Reductions step index
* OpenCL parameters:
*   get_num_groups(0)	Number of level 1 problems
*   get_num_groups(1) 	Number of level 2 problems
*   get_num_groups(2)   Number of work groups per system
*/
__attribute__((reqd_work_group_size(L3_A_WG_SIZE, 1, 1)))
__kernel void l3_a2(
		__global _var_t *_f,
		__global _var_t *_coef_matrix,
		int              _r
		) {

	const size_t local_id = get_local_id(0);
	const size_t half_local_id = local_id - L3_A_WG_SIZE/2;
	const size_t gid0 = get_group_id(0);
	const size_t gid1 = get_group_id(1);

	// Allocate local memory
	ALLOC_LOCAL_MEM(work, tmp_work, L3_A_WG_SIZE);

	// Get the right-hand side vector
	_var_global_array glo_f = get_right_hand_side(_f, gid0, gid1);

	// Get the coefficient matrix
	_var_global_array glo_a, glo_b, glo_c;
	get_coef_matrix(_coef_matrix, &glo_a, &glo_b, &glo_c, gid0, gid1);

	// Calculate the size and boundaries of the segment
#if L3_PARALLEL_STAGE_A2
		// Calculates the number of section pairs reserved for each work
		// group
		const int segment_size = L3_LDF3 / NEXTPOW2(get_num_groups(2));
		const int begin = get_group_id(2) * segment_size;
		const int end = (get_group_id(2)+1) * segment_size;

		if(N3 <= begin)
			return;
#else
		const int segment_size = L3_LDF3;
		const int begin = 0;
		const int end = L3_LDF3;
#endif

#if L3_PARALLEL_STAGE_A2
	{ int r = _r;
#else
	for(int r = _r; 0 <= r; r--) {
#endif

		// Calculate the size of intersection of the remaining linear system
		// and the segment
		const int intersection_size = DIVBYPOW2(min(N3-begin, segment_size), r);

		int first_part = begin + (POW2(r)-1)*L3_A_WG_SIZE;
		int second_part = begin + (POW2(r+1)-1)*L3_A_WG_SIZE;

		// The remaining system is divided into smaller sections which are then
		// processed in pairs starting from the first section pair.
		const int part_count = DIVBYPOW2(segment_size, r+1)/L3_A_WG_SIZE;
		for(int i = 0; i < part_count; i++) {

			// Nothing to do?
			if(intersection_size <= 2*i*L3_A_WG_SIZE)
				break;

			// Determine whether or not there exist a row-pair for this
			// work-item
			const int process =
					2*(i*L3_A_WG_SIZE + local_id) < intersection_size;

			// Even numbered row above the current (odd numbered) row and even
			// numbered row below the current (odd numbered) row.
			_var upper, lower;

			if(local_id == 0) {
				// For the first work-item of the work group, the uppermost
				// even numbered row resides in the upper segment and must be
				// accessed separately.
				if (begin == 0 && i == 0)
					upper = 0.0;
#if L3_PARALLEL_STAGE_A2
				else if(0 < begin && i == 0)
					upper = GLOAD(begin-1, glo_f);
#endif
				else
					// Reuse data from previous part
					upper = LLOAD(L3_A_WG_SIZE-1, work);
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
			// Load the lower even numbered row and store it into the local
			// memory
			if(process) {
				lower = GLOAD(second_part+local_id, glo_f);
				LSTORE(lower, local_id, work);
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			// Matrix elements
			_var a, b, c;
			
			_var middle;
			
			if(process) {

				if(0 < local_id)
					upper = LLOAD(local_id-1, work);

				// If this is the first part of the first segment, then the
				// first row must be handled separately
				if(begin == 0 && i == 0)
					a = local_id == 0 ? 0.0 :
						GLOAD(first_part+local_id, glo_a);
				else
					a = GLOAD(first_part+local_id, glo_a);

				b = GLOAD(first_part+local_id, glo_b);

				// Last row of the system must be handled separately
				if(2*(i*L3_A_WG_SIZE + local_id)+1 < DIVBYPOW2(N3-begin, r)) {
					c = GLOAD(first_part+local_id, glo_c);
				} else {
					c = 0;
					lower = 0;
				}

				// Solve the row
				middle = DIV(GLOAD(first_part+local_id, glo_f) -
						MUL(a,upper) - MUL(c,lower), b);
			}
			
			// No idea why this is neccessary but the the compiler freaks out 
			// otherwise
			barrier(CLK_LOCAL_MEM_FENCE); 
			
			if(process) {
	   
				// Save
				_var2 ret;
				SET_VAR2_S0(middle, &ret);
				SET_VAR2_S1(lower, &ret);
				
				if(local_id < L3_A_WG_SIZE/2)
					GSTORE2(ret, first_part/2 + local_id, glo_f);
				else
					GSTORE2(ret, second_part/2 + half_local_id, glo_f);
			}

			barrier(CLK_GLOBAL_MEM_FENCE);

			first_part += POW2(r+1)*L3_A_WG_SIZE;
			second_part += POW2(r+1)*L3_A_WG_SIZE;
		}
	}
}

#endif // L3_STAGE_A

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
* Performs stages B, C and D.
* Arguments:
* 	_a					Lower off-diagonal
* 	_b					Diagonal
* 	_c					Upper off-diagonal
*   _f					Right hand side vector
*   _n					System size
*   _k					log(n+1)
* OpenCL parameters:
*   get_num_groups(0)	Number of level 1 problems
*   get_num_groups(1) 	Number of level 2 problems
*/
void perform_local_solver(
		_var_local_array _a,
		_var_local_array _b,
		_var_local_array _c,
		_var_local_array _f,
		const int        _n,
		const int        _k) {

	const size_t local_id = get_local_id(0);
	const size_t half_local_id = local_id - L3_BCD_WG_SIZE/2;

	int r = 1;

#if L3_STAGE_B

	///////////////////////////////////////////////////////////////////////////
	/////////////////////////////  STAGE B BEGIN  /////////////////////////////
	///////////////////////////////////////////////////////////////////////////

	// Note: It is assumed that the rows are already permuted as required

	for(; r <= _k-1 && L3_BCD_WG_SIZE <= DIVBYPOW2(_n, r); r++) {
		int first_part =  (POW2(r-1)-1)*L3_BCD_WG_SIZE;
		int second_part = (POW2(r)-1)*L3_BCD_WG_SIZE;
		int third_part =  (3*POW2(r-1)-1)*L3_BCD_WG_SIZE;
		int fourth_part = (POW2(r+1)-1)*L3_BCD_WG_SIZE;

		// The remaining system is divided into sections which are then
		// processed sequentially in groups of four
		const int part_count = L3_LOCAL_MEM_SIZE/(POW2(r+1)*L3_BCD_WG_SIZE);
		for(int i = 0; i < part_count; i++) {

			if(DIVBYPOW2(_n, r-1) <= 4*i*L3_BCD_WG_SIZE)
				break;

			// Check if there exist a row in the second / fourth segment. The
			// permutation pattern requires that odd numbered row below the
			// current even numbered row is also processed. This is especially
			// important when the size of the remaining linear system is an
			// odd number.
			int process1 =
				2*(2*i*L3_BCD_WG_SIZE + local_id) < DIVBYPOW2(_n, r-1);
			int process2 =
				2*((2*i+1)*L3_BCD_WG_SIZE + local_id) < DIVBYPOW2(_n, r-1);

			_var u_a1, u_b1, u_c1, u_f1;
			_var u_a2, u_b2, u_c2, u_f2;
			
			_var m_a1, m_b1, m_c1, m_f1;
			_var m_a2, m_b2, m_c2, m_f2;
			
			_var l_a1, l_b1, l_c1, l_f1;
			_var l_a2, l_b2, l_c2, l_f2;
			
			if(process1) {
				u_a1 = LLOAD(first_part+local_id, _a);
				u_b1 = LLOAD(first_part+local_id, _b);
				u_c1 = LLOAD(first_part+local_id, _c);
				u_f1 = LLOAD(first_part+local_id, _f);

				m_a1 = LLOAD(second_part+local_id, _a);
				m_b1 = LLOAD(second_part+local_id, _b);
				m_c1 = LLOAD(second_part+local_id, _c);
				m_f1 = LLOAD(second_part+local_id, _f);
				
				// Deduce the address which contains the (odd numbered) row below
				// the current (even numbered) row
				int lower1_add = local_id < L3_BCD_WG_SIZE-1 ?
						first_part+local_id+1 : third_part;

				// Note: The last work-item of the work group accesses the
				// first row from the _third_ part
				l_a1 = LLOAD(lower1_add, _a);
				l_b1 = LLOAD(lower1_add, _b);
				l_c1 = LLOAD(lower1_add, _c);
				l_f1 = LLOAD(lower1_add, _f);
				
			const _var alpha1 = -1.0 * DIV(m_a1, u_b1);
				_var beta1 = 0.0;
				if(2*(2*i*L3_BCD_WG_SIZE + local_id)+2<DIVBYPOW2(_n,r-1)) {
					beta1 = -1.0 * DIV(m_c1, l_b1);
				} else {
					l_a1 = 0.0;
					l_b1 = 0.0;
					l_c1 = 0.0;
					l_f1 = 0.0;
				}	

				m_a1 = MUL(alpha1, u_a1);
				m_b1 = m_b1 + MUL(alpha1, u_c1) + MUL(beta1, l_a1);
				m_c1 = MUL(beta1, l_c1);
				m_f1 = m_f1 + MUL(alpha1, u_f1) + MUL(beta1, l_f1);
			}

			if(process2) {
				u_a2 = LLOAD(third_part+local_id, _a);
				u_b2 = LLOAD(third_part+local_id, _b);
				u_c2 = LLOAD(third_part+local_id, _c);
				u_f2 = LLOAD(third_part+local_id, _f);
			
				m_a2 = LLOAD(fourth_part+local_id, _a);
				m_b2 = LLOAD(fourth_part+local_id, _b);
				m_c2 = LLOAD(fourth_part+local_id, _c);
				m_f2 = LLOAD(fourth_part+local_id, _f);
				
			if(i != part_count-1) { 
					// Deduce the address which contains the (odd numbered) row
					// below the current (even numbered) row. The last work-
					// item of the work group accesses the first row from the
					// _first_ part of the _next_ group.
					int lower2_add = local_id < L3_BCD_WG_SIZE-1 ?
							third_part+local_id+1 :
							first_part+POW2(r+1)*L3_BCD_WG_SIZE;

					l_a2 = LLOAD(lower2_add, _a);
					l_b2 = LLOAD(lower2_add, _b);
					l_c2 = LLOAD(lower2_add, _c);
					l_f2 = LLOAD(lower2_add, _f);
				} else if(local_id < L3_BCD_WG_SIZE-1) {
					l_a2 = LLOAD(third_part+local_id+1, _a);
					l_b2 = LLOAD(third_part+local_id+1, _b);
					l_c2 = LLOAD(third_part+local_id+1, _c);
					l_f2 = LLOAD(third_part+local_id+1, _f);
				}
				
				_var alpha2 = -1.0 * DIV(m_a2, u_b2);
				_var beta2 = 0.0;
				if(2*((2*i+1)*L3_BCD_WG_SIZE + local_id)+2<DIVBYPOW2(_n,r-1)) {
					beta2 = -1.0 * DIV(m_c2, l_b2);
				} else {
					l_a2 = 0.0;
					l_b2 = 0.0;
					l_c2 = 0.0;
					l_f2 = 0.0;
				}					
				
				m_a2 = MUL(alpha2, u_a2);
				m_b2 = m_b2 + MUL(alpha2, u_c2) + MUL(beta2, l_a2);
				m_c2 = MUL(beta2, l_c2);
				m_f2 = m_f2 + MUL(alpha2, u_f2) + MUL(beta2, l_f2);
			}
			
			barrier(CLK_LOCAL_MEM_FENCE);

			// Determine the part where the results are stored in
			int locat = local_id & 0x01 ? fourth_part : second_part;

			if(process1) {
				LSTORE(m_a1, locat+(local_id/2), _a);
				LSTORE(m_b1, locat+(local_id/2), _b);
				LSTORE(m_c1, locat+(local_id/2), _c);
				LSTORE(m_f1, locat+(local_id/2), _f);
			}
			
			if(process2) {
				LSTORE(m_a2, locat+(L3_BCD_WG_SIZE/2) + (local_id/2), _a);
				LSTORE(m_b2, locat+(L3_BCD_WG_SIZE/2) + (local_id/2), _b);
				LSTORE(m_c2, locat+(L3_BCD_WG_SIZE/2) + (local_id/2), _c);
				LSTORE(m_f2, locat+(L3_BCD_WG_SIZE/2) + (local_id/2), _f);
			}

			barrier(CLK_LOCAL_MEM_FENCE);
			
			int add = POW2(r+1)*L3_BCD_WG_SIZE;

			first_part  += add;
			second_part += add;
			third_part  += add;
			fourth_part += add;
		}
	
	}

	///////////////////////////////////////////////////////////////////////////
	//////////////////////////////  STAGE B END  //////////////////////////////
	///////////////////////////////////////////////////////////////////////////

	// Stage C second part is the same as the stage B second part
	int second_part = (POW2(r-1)-1)*L3_BCD_WG_SIZE;

	// Stage C first part is the same as the stage B fourth part
	int first_part = (POW2(r)-1)*L3_BCD_WG_SIZE;

#else

	int first_part = 0;
	int second_part = _n/2;

#endif // end else

#if L3_STAGE_C

	///////////////////////////////////////////////////////////////////////////
	/////////////////////////////  STAGE C BEGIN  /////////////////////////////
	///////////////////////////////////////////////////////////////////////////

	// The main iteration is repeated as long as the size of the remaining
	// system is greater that given parameter
	for(; r <= _k-1 && PCR_LIMIT < DIVBYPOW2(_n, r-1); r++) {

		_var m_a, m_b, m_c, m_f;

		if(local_id < DIVBYPOW2(_n, r)) {
			_var u_a, u_b, u_c, u_f;
			_var l_a, l_b, l_c, l_f;

			u_a = LLOAD(second_part+local_id, _a);
			u_b = LLOAD(second_part+local_id, _b);
			u_c = LLOAD(second_part+local_id, _c);
			u_f = LLOAD(second_part+local_id, _f);

			m_a = LLOAD(first_part+local_id, _a);
			m_b = LLOAD(first_part+local_id, _b);
			m_c = LLOAD(first_part+local_id, _c);
			m_f = LLOAD(first_part+local_id, _f);
			
			_var alpha = -1.0 * DIV(m_a, u_b);

			_var beta = 0.0;
			l_a = l_b = l_c = l_f = 0.0;
			if(2*local_id+2 < DIVBYPOW2(_n, r-1)) {
				l_a = LLOAD(second_part+local_id+1, _a);
				l_b = LLOAD(second_part+local_id+1, _b);
				l_c = LLOAD(second_part+local_id+1, _c);
				l_f = LLOAD(second_part+local_id+1, _f);

				beta = -1.0 * DIV(m_c, l_b);
			}
		
			m_a = MUL(alpha, u_a);
			m_b = m_b + MUL(alpha, u_c) + MUL(beta, l_a);
			m_c = MUL(beta, l_c);
			m_f = m_f + MUL(alpha, u_f) + MUL(beta, l_f);
		
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(local_id < DIVBYPOW2(_n, r)) {
			if(PCR_LIMIT < DIVBYPOW2(_n, r)) {
				int pos = first_part + ((local_id+1) & 0x01) *
						DIVBYPOW2(_n, r+1) + (local_id/2);
				LSTORE(m_a, pos, _a);
				LSTORE(m_b, pos, _b);
				LSTORE(m_c, pos, _c);
				LSTORE(m_f, pos, _f);
			} else {
				// Prepare for the parallel cyclic reduction stage
				LSTORE(m_a, first_part + local_id, _a);
				LSTORE(m_b, first_part + local_id, _b);
				LSTORE(m_c, first_part + local_id, _c);
				LSTORE(m_f, first_part + local_id, _f);
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		second_part = first_part + DIVBYPOW2(_n, r+1);
	}

	///////////////////////////////////////////////////////////////////////////
	//////////////////////////////  STAGE C END  //////////////////////////////
	///////////////////////////////////////////////////////////////////////////

#else

	first_part = 0;

#endif

#if L3_STAGE_D

	second_part = first_part + DIVBYPOW2(_n, r-1);

	///////////////////////////////////////////////////////////////////////////
	/////////////////////////////  STAGE D BEGIN  /////////////////////////////
	///////////////////////////////////////////////////////////////////////////

	// Note: This stage uses the parallel cyclic reduction method

	// Calculate the size of the remaining system
	const int nn = DIVBYPOW2(_n, r-1);

	for(int l = 0; l < L3_PCR_STEPS && l <= _k-r; l++) {
		_var u_a, u_c, u_f;
		_var l_a, l_c, l_f;
		_var alpha, beta;

		u_a = u_c = u_f = 0.0;
		l_a = l_c = l_f = 0.0;
		alpha = beta = 0.0;

		if(local_id < nn) {

			// Locations of the rows above and below the current row
			const int u_ad = local_id-POW2(l);
			const int l_ad = local_id+POW2(l);
	
			// a_1 = 0
			if(0 < u_ad)
				u_a = LLOAD(first_part+u_ad, _a);
			
			if(0 <= u_ad) {
				_var m_a = LLOAD(first_part+local_id, _a);
				_var u_b = LLOAD(first_part+u_ad, _b);

				u_c = LLOAD(first_part+u_ad, _c);
				u_f = LLOAD(first_part+u_ad, _f);

				alpha = -1.0 * DIV(m_a, u_b);
			}

			// c_n = 0
			if(l_ad < nn-1)
				l_c = LLOAD(first_part+l_ad, _c);

			if(l_ad < nn) {
				_var m_c = LLOAD(first_part+local_id, _c);
				_var l_b = LLOAD(first_part+l_ad, _b);

				l_a = LLOAD(first_part+l_ad, _a);
				l_f = LLOAD(first_part+l_ad, _f);

				beta = -1.0 * DIV(m_c, l_b);
			} 
		
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(local_id < nn) {
			_var m_b = LLOAD(first_part+local_id, _b);
			_var m_f = LLOAD(first_part+local_id, _f);

			_var n_a = MUL(alpha, u_a);
			_var n_b = m_b + MUL(alpha, u_c) + MUL(beta, l_a);
			_var n_c = MUL(beta, l_c);
			_var n_f = m_f + MUL(alpha, u_f) + MUL(beta, l_f);

			LSTORE(n_a, first_part+local_id, _a);
			LSTORE(n_b, first_part+local_id, _b);
			LSTORE(n_c, first_part+local_id, _c);
			LSTORE(n_f, first_part+local_id, _f);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	

#if L3_STAGE_E

	///////////////////////////////////////////////////////////////////////////
	//////////////////////////////  STAGE E BEGIN  ////////////////////////////
	///////////////////////////////////////////////////////////////////////////

#define I(l) ((l)*POW2(L3_PCR_STEPS)+local_id)

	if(local_id < POW2(L3_PCR_STEPS) && I(1) < nn) {
	
		const int l0 = first_part+I(0);

		_var n_c = DIV(LLOAD(l0, _c), LLOAD(l0, _b));
		LSTORE(n_c, l0, _c);
		
		for(int l = 1, ll = first_part+I(1); I(l+1) < nn; ll = first_part+I(++l)) {
			n_c = DIV(
				LLOAD(ll, _c), 
				LLOAD(ll, _b) - MUL(LLOAD(ll, _a), n_c));
			LSTORE(n_c, ll, _c);
		}
		
		_var n_f = DIV(LLOAD(l0, _f), LLOAD(l0, _b));
		LSTORE(n_f, l0, _f);
		
		for(int l = 1, ll = first_part+I(1); I(l) < nn; ll = first_part+I(++l)) {
			_var o_a = LLOAD(ll, _a);
			n_f = DIV(
				LLOAD(ll, _f) - MUL(o_a, n_f), 
				LLOAD(ll, _b) - MUL(o_a ,LLOAD(I(l-1), _c)));
			LSTORE(n_f, ll, _f);
		}
		
		// ceil(nn/2^L3_PCR_STEPS)
		int l = DIVBYPOW2(nn, L3_PCR_STEPS) + 1;
		while(nn <= I(--l)); 
		_var n_x = LLOAD(first_part+I(l), _f);
		for(int ll = first_part+I(--l); first_part <= ll; ll = first_part+I(--l)) {
			n_x = LLOAD(ll, _f) - MUL(LLOAD(ll, _c), n_x);
			LSTORE(n_x, ll, _f);
		}
		
	} else if(local_id < POW2(L3_PCR_STEPS) && I(0) < nn) {
		_var m_f = LLOAD(first_part+I(0), _f);
		_var m_b = LLOAD(first_part+I(0), _b);

		_var n_f = DIV(m_f, m_b);

		LSTORE(n_f, first_part+I(0), _f);
	}
	
#undef I

	///////////////////////////////////////////////////////////////////////////
	//////////////////////////////  STAGE E END  //////////////////////////////
	///////////////////////////////////////////////////////////////////////////

#else
	
	// Final step
	if(local_id < nn) {
		_var m_f = LLOAD(first_part+local_id, _f);
		_var m_b = LLOAD(first_part+local_id, _b);

		_var n_f = DIV(m_f, m_b);

		LSTORE(n_f, first_part+local_id, _f);
	}

#endif
	
	barrier(CLK_LOCAL_MEM_FENCE);


	///////////////////////////////////////////////////////////////////////////
	//////////////////////////////  STAGE D END  //////////////////////////////
	///////////////////////////////////////////////////////////////////////////

#else
	if(local_id == 0) {
		_var m_f = LLOAD(first_part+local_id, _f);
		_var m_b = LLOAD(first_part+local_id, _b);

		_var n_f = DIV(m_f, m_b);

		LSTORE(n_f, first_part+local_id, _f);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	second_part = first_part + 1;
#endif // L3_STAGE_D

#if L3_STAGE_C

	///////////////////////////////////////////////////////////////////////////
	/////////////////////////////  STAGE C BEGIN  /////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	
	// Prepare for situations where only one stage B step is required
#if L3_STAGE_B
	if(L3_BCD_WG_SIZE < DIVBYPOW2(_n, r-2))
		second_part = (POW2(r-2)-1)*L3_BCD_WG_SIZE;
#endif

	// The main iteration is repeated as long as number of odd numbered rows
	// is greater or equal to the used work group size
	for(r-=2; DIVBYPOW2(_n, r+1) <= L3_BCD_WG_SIZE && 0 <= r; r--) {

		_var u_f, m_f, l_f;
		_var m_a, m_b, m_c;

		m_a = m_c = 0.0;
		u_f = l_f = 0.0;

		if(2*local_id < DIVBYPOW2(_n, r)) {
			if(0 < local_id) {
				u_f = LLOAD(first_part+local_id-1, _f);
				m_a = LLOAD(second_part+local_id, _a);
			}

			m_f = LLOAD(second_part+local_id, _f);
			m_b = LLOAD(second_part+local_id, _b);

			if(2*local_id+1 < DIVBYPOW2(_n, r)) {
				l_f = LLOAD(first_part+local_id, _f);
				m_c = LLOAD(second_part+local_id, _c);
			}

			m_f = DIV((m_f - MUL(m_a, u_f) - MUL(m_c, l_f)), m_b);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

#if L3_STAGE_B
		int pos;
		// Prepare for the stage B
		if(L3_BCD_WG_SIZE < DIVBYPOW2(_n, r))
			pos = local_id < L3_BCD_WG_SIZE/2 ?
					second_part+2*local_id : first_part+2*half_local_id;
		else
			pos = first_part+2*local_id;
#else
		int pos = first_part+2*local_id;
#endif

		if(2*local_id < DIVBYPOW2(_n, r))
			LSTORE(m_f, pos, _f);
		if(2*local_id+1 < DIVBYPOW2(_n, r))
			LSTORE(l_f, pos+1, _f);

		barrier(CLK_LOCAL_MEM_FENCE);

#if L3_STAGE_B
		if(L3_BCD_WG_SIZE < DIVBYPOW2(_n, r-1))
			second_part = (POW2(r-1)-1)*L3_BCD_WG_SIZE;
		else
			second_part = first_part + DIVBYPOW2(_n, r);
#else
		second_part = first_part + DIVBYPOW2(_n, r);
#endif

	}

	///////////////////////////////////////////////////////////////////////////
	//////////////////////////////  STAGE C END  //////////////////////////////
	///////////////////////////////////////////////////////////////////////////

#endif

#if L3_STAGE_B
	
	///////////////////////////////////////////////////////////////////////////
	/////////////////////////////  STAGE B BEGIN  /////////////////////////////
	///////////////////////////////////////////////////////////////////////////

	for(; 0 <= r; r--) {

		first_part  = L3_LOCAL_MEM_SIZE - (3*POW2(r)+1)*L3_BCD_WG_SIZE;
		second_part = L3_LOCAL_MEM_SIZE - (POW2(r+1)+1)*L3_BCD_WG_SIZE;
		int third_part  = L3_LOCAL_MEM_SIZE - (POW2(r)+1)*L3_BCD_WG_SIZE;
		int fourth_part = L3_LOCAL_MEM_SIZE - L3_BCD_WG_SIZE;

		for(int i = L3_LOCAL_MEM_SIZE/(POW2(r+2)*L3_BCD_WG_SIZE)-1;
				0 <= i; i--) {

			if(DIVBYPOW2(_n, r) <= 4*i*L3_BCD_WG_SIZE)
				continue;

			_var upper1, middle1, lower1;
			_var upper2, middle2, lower2;

			// Check if there exist a row in the third segment
			int process2 =
				2*((2*i+1)*L3_BCD_WG_SIZE + local_id) < DIVBYPOW2(_n, r);

			// The first row of the subsystem must be processed separately.
			if(i == 0) {
				upper1 = local_id == 0 ?
						0.0 : LLOAD(second_part+local_id-1, _f);
			} else {
				// The first work-item accesses the fourth part from the next
				// group.
				int upper1_addr = 0 < local_id ? second_part+local_id-1 :
						fourth_part+(1-POW2(r+2))*L3_BCD_WG_SIZE-1;

				upper1 =
						LLOAD(upper1_addr, _f);
			}

			// The last row of the subsystem system must be processed
			// separately.
			lower1 = LLOAD(second_part+local_id, _f);

			_var a_1 = LLOAD(first_part+local_id, _a);
			_var b_1 = LLOAD(first_part+local_id, _b);
			_var c_1 = LLOAD(first_part+local_id, _c);

			middle1 = LLOAD(first_part+local_id, _f);

			middle1 = DIV(middle1 - MUL(a_1, upper1) - MUL(c_1, lower1), b_1);

			if(process2) {

				// The first work-item accesses the second part
				int upper2_addr = 0 < local_id ? fourth_part+local_id-1 :
						second_part+L3_BCD_WG_SIZE-1;

				upper2 = LLOAD(upper2_addr, _f);
		
				_var c_2;
				lower2 = c_2 = 0.0;

				// The last row of the subsystem must be processed separately.
				if(2*((2*i+1)*L3_BCD_WG_SIZE + local_id) + 1 <
						DIVBYPOW2(_n, r)) {
					lower2 = LLOAD(fourth_part+local_id, _f);
					c_2 = LLOAD(third_part+local_id, _c);
				}

				_var a_2 = LLOAD(third_part+local_id, _a);
				_var b_2 = LLOAD(third_part+local_id, _b);

				middle2 = LLOAD(third_part+local_id, _f);

				middle2 = DIV(middle2 -
						MUL(a_2, upper2) - MUL(c_2, lower2), b_2);
			}
	
			barrier(CLK_LOCAL_MEM_FENCE);
			
			_var2 ret1;
			SET_VAR2_S0(middle1, &ret1);
			SET_VAR2_S1(lower1, &ret1);

			if(local_id < L3_BCD_WG_SIZE/2)
				LSTORE2(ret1, first_part/2 + local_id, _f);
			else
				LSTORE2(ret1, second_part/2 + half_local_id, _f);

			if(process2) {
				_var2 ret2;
				SET_VAR2_S0(middle2, &ret2);
				SET_VAR2_S1(lower2, &ret2);

				if(local_id < L3_BCD_WG_SIZE/2)
					LSTORE2(ret2, third_part/2 + local_id, _f);
				else
					LSTORE2(ret2, fourth_part/2 + half_local_id, _f);
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			int add = POW2(r+2)*L3_BCD_WG_SIZE;
			first_part -= add;
			second_part -= add;
			third_part -= add;
			fourth_part -= add; 
		}
	}
	
	///////////////////////////////////////////////////////////////////////////
	//////////////////////////////  STAGE B END  //////////////////////////////
	///////////////////////////////////////////////////////////////////////////

#endif // L3_STAGE_B

}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#if L3_STAGE_A

/*
* Copies the data into the local memory and then performs stages B, C and D.
* Arguments:
* 	_f					The buffer in which the right-hand side vector are
* 						stored in.
* 	_coef_matrix		The buffer in which the coefficient matrices are stored
*   					in.
*   _r 					Reduction index.
* OpenCL parameters:
*   get_num_groups(0)	Number of level 1 problems
*   get_num_groups(1) 	Number of level 2 problems
*/
__attribute__((reqd_work_group_size(L3_BCD_WG_SIZE, 1, 1)))
__kernel void l3_bcd_cpy_sys(
		__global _var_t *_f,
		__global _var_t *_coef_matrix,
		int              _r ) {

	const size_t local_id = get_local_id(0);
	const size_t gid0 = get_group_id(0);
	const size_t gid1 = get_group_id(1);
	
	// Allocate local memory
	ALLOC_LOCAL_MEM(loc_a, tmp_loc_a, L3_LOCAL_MEM_SIZE);
	ALLOC_LOCAL_MEM(loc_b, tmp_loc_b, L3_LOCAL_MEM_SIZE);
	ALLOC_LOCAL_MEM(loc_c, tmp_loc_c, L3_LOCAL_MEM_SIZE);
	ALLOC_LOCAL_MEM(loc_f, tmp_loc_f, L3_LOCAL_MEM_SIZE);
	
	// Prepare global memory buffer handles
	_var_global_array glo_f = get_right_hand_side(_f, gid0, gid1);

	_var_global_array glo_a, glo_b, glo_c;
	get_coef_matrix(_coef_matrix, &glo_a, &glo_b, &glo_c, gid0, gid1);

	// Calculate system size
	const int k = K3-_r+1;
	const int n = DIVBYPOW2(N3, _r-1);

#if L3_STAGE_B
	for(int i = local_id; 2*i < n; i += L3_BCD_WG_SIZE) {
		int g_part_id = i/(L3_A_WG_SIZE/2);
		int g_line_id = i%(L3_A_WG_SIZE/2);

		int g_idx = ((POW2(_r-1)-1)+g_part_id*POW2(_r-1))*L3_A_WG_SIZE/2
				+ g_line_id;

		_var2 va = GLOAD2(g_idx, glo_a);
		_var2 vb = GLOAD2(g_idx, glo_b);
		_var2 vc = GLOAD2(g_idx, glo_c);
		_var2 vf = GLOAD2(g_idx, glo_f);

		int l_part_id = i/L3_BCD_WG_SIZE;
		int l_line_id = local_id;

		int l_idx0 = 2*l_part_id*L3_BCD_WG_SIZE+l_line_id;
		int l_idx1 = (2*l_part_id+1)*L3_BCD_WG_SIZE+l_line_id;
		
		LSTORE(VAR2_S0(va), l_idx0, loc_a);
		LSTORE(VAR2_S0(vb), l_idx0, loc_b);
		LSTORE(VAR2_S0(vc), l_idx0, loc_c);
		LSTORE(VAR2_S0(vf), l_idx0, loc_f);

		LSTORE(VAR2_S1(va), l_idx1, loc_a);
		LSTORE(VAR2_S1(vb), l_idx1, loc_b);
		LSTORE(VAR2_S1(vc), l_idx1, loc_c);
		LSTORE(VAR2_S1(vf), l_idx1, loc_f);
	}
#elif L3_STAGE_C
	if(2*local_id < n) {

		int idx = (POW2(_r-1)-1)*L3_A_WG_SIZE/2+local_id;
		
		_var2 va = GLOAD2(idx, glo_a);
		_var2 vb = GLOAD2(idx, glo_b);
		_var2 vc = GLOAD2(idx, glo_c);
		_var2 vf = GLOAD2(idx, glo_f);

		LSTORE(VAR2_S0(va), n/2+local_id, loc_a);
		LSTORE(VAR2_S0(vb), n/2+local_id, loc_b);
		LSTORE(VAR2_S0(vc), n/2+local_id, loc_c);
		LSTORE(VAR2_S0(vf), n/2+local_id, loc_f);
		
		if(local_id < n/2) {
			LSTORE(VAR2_S1(va), local_id, loc_a);
			LSTORE(VAR2_S1(vb), local_id, loc_b);
			LSTORE(VAR2_S1(vc), local_id, loc_c);
			LSTORE(VAR2_S1(vf), local_id, loc_f);
		}
	}
#else
	if(local_id < n) {
		int idx = (POW2(_r-1)-1)*L3_A_WG_SIZE+local_id;

		_var va = GLOAD(idx, glo_a);
		_var vb = GLOAD(idx, glo_b);
		_var vc = GLOAD(idx, glo_c);
		_var vf = GLOAD(idx, glo_f);

		LSTORE(va, local_id, loc_a);
		LSTORE(vb, local_id, loc_b);
		LSTORE(vc, local_id, loc_c);
		LSTORE(vf, local_id, loc_f);
	}
#endif

	// Local stages B, C, D
	barrier(CLK_LOCAL_MEM_FENCE);
	perform_local_solver(loc_a, loc_b, loc_c, loc_f, n, k);
	barrier(CLK_LOCAL_MEM_FENCE);

	// Copy the subsystem back to the global memory
	for(int i = local_id; i < n; i += L3_BCD_WG_SIZE) {
		int part_id = i/L3_A_WG_SIZE;
		int line_id = i%L3_A_WG_SIZE;

		int idx = ((POW2(_r-1)-1)+part_id*POW2(_r-1))*L3_A_WG_SIZE+line_id;

		GSTORE(LLOAD(i, loc_f), idx, glo_f);
	}
}

#else

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
* Generates the coefficient matrix into the local memory and then performs
* stages B, C and D.
* Arguments:
*   _f					The buffer in which the right-hand side vectors are
*   					stored in.
*   _matrix_components	Data structure which defines the matrix A and M
*   _lambda1			Eigenvalues for the level 1 problem
*   _lambda2			Eigenvalues for the level 2 problems
*   _lambda1_stride		The index of the first level 1 eigenvalue
*   _lambda2_stride		The index of the first level 2 eigenvalue
*   _ch					Additional coefficient
* OpenCL parameters:
*   get_num_groups(0)	Number of level 1 problems
*   get_num_groups(1) 	Number of level 2 problems
*/
__attribute__((reqd_work_group_size(L3_BCD_WG_SIZE, 1, 1)))
__kernel void l3_bcd_gen_sys(
		__global _var_t     *_f,
		__global _var_t     *_matrix_components,
		const __global _var *_lambda1,
		const __global _var *_lambda2,
        int                  _lambda1_stride,
        int                  _lambda2_stride,
        _var                 _ch) {

	const size_t local_id = get_local_id(0);
	const size_t gid0 = get_group_id(0);
	const size_t gid1 = get_group_id(1);

	const _var mul = get_lambda(_lambda1, _lambda2,
			_lambda1_stride+gid0, _lambda2_stride+gid1) + _ch;
	
	ALLOC_LOCAL_MEM(a, tmp_a, L3_LOCAL_MEM_SIZE);
	ALLOC_LOCAL_MEM(b, tmp_b, L3_LOCAL_MEM_SIZE);
	ALLOC_LOCAL_MEM(c, tmp_c, L3_LOCAL_MEM_SIZE);
	ALLOC_LOCAL_MEM(f, tmp_f, L3_LOCAL_MEM_SIZE);

	_var_global_array glo_f = get_right_hand_side(_f, gid0, gid1);

	_var_global_array a3_diag, a3_offdiag, m3_diag;
#if M3_TRIDIAG
	_var_global_array m3_offdiag;
	get_coef_matrix_components(
			_matrix_components, &a3_diag, &a3_offdiag, &m3_diag, &m3_offdiag);
#else
	get_coef_matrix_components(
			_matrix_components, &a3_diag, &a3_offdiag, &m3_diag, 0);
#endif

#if L3_STAGE_B

	for(int i = 0; i*2*L3_BCD_WG_SIZE < N3; i++) {

		const int l = i*L3_BCD_WG_SIZE+local_id;

		_var2 ff = GLOAD2(l, glo_f);

		_var2 diag = GLOAD2(l, a3_diag) + MUL2(mul, GLOAD2(l, m3_diag));

#if M3_TRIDIAG
		_var2 offdiag = GLOAD2(l, a3_offdiag) +
				MUL2(mul, GLOAD2(l, m3_offdiag));
#else
		_var2 offdiag = GLOAD2(l, a3_offdiag);
#endif

		const int l0 = i*2*L3_BCD_WG_SIZE+local_id;
		const int l1 = (i*2+1)*L3_BCD_WG_SIZE+local_id;

		LSTORE(VAR2_S0(ff), l0, f);
		LSTORE(VAR2_S1(ff), l1, f);

		LSTORE(VAR2_S0(offdiag), l0, a);
		LSTORE(VAR2_S1(offdiag), l1, a);

		LSTORE(VAR2_S0(diag), l0, b);
		LSTORE(VAR2_S1(diag), l1, b);

		LSTORE(VAR2_S1(offdiag), l0, c);

		if(0 < l) {
			int jump = local_id == 0 ?
					(i*2)*L3_BCD_WG_SIZE-1 : l1-1;
			LSTORE(VAR2_S0(offdiag), jump, c);
		}
	}
	
#elif L3_STAGE_C

	if(2*local_id < N3) {

		_var2 ff = GLOAD2(local_id, glo_f);

		_var2 diag = GLOAD2(local_id, a3_diag) +
				MUL2(mul, GLOAD2(local_id, m3_diag));

#if M3_TRIDIAG
		_var2 offdiag = GLOAD2(local_id, a3_offdiag) +
				MUL2(mul, GLOAD2(local_id, m3_offdiag));
#else
		_var2 offdiag = GLOAD2(local_id, a3_offdiag);
#endif

		const int l0 = local_id;
		const int l1 = N3/2+local_id;

		if(local_id < N3/2) {
			LSTORE(VAR2_S1(ff), l0, f);
			LSTORE(VAR2_S1(offdiag), l0, a);
			LSTORE(VAR2_S1(diag), l0, b);
		}

		if(0 < local_id && local_id-1 < N3/2)
			LSTORE(VAR2_S0(offdiag), l0-1, c);

		LSTORE(VAR2_S0(ff), l1, f);
		LSTORE(VAR2_S0(offdiag), l1, a);
		LSTORE(VAR2_S0(diag), l1, b);

		LSTORE(VAR2_S1(offdiag), l1, c);
	}

#else

	// If only the stage C is used
	if(local_id < N3) {
		LSTORE(GLOAD(local_id, glo_f), local_id, f);

		_var diag = GLOAD(local_id, a3_diag) +
				MUL(mul, GLOAD(local_id, m3_diag));
#if M3_TRIDIAG
		_var offdiag = GLOAD(local_id, a3_offdiag) +
				MUL(mul, GLOAD(local_id, m3_offdiag));
#else
		_var offdiag = GLOAD(local_id, a3_offdiag);
#endif

		LSTORE(offdiag, local_id, a);
		LSTORE(diag, local_id, b);
		if(1 <= local_id)
			LSTORE(offdiag, local_id-1, c);
	}

#endif

	barrier(CLK_LOCAL_MEM_FENCE);
	perform_local_solver(a, b, c, f, N3, K3);
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int l = local_id; l < N3; l += L3_BCD_WG_SIZE)
		GSTORE(LLOAD(l, f), l, glo_f);
}


#endif
