/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 * 
 * 
 * A GPU radix-4 PSCR implementation - GPU-side code. 
 * 
 * This same file is used for the level 1 and level 2 problems. The level is 
 * specified by the define LEVEL2. There are eight kernels, four for each
 * stage:
 *  lx_stage_11		This kernel forms the right hand side vectors for the level
 * 					2 / 3 subproblems.  
 *  lx_stage_12a		This kernel is the first step of the recursive vector 
 * 					summation. It multiplies each level 2 / 3 subproblems 
 * 					solution vector with the corresponding eigenvector component
 * 					and calculates the partial sums.
 *  lx_stage_12b		This kernel is the second step of the recursive vector 
 * 					summation. It is called multiple times until the recursive
 * 					summation is ready.
 *  lx_stage_12c		This kernel is the third step of the recursive summation.
 * 					It updates the right hand side vector accordingly.
 * 
 *  lx_stage_21		This kernel is analogous to the kernel lx_stage_11.
 *  lx_stage_22a		This kernel is analogous to the kernel lx_stage_12a.
 *  lx_stage_22b		This kernel is analogous to the kernel lx_stage_12b.
 *  lx_stage_22c		This kernel is analogous to the kernel lx_stage_12c.
 * 
 * All kernels are of modular design and can be used in a multi-GPU 
 * configuration with minimal modifications. Each kernel has it own guiding
 * information structure which is used to map each work group to a specific 
 * task.
 */

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * LEVEL2
 * N1
 * N2
 * N3
 * L1_LDF1
 * L1_LDF2
 * L1_LDF3
 * L2_LDF1
 * L2_LDF2
 * L2_LDF3
 * L3_LDF1
 * L3_LDF2
 * L3_LDF3
 * PN1							Size of the section of the level 1 problem 
 * 								mapped to this OpenCL device
 * PN2							Size of the section of the level 2 problem
 * 								mapped to this OpenCL device 
 * PARALLEL_PART_TOTAL_COUNT
 * PARALLEL_PART_COUNT
 * PARALLEL_PART_SIZE 
 */

#if LEVEL2
#define LX_LDF3 L2_LDF3
#else
#define LX_LDF3 L1_LDF3
#endif

#define LX_STAGE_Y2A_MAX_SUM_SIZE POW2(LX_STAGE_Y2A_MAX_SUM_SIZE_EXP)
#define LX_STAGE_Y2B_MAX_SUM_SIZE POW2(LX_STAGE_Y2B_MAX_SUM_SIZE_EXP)

#define TMP_PER_L2 PNX

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
 
/*
 * Lookup macro for the eigenvector component data structure. 
 * Arguments:
 *  buff		A pointer to the data structure
 *  vec_id		Index of the requested eigenvector
 *  comp_id		Index of the requested eigenvector component
 */
#define GET_EIGEN(buff, vec_id, comp_id) (buff)[(vec_id)*5+comp_id]
 
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * Work group specific guiding information for the first half of the 
 * first/second stage. 
 *  0 * sizeof(cl_int) = Location of the shared guiding information
 *  1 * sizeof(cl_int) = Subproblem index
 */
typedef __global int _stagey1_guide1;

/*
 * Corresponding lookup macro. 
 * Argument:
 *  buff	A pointer to the data structure
 *  gid		Work group index
 *  idx		Data element index {0, 1}
 *  i 		Recursion index
 */
#define GET_STAGEY1_GUIDE1_ELEM(buff, gid, idx, i) (buff)[2*((i-1)*PNX+gid)+idx]
 
// ///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
 
/*
 * Shared guiding information for the first half of the first/second stage. 
 *  0 * sizeof(cl_int) = Location of the first element
 *  1 * sizeof(cl_int) = Location of the second element
 *  2 * sizeof(cl_int) = Location of the third element
 *  3 * sizeof(cl_int) = Location of the first eigenvector
 *  4 * sizeof(cl_int) = Location of the upper element
 *  5 * sizeof(cl_int) = Location of the lower element
 * 
 * It the index of the element is negative, then is is assumed that the element
 * does not exist.
 */
typedef __global int _stageY1_guide2;

/*
 * Corresponding lookup macro
 * Arguments:
 *  buff	A pointer tot the data structure
 *  eid		Index of the requested guiding information
 *  idx		Data element index {0, 1, 2, 3}
 */
#define GET_STAGEY1_GUIDE2_ELEM(buff, eid, idx) (buff)[8*eid+idx]

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * Work group specific guiding information for the second half of the first 
 * stage, step A.
 *  0 * sizeof(cl_int) = Location of the first vector
 *  1 * sizeof(cl_int) = Location of the last vector + 1
 *  2 * sizeof(cl_int) = Location of the first eigenvector
 *  3 * sizeof(cl_int) = Upper (0) / Lower (1) sum
 */
typedef __global int _stage12a_guide;

/*
 * Corresponding lookup macro
 * Arguments:
 *   buff	A pointer to the data structure
 *   gid	Work group index
 *   idx	Data element index {0, 1, 2, 3}
 *   i		Recursion index
 */
#define GET_STAGE12A_GUIDE_ELEM(buff, gid, idx, i) \
	(buff)[4*(2*(i-1)*(PNX/2)+gid)+idx]

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * Work group specific guiding information for the second half of the first 
 * stage, step B.
 *  0 * sizeof(cl_int) = Location of the first vector
 *  1 * sizeof(cl_int) = Location of the first vector + 1
 */
typedef __global int _stagey2b_guide;
#define GET_STAGEY2B_GUIDE_ELEM(buff, gid, idx, i) \
	(buff)[2*(3*(i-1)*(PNX/2)+gid)+idx]

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
	
/*
 * Work group specific guiding information for the second half of the first 
 * stage, step C.
 *  0 * sizeof(cl_int) = Location of the updateable vector
 *  1 * sizeof(cl_int) = Location of the upper sum
 *  2 * sizeof(cl_int) = Location of the lower sum
 */
typedef __global int _stage12c_guide;
#define GET_STAGE12C_GUIDE_ELEM(buff, gid, idx, i) \
	(buff)[3*((i-1)*(PNX/2)+gid)+idx]
 
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * Work group specific guiding information for the second half of the first 
 * stage, step A.
 *  0 * sizeof(cl_int) = Location of the first vector
 *  1 * sizeof(cl_int) = Location of the last vector + 1
 *  2 * sizeof(cl_int) = Location of the first eigenvector
 *  3 * sizeof(cl_int) = Index of the eigenvector component
 */
typedef __global int _stage22a_guide;

/*
 * Corresponding lookup macro
 * Arguments:
 *   buff	A pointer to the data structure
 *   gid	Work group index
 *   idx	Data element index {0, 1, 2, 3}
 *   i		Recursion index
 */
#define GET_STAGE22A_GUIDE_ELEM(buff, gid, idx, i) \
	(buff)[4*(3*i*(PNX/2)+gid)+idx]

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
	
/*
 * Work group specific guiding information for the second half of the first 
 * stage, step C.
 *  0 * sizeof(cl_int) = Location of the updateable vector
 *  1 * sizeof(cl_int) = Location of the sum
 */
typedef __global int _stage22c_guide;
#define GET_STAGE22C_GUIDE_ELEM(buff, gid, idx, i) \
	(buff)[2*(i*PNX+gid)+idx]
	
	
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*
 * A helper function which is used when multiplying a vector (_vec) with a 
 * tridiagonal matrix. It is assumed that each work-item has already loaded the 
 * element number _i. The elements are then shared between the work-items using 
 * the local memory.
 * Arguments:
 *  _u			Element number _i - 1
 *  _m			Element number _i
 *  _l			Element number _i + 1
 *  _vec		Vector
 *  _i			Element index
 *  _local_id	Index number of the work-item
 *  _wg_size	Work groups size
 * 	_work		Local memory buffer, minimum size is _work_item * sizeof(_var)
 */
void l3_vector_load_helper(
		_var                *_u, 
		const _var          *_m, 
		_var                *_l, 
		__global const _var *_vec,
		int			    	_i,
		int                 _local_id, 
		int                 _wg_size,
		_var_local_array    _work) {
	  
#if LX_VECTOR_LOAD_HELPER
	  
	barrier(CLK_LOCAL_MEM_FENCE);
	LSTORE(*_m, _local_id, _work);
	barrier(CLK_LOCAL_MEM_FENCE);
	*_u = 0.0;
	if(0 <= _i-1)
		*_u = _local_id == 0 ?
			_vec[_i-1] : LLOAD(_local_id-1, _work);
	*_l = 0.0;
	if(_i+1 < N3)
		*_l = _local_id == _wg_size-1 ? 
			_vec[_i+1] : LLOAD(_local_id+1, _work);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
#else

	*_u = 0.0;
	if(0 <= _i-1)
		*_u = _vec[_i-1];
	*_l = 0.0;
	if(_i+1 < N3)
		*_l = _vec[_i+1];

#endif
}

/*
 * A helper function which is used when multiplying a vector with a tridiagonal 
 * matrix. This function will load the off-diagonal elements corresponding to 
 * the given row-index _i.
 * Arguments:
 *  _a			Left off-diagonal
 *  _c			Right off-diagonal
 *  _offdiag	Vector presentation of the off-diagonals
 *  _i			Element index
 *  _local_id	Index number of the work-item
 *  _wg_size	Work groups size
 * 	_work		Local memory buffer, minimum size is _wg_size * sizeof(_var)
 */
void l3_matrix_load_helper(
		_var              *_a, 
		_var              *_c, 
		_var_global_array  _offdiag, 
		int				   _i, 
		int                _local_id, 
		int                _wg_size,
		_var_local_array   _work) {

#if LX_MATRIX_LOAD_HELPER
	
	*_a = 0 < _i ? GLOAD(_i, _offdiag) : 0.0;
	barrier(CLK_LOCAL_MEM_FENCE);
	LSTORE(*_a, _local_id, _work);
	barrier(CLK_LOCAL_MEM_FENCE);
	if(_i+1 < N3)
		*_c = _local_id == _wg_size-1 ?
			GLOAD(_i+1, _offdiag) : LLOAD(_local_id+1, _work);
	barrier(CLK_LOCAL_MEM_FENCE);
	
#else
	
	*_a = *_c = 0.0;
	if(0 < _i)
		*_a = GLOAD(_i, _offdiag);
	if(_i+1 < N3)
		*_c = GLOAD(_i+1, _offdiag);

#endif
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


/* The first half of the first stage. This kernel generates the right-hand
 * side vectors for the level X-1 subproblems. 
 * Arguments:
 *  _f			Level X right-hand side vector
 *  _g			g-vector, i.e., the level X-1 right-hand side vectors
 *  _guide1		Guiding information, i.e., what each work group should do
 *  _guide2		Shared guiding information
 *  _eigen		Eigenvector components
 *  _i			Recursion index
 * Work groups:
 *  get_num_groups(0)	Number of level X-1 subproblems
 *  get_num_groups(1) 	If LEVEL2 is enabled, then this is the number of level
 * 						2 subproblem. Otherwise, it is equal to N2.
 *  get_num_groups(2)   Number of work groups per N3-vector
 * 						  = LX_STAGE_11_WG_PER_VECTOR
 */
__attribute__((reqd_work_group_size(LX_STAGE_11_WG_SIZE, 1, 1)))
__kernel void lx_stage_11(
		__global _var   *_f,
#if LEVEL2
		__global _var_t *_g,
#else
		__global _var   *_g,
#endif
		_stagey1_guide1 *_guide1,
		_stageY1_guide2 *_guide2,
		__global _var   *_eigen,
		int              _i
		) {

	// Work-item index and work group size
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);

	// Primary work group index. Specifies which level X-1 subproblem belongs 
	// to this work group.
	const int gid0 = get_group_id(0);
	
	// Secondary work group index. If LEVEL2 is enabled, then this index 
	// specifies which level 2 problem belongs to this work group. If LEVEL2
	// is disabled, then this index specifies which level N2*N3-vector belongs 
	// to this work group.
	const int gid1 = get_group_id(1);
	
#if 1 < LX_STAGE_11_WG_PER_VECTOR
	// This specifies which section of the N3-vector belongs to this work group
	const int gid2 = get_group_id(2);
#else
	const int gid2 = 0;
#endif
	
#if LX_SHARED_ISOLATED_ACCESS

	__local int iwork[6];
	ALLOC_LOCAL_MEM(fwork, fwork_tmp, 3);

	// Load the guiding information
	if(local_id < 2)
		iwork[local_id] = 
			GET_STAGEY1_GUIDE1_ELEM(_guide1, gid0, local_id, _i);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Location of the shared guiding information
	const int guide_id = iwork[0];
	
	// Index of the eigenvector
	const int s_id = iwork[1];
	
	// Load the shared guiding information
	if(local_id < 4)
		iwork[2+local_id] = 
			GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, local_id);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Positions of the elements
	const int elem1_id = iwork[2];
	const int elem2_id = iwork[3];
	const int elem3_id = iwork[4];
	
	// Position of the eigenvectors
	const int eigen_id = iwork[5];	

	// Loads eigenvector components
	if(local_id < 3)
		LSTORE(GET_EIGEN(_eigen, eigen_id+s_id, local_id+1),
			local_id, fwork);
	
	barrier(CLK_LOCAL_MEM_FENCE);

	const _var e_vec1 = LLOAD(0, fwork);
	const _var e_vec2 = LLOAD(1, fwork);
	const _var e_vec3 = LLOAD(2, fwork);
	
#else

	const int guide_id = GET_STAGEY1_GUIDE1_ELEM(_guide1, gid0, 0,  _i);
	const int s_id = GET_STAGEY1_GUIDE1_ELEM(_guide1, gid0, 1, _i);
	
	const int elem1_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 0);
	const int elem2_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 1);
	const int elem3_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 2);
	const int eigen_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 3);

	const _var e_vec1 =
			GET_EIGEN(_eigen, eigen_id+s_id, 1);
	const _var e_vec2 =
			GET_EIGEN(_eigen, eigen_id+s_id, 2);
	const _var e_vec3 =
			GET_EIGEN(_eigen, eigen_id+s_id, 3);

#endif
	
#if LEVEL2
	// Calculate the memory addresses for the elements
	const __global _var *f = _f + gid1*L2_LDF2*L2_LDF3;
	const __global _var *elem1 = f+elem1_id*L2_LDF3;
	const __global _var *elem2 = f+elem2_id*L2_LDF3;
	const __global _var *elem3 = f+elem3_id*L2_LDF3;
	
	// Calculate memory address for the g-vector
	_var_global_array g =
		SUB_GLOBAL_ARRAY(
				GLOBAL_ARRAY(_g, L3_LDF1*L3_LDF2*L3_LDF3),
				(gid1*L3_LDF2+gid0)*L3_LDF3);
				
#else

	const __global _var *elem1 = _f+(elem1_id*L1_LDF2+gid1)*L1_LDF3;
	const __global _var *elem2 = _f+(elem2_id*L1_LDF2+gid1)*L1_LDF3;
	const __global _var *elem3 = _f+(elem3_id*L1_LDF2+gid1)*L1_LDF3;

	__global _var *g = _g + (gid0*L2_LDF2+gid1)*L2_LDF3;

#endif

	const int begin = gid2 * LX_STAGE_11_WG_SIZE;
	const int jump = LX_STAGE_11_WG_PER_VECTOR * LX_STAGE_11_WG_SIZE;
	
	// Generate the right-hand side vector
	if(0 <=  elem3_id) {
		for(int d = begin+local_id; D*d < N3; d += jump) {
			_varD data;
		   	data  = MULD(e_vec1, VLOADD(d, elem1));
			data += MULD(e_vec2, VLOADD(d, elem2));
			data += MULD(e_vec3, VLOADD(d, elem3));
#if LEVEL2
			GVSTOREDD(data, d, g);
#else
			VSTOREDD(data, d, g);
#endif
		}
	} else if(0 <=  elem2_id) {
		for(int d = begin+local_id; D*d < N3; d += jump) {
			_varD data;
		   	data  = MULD(e_vec1, VLOADD(d, elem1));
			data += MULD(e_vec2, VLOADD(d, elem2));
#if LEVEL2
			GVSTOREDD(data, d, g);
#else
			VSTOREDD(data, d, g);
#endif
		}
	}
	else if(0 <= elem1_id) {
		for(int d = begin+local_id; D*d < N3; d += jump) {
			_varD data;
			data  = MULD(e_vec1, VLOADD(d, elem1));
#if LEVEL2
			GVSTOREDD(data, d, g);
#else
			VSTOREDD(data, d, g);
#endif
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/* The second half of the first stage, step A. Performs the first step of the
 * recursive summation. Each solution vector is multiplied by the corresponding
 * eigenvector component and the result of each partial sum is stored into a
 * temporary buffer. 
 * Arguments:
 *  _v			v-vector
 *  _tmp		Temporary buffer ()
 *  _guide		Guiding information, i.e., what each work group should do
 *  _eigen		Eigenvector components
 *  _i			Recursion index
 *  _
 * Work groups:
 *  get_num_groups(0)	The total number of partial sums
 *  get_num_groups(1) 	If LEVEL2 is enabled, then this is the number of level
 * 						2 subproblem. Otherwise, it is equal to N2.
 *  get_num_groups(2)  	Number of work groups per N3-vector
 * 						  = LX_STAGE_12A_WG_PER_VECTOR
 */
__attribute__((reqd_work_group_size(LX_STAGE_12A_WG_SIZE, 1, 1)))
__kernel void lx_stage_12a(
#if LEVEL2
		__global       _var_t *_v,
#else
		__global       _var   *_v,
#endif
		__global       _var   *_tmp,
		_stage12a_guide       *_guide,
		__global const _var   *_eigen,
						int    _i, 
						int    _max_sum_size
		) {

	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);

	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1);
	
#if 1 < LX_STAGE_12A_WG_PER_VECTOR
	const int gid2 = get_group_id(2);
#else
	const int gid2 = 0;
#endif

	// Determine the index of the vector sum. At this point, it is (incorrectly)
	// assumed that each sum is of the size 4^i. This divergency is handled 
	// later.
	const int s_id = gid0 / DIVCEIL(_max_sum_size, LX_STAGE_Y2A_MAX_SUM_SIZE);
	
	// Calculate the index of the partial sum (trying to avoid mod-operations)
	const int ps_id = gid0 % DIVCEIL(_max_sum_size, LX_STAGE_Y2A_MAX_SUM_SIZE);
	
#if LX_SHARED_ISOLATED_ACCESS

	__local int iwork[4];

	// Loads guiding information
	if(local_id < 4)
		iwork[local_id] = GET_STAGE12A_GUIDE_ELEM(_guide, s_id, local_id, _i);
	barrier(CLK_LOCAL_MEM_FENCE);

	// Location of the first element of the sum
	const int upper_bound = iwork[0];

	// Location of the last element of the sum + 1
	const int lower_bound = iwork[1];
	
	// Position of the eigenvectors
	const int eigen_id = iwork[2];
	
	// 0 => upper sum, 1 => lower sum
	const int sum_slct = iwork[3];

#else

	const int upper_bound = GET_STAGE12A_GUIDE_ELEM(_guide, s_id, 0, _i);
	const int lower_bound = GET_STAGE12A_GUIDE_ELEM(_guide, s_id, 1, _i);
	const int eigen_id = GET_STAGE12A_GUIDE_ELEM(_guide, s_id, 2, _i);
	const int sum_slct = GET_STAGE12A_GUIDE_ELEM(_guide, s_id, 3, _i);
	
#endif

	// Select the eigenvector component
	const int eigen_num = sum_slct ? 4 : 0;

	// Calculate partial sum boundaries
	const int s_begin = ps_id * LX_STAGE_Y2A_MAX_SUM_SIZE;
	const int s_end = min(lower_bound - upper_bound, (ps_id+1)*LX_STAGE_Y2A_MAX_SUM_SIZE);
	const int s_size = s_end - s_begin;
	
	// Return if the size of the partial sum is zero
	if(s_size <= 0)
		return;

#if LEVEL2
	
	// Calculates the location of the v-vector
	_var_global_array v = SUB_GLOBAL_ARRAY(
		GLOBAL_ARRAY(_v, L3_LDF1*L3_LDF2*L3_LDF3), 
			(gid1*L3_LDF2+upper_bound)*L3_LDF3);
	
	// Calculates the location of the temporary memory buffer
	__global _var *tmp = 
		_tmp + (gid1*TMP_PER_L2+gid0)*L2_LDF3;
	
#else

	__global _var *v = _v + (upper_bound*L2_LDF2+gid1)*L2_LDF3;
	
	__global _var *tmp = _tmp + (gid0*L1_LDF2+gid1)*L1_LDF3;

#endif	

	const int begin = gid2 * LX_STAGE_12A_WG_SIZE;
	const int jump = LX_STAGE_12A_WG_PER_VECTOR * LX_STAGE_12A_WG_SIZE;

	// Computes the u-vector (u = (R^~ W (x) I_N3)v)
	for(int d = begin+local_id; D*d < N3; d += jump) {
		_varD res = 0.0;
		for(int i = s_begin; i < s_end; i++) {
			_var eigen = 
				GET_EIGEN(_eigen, eigen_id+i, eigen_num);
#if LEVEL2
			res += MULD(eigen, GVLOADD(i*L3_LDF3/D+d, v));
#else
			res += MULD(eigen, VLOADD(i*L2_LDF2*L2_LDF3/D+d, v));
#endif
		}
		VSTOREDD(res, d, tmp);
	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/* The second half of the first stage, step B: Performs the second step of the
 * recursive summation. 
 * Arguments:
 *  _tmp			Temporary buffer ()
 *  _guide			Guiding information, i.e., what each work group should do
 *  _sum_step		Summation recursions index
 * Work groups:
 *  get_num_groups(0)	The total number of partial sums
 *  get_num_groups(1) 	If LEVEL2 is enabled, then this is the number of level
 * 						2 subproblem. Otherwise, it is equal to N2.
 *  get_num_groups(2)  	Number of work groups per N3-vector
 * 						  = LX_STAGE_Y2B_WG_PER_VECTOR
 */
__attribute__((reqd_work_group_size(LX_STAGE_Y2B_WG_SIZE, 1, 1)))
__kernel void lx_stage_y2b(
		__global       _var *_tmp,
		_stagey2b_guide     *_guide,
						int  _sum_step, 
						int  _i, 
						int  _max_sum_size
		) {

	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);
		
	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1);
	
#if 1 < LX_STAGE_Y2B_WG_PER_VECTOR
	const int gid2 = get_group_id(2);
#else
	const int gid2 = 0;
#endif

	const int s_id = gid0 / DIVCEIL(_max_sum_size, 
		POW2(LX_STAGE_Y2B_MAX_SUM_SIZE_EXP * _sum_step + 
			LX_STAGE_Y2A_MAX_SUM_SIZE_EXP));
	
	const int ps_id = gid0 % DIVCEIL(_max_sum_size, 
		POW2(LX_STAGE_Y2B_MAX_SUM_SIZE_EXP * _sum_step + 
			LX_STAGE_Y2A_MAX_SUM_SIZE_EXP));
		
#if LX_SHARED_ISOLATED_ACCESS

	__local int iwork[2];

	// Loads guiding information
	if(local_id < 2)
		iwork[local_id] = GET_STAGEY2B_GUIDE_ELEM(_guide, s_id, local_id, _i);
	barrier(CLK_LOCAL_MEM_FENCE);

	const int upper_bound = iwork[0];
	const int lower_bound = iwork[1];

#else
	const int upper_bound = GET_STAGEY2B_GUIDE_ELEM(_guide, s_id, 0, _i);
	const int lower_bound = GET_STAGEY2B_GUIDE_ELEM(_guide, s_id, 1, _i);
#endif

	// Calculate partial sum boundaries
	const int s_begin = ps_id * POW2(_sum_step*LX_STAGE_Y2B_MAX_SUM_SIZE_EXP);
	const int s_end = 
		min(lower_bound - upper_bound, 
			(ps_id+1) * POW2(_sum_step*LX_STAGE_Y2B_MAX_SUM_SIZE_EXP));
	const int s_jump = POW2((_sum_step-1) * LX_STAGE_Y2B_MAX_SUM_SIZE_EXP);
	
	if(s_end <= s_begin + s_jump)
		return;

#if LEVEL2
	__global _var *tmp = _tmp + (gid1*TMP_PER_L2 + 
		s_id*(DIVCEIL(_max_sum_size, LX_STAGE_Y2A_MAX_SUM_SIZE)))*L2_LDF3;
#else
	__global _var *tmp = _tmp + ((s_id*(DIVCEIL(_max_sum_size, 
			LX_STAGE_Y2A_MAX_SUM_SIZE)))*L1_LDF2+gid1)*L1_LDF3;
#endif

	const int begin = gid2 * LX_STAGE_Y2B_WG_SIZE;
	const int jump = LX_STAGE_Y2B_WG_PER_VECTOR * LX_STAGE_Y2B_WG_SIZE;

	for(int d = begin+local_id; D*d < N3; d += jump) {
		_varD res = 0.0;
		for(int i = s_begin; i < s_end; i += s_jump) {
#if LEVEL2
			res += VLOADD(i*L2_LDF3/D+d, tmp);
#else
			res += VLOADD(i*L1_LDF2*L1_LDF3/D+d, tmp);
#endif
		}
#if LEVEL2
		VSTOREDD(res, s_begin*L2_LDF3/D+d, tmp);
#else
		VSTOREDD(res, s_begin*L1_LDF2*L1_LDF3/D+d, tmp);
#endif
		
	}
}

/* The second half of the first stage, step B: Updates the level X right-hand
 * side vector.
 * Arguments:

 * Work groups:
 *  get_num_groups(0)	The total number updatable elements
 *  get_num_groups(1) 	If LEVEL2 is enabled, then this is the number of level
 * 						2 subproblem. Otherwise, it is equal to N2.
 *  get_num_groups(2)  	Number of work groups per N3-vector
 * 						  = LX_STAGE_12C_WG_PER_VECTOR
 */
__attribute__((reqd_work_group_size(LX_STAGE_12C_WG_SIZE, 1, 1)))
__kernel void lx_stage_12c(
		__global _var   *_f,
		__global _var  *_tmp,
		_stage12c_guide *_guide,
		int              _i, 
		_var             _ch, 
		__global _var   *_matX,
#if LEVEL2
#if MULTIBLE_LAMBDA
		__global _var_t *_matXp1, 
		__global _var 	*_lambda, 
		int              _lambda_stride
#else
		__global _var_t *_matXp1
#endif
#else
		__global _var   *_matXp1, 
		__global _var_t *_matXp2
#endif
		) {
	  

#if LEVEL2
#if (LX_VECTOR_LOAD_HELPER || LX_MATRIX_LOAD_HELPER) && (M2_TRIDIAG || M3_TRIDIAG)
	ALLOC_LOCAL_MEM(fwork, fwork_tmp, LX_STAGE_12C_WG_SIZE);
#else
	ALLOC_LOCAL_MEM(fwork, fwork_tmp, 4);
#endif
#else 
#if (LX_VECTOR_LOAD_HELPER || LX_MATRIX_LOAD_HELPER) && (M1_TRIDIAG  || M2_TRIDIAG || M3_TRIDIAG)
	ALLOC_LOCAL_MEM(fwork, fwork_tmp, LX_STAGE_12C_WG_SIZE);
#else
	ALLOC_LOCAL_MEM(fwork, fwork_tmp, 10);
#endif
#endif

	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);

	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1);

#if 1 < LX_STAGE_12C_WG_PER_VECTOR
	const int gid2 = get_group_id(2);
#else
	const int gid2 = 0;
#endif
	
#if LX_SHARED_ISOLATED_ACCESS
	
	/*
	 * Load guiding information
	 */
	
	__local int iwork[3];
	if(local_id < 3)
		iwork[local_id] = GET_STAGE12C_GUIDE_ELEM(_guide, gid0, local_id, _i);
	barrier(CLK_LOCAL_MEM_FENCE);

	// Location of the element which is going to be updated
	const int update_loc = iwork[0];
	
	// Location of the upper sum
	const int upper_loc = iwork[1];

	// Location of the lower sum
	const int lower_loc = iwork[2];

	/*
     * Load level X offdiagonal elements
     */
	
#if LEVEL2

	// Load A_2 offdiagonal
	if(local_id == 0) {
		if (0 <= upper_loc)
			LSTORE(_matX[A2_CODIAG_STRIDE+update_loc], 0, fwork);
		if (0 <= lower_loc)
			LSTORE(_matX[A2_CODIAG_STRIDE+update_loc+1], 1, fwork);
		
#if M2_TRIDIAG
		// Load M_2 offdiagonal
		if (0 <= upper_loc)
			LSTORE(_matX[M2_CODIAG_STRIDE+update_loc], 2, fwork);
		if (0 <= lower_loc)
			LSTORE(_matX[M2_CODIAG_STRIDE+update_loc+1], 3, fwork);
#endif
	}
		
	barrier(CLK_LOCAL_MEM_FENCE);

	const _var left_a2_offdiag = 0 <= upper_loc ? LLOAD(0, fwork) : 0.0;
	const _var right_a2_offdiag = 0 <= lower_loc ? LLOAD(1, fwork) : 0.0;
	
#if M2_TRIDIAG
	const _var left_m2_offdiag = 0 <= upper_loc ? LLOAD(2, fwork) : 0.0;
	const _var right_m2_offdiag = 0 <= lower_loc ? LLOAD(3, fwork) : 0.0;
#endif
	
#else // if LEVEL2 ... ***

	// Load all neccessary level 1 offdiagonal elements
	if(local_id == 0) {
		if (0 <= upper_loc)
			LSTORE(_matX[A1_CODIAG_STRIDE+update_loc], 0, fwork);
		if (0 <= lower_loc)
			LSTORE(_matX[A1_CODIAG_STRIDE+update_loc+1], 1, fwork);
#if M1_TRIDIAG
		if (0 <= upper_loc)
			LSTORE(_matX[M1_CODIAG_STRIDE+update_loc], 2, fwork);
		if (0 <= lower_loc)
			LSTORE(_matX[M1_CODIAG_STRIDE+update_loc+1], 3, fwork);
#endif
	}

	// Load all neccessary level 2 offdiagonal elements
	if(local_id == 0)
		LSTORE(_matXp1[A2_DIAG_STRIDE+gid1], 4, fwork);
	if(local_id < 2)
		LSTORE(_matXp1[A2_CODIAG_STRIDE+gid1+local_id],  5+local_id, fwork);
	if(local_id == 0)
		LSTORE(_matXp1[M2_DIAG_STRIDE+gid1], 7, fwork);
#if M2_TRIDIAG
	if(local_id < 2)
		LSTORE(_matXp1[M2_CODIAG_STRIDE+gid1+local_id], 8+local_id, fwork);
#endif

	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Read shared data from the local memory
	const _var left_a1_offdiag = 0 <= upper_loc ? LLOAD(0, fwork) : 0.0;
	const _var right_a1_offdiag = 0 <= lower_loc ? LLOAD(1, fwork) : 0.0;
#if M1_TRIDIAG
	const _var left_m1_offdiag = 0 <= upper_loc ? LLOAD(2, fwork) : 0.0;
	const _var right_m1_offdiag = 0 <= lower_loc ? LLOAD(3, fwork) : 0.0;
#endif
	const _var a2_diag = LLOAD(4, fwork);
	const _var left_a2_offdiag = 0 <= gid1-1 ? LLOAD(5, fwork) : 0.0;
	const _var right_a2_offdiag = gid1+1 < N2 ? LLOAD(6, fwork) : 0.0;
	const _var m2_diag = LLOAD(7, fwork);
#if M2_TRIDIAG
	const _var left_m2_offdiag = 0 <= gid1-1 ? LLOAD(8, fwork) : 0.0;
	const _var right_m2_offdiag = gid1+1 < N2 ? LLOAD(9, fwork) : 0.0;
#endif

	barrier(CLK_LOCAL_MEM_FENCE);
	
#endif // if LEVEL2 ... else ... ***
	
#else // if LX_SHARED_ISOLATED_ACCESS ... ***

	const int update_loc = GET_STAGE12C_GUIDE_ELEM(_guide, gid0, 0, _i);
	const int upper_loc = GET_STAGE12C_GUIDE_ELEM(_guide, gid0, 1, _i);
	const int lower_loc = GET_STAGE12C_GUIDE_ELEM(_guide, gid0, 2, _i);
	
#if LEVEL2

	const _var left_a2_offdiag = 
		0 <= upper_loc ? _matX[A2_CODIAG_STRIDE+update_loc] : 0.0;
	const _var right_a2_offdiag = 
		0 <= lower_loc ? _matX[A2_CODIAG_STRIDE+update_loc+1] : 0.0;
	
#if M2_TRIDIAG
	const _var left_m2_offdiag = 
		0 <= upper_loc ? _matX[M2_CODIAG_STRIDE+update_loc] : 0.0;
	const _var right_m2_offdiag = 
		0 <= lower_loc ? _matX[M2_CODIAG_STRIDE+update_loc+1] : 0.0;
#endif

#else // if LEVEL2 ... ***
	
	const _var left_a1_offdiag = 
		0 <= upper_loc ? _matX[A1_CODIAG_STRIDE+update_loc] : 0.0;
	const _var right_a1_offdiag = 
		0 <= lower_loc ? _matX[A1_CODIAG_STRIDE+update_loc+1] : 0.0;
#if M1_TRIDIAG
	const _var left_m1_offdiag = 
		0 <= upper_loc ? _matX[M1_CODIAG_STRIDE+update_loc] : 0.0;
	const _var right_m1_offdiag = 
		0 <= lower_loc ? _matX[M1_CODIAG_STRIDE+update_loc+1] : 0.0;
#endif
	const _var a2_diag = _matXp1[A2_DIAG_STRIDE+gid1];
	const _var left_a2_offdiag = _matXp1[A2_CODIAG_STRIDE+gid1];
	const _var right_a2_offdiag = _matXp1[A2_CODIAG_STRIDE+gid1+1];
	const _var m2_diag = _matXp1[M2_DIAG_STRIDE+gid1];
#if M2_TRIDIAG
	const _var left_m2_offdiag = _matXp1[M2_CODIAG_STRIDE+gid1];
	const _var right_m2_offdiag = _matXp1[M2_CODIAG_STRIDE+gid1+1];
#endif

#endif // if LEVEL2 ... else ... ***

#endif // if LX_SHARED_ISOLATED_ACCESS ... else ... ***
	
	/*
     * Level 2 problem
     */
	
#if LEVEL2

	// Locate the updateable vector and the results of each sums
	__global _var *update = _f + (gid1*L2_LDF2 + update_loc)*L2_LDF3;
	__global const _var *upper_sum = 
		_tmp + (gid1*TMP_PER_L2+upper_loc)*L2_LDF3;
	__global const _var *lower_sum = 
		_tmp + (gid1*TMP_PER_L2+lower_loc)*L2_LDF3;

	/*
     * Load level 3 matrices
     */
	
#if M2_TRIDIAG
	_var_global_array a3_diag, a3_offdiag;
	 get_coef_matrix_components(_matXp1, &a3_diag, &a3_offdiag, 0, 0);
#endif
	
#if M3_TRIDIAG
	_var_global_array m3_diag, m3_offdiag;
	 get_coef_matrix_components(_matXp1, 0, 0, &m3_diag,  &m3_offdiag);
#else
	_var_global_array m3_diag;
	 get_coef_matrix_components(_matXp1, 0, 0, &m3_diag, 0);
#endif
	
	const int begin = gid2 * LX_STAGE_12C_WG_SIZE;
	const int jump = LX_STAGE_12C_WG_PER_VECTOR * LX_STAGE_12C_WG_SIZE;

#if MULTIBLE_LAMBDA
	const _var mul = _lambda[_lambda_stride + gid1] + _ch;
#else
	const _var mul = _ch;
#endif
	
	for(int i = begin+local_id; i < N3; i+= jump) {

		// Load upper sum
		_var upper_m = 0.0;
#if M2_TRIDIAG || M3_TRIDIAG
		_var upper_u = 0.0, upper_l = 0.0;
#endif
		if(0 <= upper_loc) { 
			upper_m = upper_sum[i];
#if M2_TRIDIAG || M3_TRIDIAG
			l3_vector_load_helper(&upper_u, &upper_m, &upper_l, upper_sum, i, 
				local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		}

		// Load lower sum
		_var lower_m = 0.0;
#if M2_TRIDIAG || M3_TRIDIAG
		_var lower_u = 0.0, lower_l = 0.0;
#endif
		if(0 <= lower_loc) {
			lower_m = lower_sum[i];
#if M2_TRIDIAG || M3_TRIDIAG
		l3_vector_load_helper(&lower_u, &lower_m, &lower_l, lower_sum, i, 
			local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		}
		
		// Load A_3
#if M2_TRIDIAG
		const _var a3_b = GLOAD(i, a3_diag);
		_var a3_a, a3_c;
		l3_matrix_load_helper(&a3_a, &a3_c, a3_offdiag, i, 
			local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		
		// Load M_3
		const _var m3_b = GLOAD(i, m3_diag);
#if M3_TRIDIAG
		_var m3_a, m3_c;
		l3_matrix_load_helper(&m3_a, &m3_c, m3_offdiag, i, 
			local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif

		_var tmp = 0.0;

#define MUL_HELPER(left_offdiag, right_offdiag, a, b, c) ( 	\
		MUL(left_offdiag, (									\
			MUL(a, upper_u) +								\
			MUL(b, upper_m) +								\
			MUL(c, upper_l))) +								\
		MUL(right_offdiag, (								\
			MUL(a, lower_u) +								\
			MUL(b, lower_m) +								\
			MUL(c, lower_l))) )                             

		// A_2 (x) M_3 
#if M3_TRIDIAG
		tmp += MUL_HELPER(left_a2_offdiag, right_a2_offdiag, m3_a, m3_b, m3_c);
#else
		tmp += 
			MUL(left_a2_offdiag, MUL(m3_b, upper_m)) +
			MUL(right_a2_offdiag, MUL(m3_b, lower_m));
#endif

		// M_2 (x) A_3 
#if M2_TRIDIAG
		tmp += MUL_HELPER(left_m2_offdiag, right_m2_offdiag, a3_a, a3_b, a3_c);
#endif

		// mul * (M_2 (x) M_3) 
#if M2_TRIDIAG && M3_TRIDIAG
		tmp += MUL(mul, 
			MUL_HELPER(left_m2_offdiag, right_m2_offdiag, m3_a, m3_b, m3_c));
#elif M2_TRIDIAG
		tmp += MUL(mul, 
			MUL(left_m2_offdiag, MUL(m3_b, upper_m)) +
			MUL(right_m2_offdiag, MUL(m3_b, lower_m)));
#endif
	

#undef MUL_HELPER
	
		// And finally, store the result
		update[i] -= tmp;
	}

#else // if LEVEL2 ... ***

	/*
     * Level 1 problem
     */

	// Locate the updateable vector and the results of each sums
	__global _var *update = _f + (update_loc*L1_LDF2+gid1)*L1_LDF3;
	const __global _var *upper_sum_m = _tmp + (upper_loc*L1_LDF2+gid1)*L1_LDF3;
	const __global _var *lower_sum_m = _tmp + (lower_loc*L1_LDF2+gid1)*L1_LDF3;

#if M1_TRIDIAG || M2_TRIDIAG
	const __global _var *upper_sum_u = _tmp + (upper_loc*L1_LDF2+gid1-1)*L1_LDF3;
	const __global _var *lower_sum_u = _tmp + (lower_loc*L1_LDF2+gid1-1)*L1_LDF3;
	const __global _var *upper_sum_l = _tmp + (upper_loc*L1_LDF2+gid1+1)*L1_LDF3;
	const __global _var *lower_sum_l = _tmp + (lower_loc*L1_LDF2+gid1+1)*L1_LDF3;
#endif
	
	/*
     * Load level 3 matrices
     */
	
#if M1_TRIDIAG
	_var_global_array a3_diag, a3_offdiag;
	 get_coef_matrix_components(_matXp2, &a3_diag, &a3_offdiag, 0, 0);
#endif
	
#if M3_TRIDIAG
	_var_global_array m3_diag, m3_offdiag;
	 get_coef_matrix_components(_matXp2, 0, 0, &m3_diag, &m3_offdiag);
#else
	_var_global_array m3_diag;
	 get_coef_matrix_components(_matXp2, 0, 0, &m3_diag, 0);
#endif
	
	const int begin = gid2 * LX_STAGE_12C_WG_SIZE;
	const int jump = LX_STAGE_12C_WG_PER_VECTOR * LX_STAGE_12C_WG_SIZE;
	
	for(int i = begin+local_id; i < N3; i+= jump) {
	
#if M1_TRIDIAG || M2_TRIDIAG
		// Load the upper section from the upper sum
		_var upper_um = 0.0;
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
		_var upper_uu = 0.0, upper_ul = 0.0;
#endif
		if(0 <= upper_loc && 0 <= gid1-1) {
			upper_um = upper_sum_u[i];
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
			l3_vector_load_helper(&upper_uu, &upper_um, &upper_ul, upper_sum_u, 
				i, local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		}
#endif
		
		// Load the middle section from the upper sum
		_var upper_mm = 0.0;
#if M1_TRIDIAG || M3_TRIDIAG
		_var upper_mu = 0.0, upper_ml = 0.0;
#endif
		if(0 <= upper_loc) {
			upper_mm = upper_sum_m[i];
#if M1_TRIDIAG || M3_TRIDIAG
			l3_vector_load_helper(&upper_mu, &upper_mm, &upper_ml, upper_sum_m, i, 
				local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		}

#if M1_TRIDIAG || M2_TRIDIAG
		// Load the lower section from the upper sum
		_var upper_lm = 0.0;
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
		_var upper_lu = 0.0, upper_ll = 0.0;
#endif
		if(0 <= upper_loc && gid1+1 < N2) {
			upper_lm = upper_sum_l[i];
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
			l3_vector_load_helper(&upper_lu, &upper_lm, &upper_ll, upper_sum_l, 
				i, local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		}
#endif
		
#if M1_TRIDIAG || M2_TRIDIAG
		// Load the upper section from the lower sum
		_var lower_um = 0.0;
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
		_var lower_uu = 0.0, lower_ul = 0.0;
#endif
		if(0 <= lower_loc && 0 <= gid1-1) {
			lower_um = lower_sum_u[i];
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
			l3_vector_load_helper(&lower_uu, &lower_um, &lower_ul, lower_sum_u, 
				i, local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		}
#endif
		
		// Load the middle section from the lower sum
		_var lower_mm = 0.0;
#if M1_TRIDIAG || M3_TRIDIAG
		_var lower_mu = 0.0, lower_ml = 0.0;
#endif
		if(0 <= lower_loc) {
			lower_mm = lower_sum_m[i];
#if M1_TRIDIAG || M3_TRIDIAG
			l3_vector_load_helper(&lower_mu, &lower_mm, &lower_ml, lower_sum_m, i, 
				local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		}

#if M1_TRIDIAG || M2_TRIDIAG
		// Load the lower section from the lower sum
		_var lower_lm = 0.0;
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
		_var lower_lu = 0.0, lower_ll = 0.0;
#endif
		if(0 <= lower_loc && gid1+1 < N2) {
			lower_lm = lower_sum_l[i];
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
			l3_vector_load_helper(&lower_lu, &lower_lm, &lower_ll, lower_sum_l, 
				i, local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		}
#endif
		
		// Load A_3
#if M1_TRIDIAG
		const _var a3_b = GLOAD(i, a3_diag);
		_var a3_a, a3_c;
		l3_matrix_load_helper(&a3_a, &a3_c, a3_offdiag, i, 
			local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
		
		// Load M_3
		const _var m3_b = GLOAD(i, m3_diag);
#if M3_TRIDIAG
		_var m3_a, m3_c;
		l3_matrix_load_helper(&m3_a, &m3_c, m3_offdiag, i, 
			local_id, LX_STAGE_12C_WG_SIZE, fwork);
#endif
			
		_var tmp = 0.0;

#define MUL_HELPER_123(left_offdiag, right_offdiag, a2, b2, c2, a3, b3, c3) ( \
		MUL(left_offdiag, ( 			\
			MUL(a2, 					\
				MUL(a3, upper_uu) + 	\
				MUL(b3, upper_um) + 	\
				MUL(c3, upper_ul)) + 	\
			MUL(b2, 					\
				MUL(a3, upper_mu) + 	\
				MUL(b3, upper_mm) + 	\
				MUL(c3, upper_ml)) + 	\
			MUL(c2, 					\
				MUL(a3, upper_lu) + 	\
				MUL(b3, upper_lm) +		\
				MUL(c3, upper_ll)))) +	\
		MUL(right_offdiag, (			\
			MUL(a2, 					\
				MUL(a3, lower_uu) + 	\
				MUL(b3, lower_um) +		\
				MUL(c3, lower_ul)) +	\
			MUL(b2, 					\
				MUL(a3, lower_mu) + 	\
				MUL(b3, lower_mm) +		\
				MUL(c3, lower_ml)) +	\
			MUL(c2, 					\
				MUL(a3, lower_lu) + 	\
				MUL(b3, lower_lm) +		\
				MUL(c3, lower_ll)))) )
				
#define MUL_HELPER_1X3(left_offdiag, right_offdiag, b2, a3, b3, c3) ( \
		MUL(left_offdiag, ( 			\
			MUL(b2, 					\
				MUL(a3, upper_mu) + 	\
				MUL(b3, upper_mm) + 	\
				MUL(c3, upper_ml)))) +	\
		MUL(right_offdiag, (			\
			MUL(b2, 					\
				MUL(a3, lower_mu) + 	\
				MUL(b3, lower_mm) +		\
				MUL(c3, lower_ml)))) )
				
#define MUL_HELPER_12X(left_offdiag, right_offdiag, a2, b2, c2, b3) ( \
		MUL(left_offdiag, ( 			\
			MUL(a2, 					\
				MUL(b3, upper_um)) + 	\
			MUL(b2, 					\
				MUL(b3, upper_mm)) + 	\
			MUL(c2, 					\
				MUL(b3, upper_lm)))) +	\
		MUL(right_offdiag, (			\
			MUL(a2, 					\
				MUL(b3, lower_um)) +	\
			MUL(b2, 					\
				MUL(b3, lower_mm)) +	\
			MUL(c2, 					\
				MUL(b3, lower_lm)))) )
				
#define MUL_HELPER_1XX(left_offdiag, right_offdiag, b2, b3) ( \
		MUL(left_offdiag, ( 			\
			MUL(b2, 					\
				MUL(b3, upper_mm)))) +	\
		MUL(right_offdiag, (			\
			MUL(b2, 					\
				MUL(b3, lower_mm)))) )
		
		/*
         * A_1 (x) M_2 (x) M_3
         */ 
		
#if M2_TRIDIAG && M3_TRIDIAG
		tmp += MUL_HELPER_123(left_a1_offdiag, right_a1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, m3_a, m3_b, m3_c);
#elif M2_TRIDIAG
		tmp += MUL_HELPER_12X(left_a1_offdiag, right_a1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, m3_b);
#elif M3_TRIDIAG
		tmp += MUL_HELPER_1X3(left_a1_offdiag, right_a1_offdiag, 
			m2_diag, m3_a, m3_b, m3_c);
#else
		tmp += MUL_HELPER_1XX(left_a1_offdiag, right_a1_offdiag, m2_diag, m3_b);
#endif
		
		/*
         * M_1 (x) A_2 (x) M_3
         */ 
		
#if M1_TRIDIAG
#if M3_TRIDIAG
		tmp += MUL_HELPER_123(left_m1_offdiag, right_m1_offdiag, 
			left_a2_offdiag, a2_diag, right_a2_offdiag, m3_a, m3_b, m3_c);
#else
		tmp += MUL_HELPER_12X(left_m1_offdiag, right_m1_offdiag, 
			left_a2_offdiag, a2_diag, right_a2_offdiag, m3_b);
#endif
#endif
		/*
         * M_1 (x) M_2 (x) A_3
         */
		
#if M1_TRIDIAG
#if M2_TRIDIAG
		tmp += MUL_HELPER_123(left_m1_offdiag, right_m1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, a3_a, a3_b, a3_c);
#else
		tmp += MUL_HELPER_1X3(left_m1_offdiag, right_m1_offdiag, 
			m2_diag, a3_a, a3_b, a3_c);
#endif
#endif
		
		/*
         * c * (M_1 (x) M_2 (x) M_3)
         */
		
#if M1_TRIDIAG
#if M2_TRIDIAG && M3_TRIDIAG
		tmp += MUL(_ch, MUL_HELPER_123(left_m1_offdiag, right_m1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, m3_a, m3_b, m3_c));
#elif M2_TRIDIAG
		tmp += MUL(_ch, MUL_HELPER_12X(left_m1_offdiag, right_m1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, m3_b));
#elif M3_TRIDIAG
		tmp += MUL(_ch, MUL_HELPER_1X3(left_m1_offdiag, right_m1_offdiag, 
			m2_diag, m3_a, m3_b, m3_c));
#else
		tmp += MUL(_ch, MUL_HELPER_1XX(left_m1_offdiag, right_m1_offdiag, 
			m2_diag, m3_b));
#endif
#endif
		
					
#undef MUL_HELPER_123
#undef MUL_HELPER_1X3
#undef MUL_HELPER_12X
#undef MUL_HELPER_1XX
					
		update[i] -= tmp;
	}

#endif // if LEVEL2 ... else ... ***
	
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

__attribute__((reqd_work_group_size(LX_STAGE_21_WG_SIZE, 1, 1)))
__kernel void lx_stage_21(
		__global _var   *_f,
#if LEVEL2
		__global _var_t *_g,
#else
		__global _var   *_g,
#endif
		_stagey1_guide1 *_guide1,
		_stageY1_guide2 *_guide2,
		__global _var   *_eigen,
		int              _i, 
		_var             _ch, 
		__global _var  *_matX,
#if LEVEL2
#if MULTIBLE_LAMBDA
		__global _var_t *_matXp1, 
		__global _var 	*_lambda, 
		int              _lambda_stride
#else
		__global _var_t *_matXp1
#endif
#else
		__global _var   *_matXp1, 
		__global _var_t *_matXp2
#endif
		) {

#if LEVEL2
#if (LX_VECTOR_LOAD_HELPER || LX_MATRIX_LOAD_HELPER) && \
(M2_TRIDIAG || M3_TRIDIAG)
	ALLOC_LOCAL_MEM(fwork, fwork_tmp, LX_STAGE_21_WG_SIZE);
#else
	ALLOC_LOCAL_MEM(fwork, fwork_tmp, 9);
#endif
#else 
#if (LX_VECTOR_LOAD_HELPER || LX_MATRIX_LOAD_HELPER) && \
(M1_TRIDIAG  || M2_TRIDIAG || M3_TRIDIAG)
	ALLOC_LOCAL_MEM(fwork, fwork_tmp, LX_STAGE_21_WG_SIZE);
#else
	ALLOC_LOCAL_MEM(fwork, fwork_tmp, 15);
#endif
#endif
		
	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);

	// Primary work group index. Specifies which level X-1 subproblem belongs 
	// to this work group.
	const int gid0 = get_group_id(0);
	
	// Secondary work group index. If LEVEL2 is enabled, then this index 
	// specifies which level 2 problem belongs to this work group. If LEVEL2
	// is disabled, then this index specifies which level N2*N3-vector belongs 
	// to this work group.
	const int gid1 = get_group_id(1);
	
#if 1 < LX_STAGE_21_WG_PER_VECTOR
	// This specifies which section of the N3-vector belongs to this work group
	const int gid2 = get_group_id(2);
#else
	const int gid2 = 0;
#endif
	
#if LX_SHARED_ISOLATED_ACCESS

	__local int iwork[10];

	// Load the guiding information
	if(local_id < 2)
		iwork[local_id] = 
			GET_STAGEY1_GUIDE1_ELEM(_guide1, gid0, local_id, _i+1);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Location of the shared guiding information
	const int guide_id = iwork[0];
	
	// Index of the eigenvector
	const int s_id = iwork[1];
	
	// Load the shared guiding information
	if(local_id < 8)
		iwork[2+local_id] = 
			GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, local_id);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Positions of the elements
	const int elem1_id = iwork[2];
	const int elem2_id = iwork[3];
	const int elem3_id = iwork[4];
	const int eigen_id = iwork[5];	
	const int elemU_id = iwork[6];
	const int elemL_id = iwork[7];
	const int mat_elemU_id = iwork[8];
	const int mat_elemL_id = iwork[9];
	
	// Loads eigenvector components
	if(local_id < 5)
		LSTORE(GET_EIGEN(_eigen, eigen_id+s_id, local_id),
			local_id, fwork);
	
#if LEVEL2

	if(local_id == 0) {
		// Load A_2 offdiagonal
		if(0 <= mat_elemU_id)
			LSTORE(_matX[A2_CODIAG_STRIDE+mat_elemU_id], 5, fwork);
		if(0 <= mat_elemL_id)
			LSTORE(_matX[A2_CODIAG_STRIDE+mat_elemL_id+1], 6, fwork);
			
#if M2_TRIDIAG
		// Load M_2 offdiagonal
		if(0 <= mat_elemU_id)
			LSTORE(_matX[M2_CODIAG_STRIDE+mat_elemU_id], 7, fwork);
		if(0 <= mat_elemL_id)
			LSTORE(_matX[M2_CODIAG_STRIDE+mat_elemL_id+1], 8, fwork);
#endif
	}
			
	barrier(CLK_LOCAL_MEM_FENCE);

	const _var left_a2_offdiag = 0 <= mat_elemU_id ?  LLOAD(5, fwork) : 0.0;
	const _var right_a2_offdiag = 0 <= mat_elemL_id ?  LLOAD(6, fwork) : 0.0;
	
#if M2_TRIDIAG
	const _var left_m2_offdiag = 0 <= mat_elemU_id ?  LLOAD(7, fwork) : 0.0;
	const _var right_m2_offdiag = 0 <= mat_elemL_id ?  LLOAD(8, fwork) : 0.0;
#endif
	
#else // if LEVEL2 ... ***

	// Load all neccessary level 1 offdiagonal elements
	if(local_id == 0) {
		if(0 <= mat_elemU_id)
			LSTORE(_matX[A1_CODIAG_STRIDE+mat_elemU_id], 5, fwork);
		if(0 <= mat_elemL_id)
			LSTORE(_matX[A1_CODIAG_STRIDE+mat_elemL_id+1], 6, fwork);
#if M1_TRIDIAG
		if(0 <= mat_elemU_id)
			LSTORE(_matX[M1_CODIAG_STRIDE+mat_elemU_id], 7, fwork);
		if(0 <= mat_elemL_id)
			LSTORE(_matX[M1_CODIAG_STRIDE+mat_elemL_id+1], 8, fwork);
#endif
	}

	// Load all neccessary level 2 offdiagonal elements
	if(local_id == 0)
			LSTORE(_matXp1[A2_DIAG_STRIDE+gid1], 9, fwork);
	if(local_id < 2)
			LSTORE(_matXp1[A2_CODIAG_STRIDE+gid1+local_id], 10+local_id, fwork);
	if(local_id == 0)
			LSTORE(_matXp1[M2_DIAG_STRIDE+gid1], 12, fwork);
#if M2_TRIDIAG
	if(local_id < 2)
			LSTORE(_matXp1[M2_CODIAG_STRIDE+gid1+local_id], 13+local_id, fwork);
#endif

	barrier(CLK_LOCAL_MEM_FENCE);

	// Read shared data from the local memory
	const _var left_a1_offdiag = 0 <= mat_elemU_id ? LLOAD(5, fwork) : 0.0;
	const _var right_a1_offdiag = 0 <= mat_elemL_id ? LLOAD(6, fwork) : 0.0;
#if M1_TRIDIAG
	const _var left_m1_offdiag = 0 <= mat_elemU_id ? LLOAD(7, fwork) : 0.0;
	const _var right_m1_offdiag = 0 <= mat_elemL_id ? LLOAD(8, fwork) : 0.0;
#endif
	const _var a2_diag = LLOAD(9, fwork);
	const _var left_a2_offdiag = LLOAD(10, fwork);
	const _var right_a2_offdiag = LLOAD(11, fwork);
	const _var m2_diag = LLOAD(12, fwork);
#if M2_TRIDIAG
	const _var left_m2_offdiag = LLOAD(13, fwork);
	const _var right_m2_offdiag = LLOAD(14, fwork);
#endif

#endif

	const _var e_vec1 = LLOAD(1, fwork);
	const _var e_vec2 = LLOAD(2, fwork);
	const _var e_vec3 = LLOAD(3, fwork);
	const _var e_vecU = 0 <= elemU_id ? LLOAD(0, fwork) : 0.0;
	const _var e_vecL = 0 <= elemL_id ? LLOAD(4, fwork) : 0.0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
#else

	const int guide_id = GET_STAGEY1_GUIDE1_ELEM(_guide1, gid0, 0, _i+1);
	const int s_id = GET_STAGEY1_GUIDE1_ELEM(_guide1, gid0, 1, _i+1);
	
	const int elem1_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 0);
	const int elem2_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 1);
	const int elem3_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 2);
	const int eigen_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 3);
	const int elemU_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 4);
	const int elemL_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 5);
	const int mat_elemU_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 6);
	const int mat_elemL_id = GET_STAGEY1_GUIDE2_ELEM(_guide2, guide_id, 7);
	
	
#if LEVEL2

	const _var left_a2_offdiag = 
		0 <= mat_elemU_id ? _matX[A2_CODIAG_STRIDE+mat_elemU_id] : 0.0;
	const _var right_a2_offdiag = 
		0 <= mat_elemL_id ? _matX[A2_CODIAG_STRIDE+mat_elemL_id+1] : 0.0;
	
#if M2_TRIDIAG
	const _var left_m2_offdiag = 
		0 <= mat_elemU_id ? _matX[M2_CODIAG_STRIDE+mat_elemU_id] : 0.0;
	const _var right_m2_offdiag = 
		0 <= mat_elemL_id ? _matX[M2_CODIAG_STRIDE+mat_elemL_id+1] : 0.0;
#endif

#else

	const _var left_a1_offdiag = 
		0 <= mat_elemU_id ? _matX[A1_CODIAG_STRIDE+mat_elemU_id] : 0.0;
	const _var right_a1_offdiag = 
		0 <= mat_elemL_id ? _matX[A1_CODIAG_STRIDE+mat_elemL_id+1] : 0.0;
#if M1_TRIDIAG
	const _var left_m1_offdiag = 
		0 <= mat_elemU_id ? _matX[M1_CODIAG_STRIDE+mat_elemU_id] : 0.0;
	const _var right_m1_offdiag = 
		0 <= mat_elemL_id ? _matX[M1_CODIAG_STRIDE+mat_elemL_id+1] : 0.0;
#endif
	const _var a2_diag = _matXp1[A2_DIAG_STRIDE+gid1];
	const _var left_a2_offdiag = _matXp1[A2_CODIAG_STRIDE+gid1];
	const _var right_a2_offdiag = _matXp1[A2_CODIAG_STRIDE+gid1+1];
	
	const _var m2_diag = _matXp1[M2_DIAG_STRIDE+gid1];
#if M2_TRIDIAG
	const _var left_m2_offdiag = _matXp1[M2_CODIAG_STRIDE+gid1];
	const _var right_m2_offdiag = _matXp1[M2_CODIAG_STRIDE+gid1+1];
#endif

#endif
	
	const _var e_vecU = 0 <= elemU_id ? 
			GET_EIGEN(_eigen, eigen_id+s_id, 0) : 0.0;
	const _var e_vec1 =
			GET_EIGEN(_eigen, eigen_id+s_id, 1);
	const _var e_vec2 =
			GET_EIGEN(_eigen, eigen_id+s_id, 2);
	const _var e_vec3 =
			GET_EIGEN(_eigen, eigen_id+s_id, 3);
	const _var e_vecL = 0 <= elemL_id ? 
			GET_EIGEN(_eigen, eigen_id+s_id, 4) : 0.0;

#endif
	
#if LEVEL2
	// Calculate the memory addresses for the elements
	const __global _var *f = _f + gid1*L2_LDF2*L2_LDF3;
	const __global _var *elemU = f+elemU_id*L2_LDF3;
	const __global _var *elem1 = f+elem1_id*L2_LDF3;
	const __global _var *elem2 = f+elem2_id*L2_LDF3;
	const __global _var *elem3 = f+elem3_id*L2_LDF3;
	const __global _var *elemL = f+elemL_id*L2_LDF3;
	
	// Calculate memory address for the g-vector
	_var_global_array g =
		SUB_GLOBAL_ARRAY(
				GLOBAL_ARRAY(_g, L3_LDF1*L3_LDF2*L3_LDF3),
				(gid1*L3_LDF2+gid0)*L3_LDF3);

	const int begin = gid2 * LX_STAGE_21_WG_SIZE;
	const int jump = LX_STAGE_21_WG_PER_VECTOR * LX_STAGE_21_WG_SIZE;
	
#if M2_TRIDIAG
	_var_global_array a3_diag, a3_offdiag;
		get_coef_matrix_components(_matXp1, &a3_diag, &a3_offdiag, 0, 0);
#endif
	
#if M3_TRIDIAG
	_var_global_array m3_diag, m3_offdiag;
		get_coef_matrix_components(_matXp1, 0, 0, &m3_diag,  &m3_offdiag);
#else
	_var_global_array m3_diag;
		get_coef_matrix_components(_matXp1, 0, 0, &m3_diag, 0);
#endif

#if MULTIBLE_LAMBDA
	const _var mul = _lambda[_lambda_stride + gid1] + _ch;
#else
	const _var mul = _ch;
#endif

	for(int d = begin+local_id; d < N3; d += jump) {

		// Load upper boundary element
		_var upper_m = 0.0;
#if M2_TRIDIAG || M3_TRIDIAG
		_var upper_u = 0.0, upper_l = 0.0;
#endif
		if(0 <= elemU_id) {
			upper_m= elemU[d];
#if M2_TRIDIAG || M3_TRIDIAG
			l3_vector_load_helper(&upper_u, &upper_m, &upper_l, elemU, d, 
				local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		}
				
		// Load lower boundary element
		_var lower_m = 0.0;
#if M2_TRIDIAG || M3_TRIDIAG
		_var lower_u = 0.0, lower_l = 0.0;
#endif
		if(0 <= elemL_id) {
			lower_m= elemL[d];
#if M2_TRIDIAG || M3_TRIDIAG
			l3_vector_load_helper(&lower_u, &lower_m, &lower_l, elemL, d, 
				local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		}
		
		// Load A_3
#if M2_TRIDIAG
		const _var a3_b = GLOAD(d, a3_diag);
		_var a3_a, a3_c;
		l3_matrix_load_helper(&a3_a, &a3_c, a3_offdiag, d, 
			local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		
		// Load M_3
		const _var m3_b = GLOAD(d, m3_diag);
#if M3_TRIDIAG
		_var m3_a, m3_c;
		l3_matrix_load_helper(&m3_a, &m3_c, m3_offdiag, d, 
			local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif

	_var tmp = 0.0;
	
#define MUL_HELPER(left_offdiag, right_offdiag, a, b, c) \
		(MUL(MUL(e_vecU, left_offdiag), (		\
				MUL(a, upper_u) +	\
				MUL(b, upper_m) +	\
				MUL(c, upper_l))) +	\
		MUL(MUL(e_vecL, right_offdiag), (		\
				MUL(a, lower_u) +	\
				MUL(b, lower_m) +	\
				MUL(c, lower_l))))                             
		
		// A_2 (x) M_3 
#if M3_TRIDIAG
		tmp += MUL_HELPER(left_a2_offdiag, right_a2_offdiag, m3_a, m3_b, m3_c);
#else
		tmp += 
			MUL(MUL(e_vecU, left_a2_offdiag), MUL(m3_b, upper_m)) +
			MUL(MUL(e_vecL, right_a2_offdiag), MUL(m3_b, lower_m));
#endif

		// M_2 (x) A_3 
#if M2_TRIDIAG
		tmp += MUL_HELPER(left_m2_offdiag, right_m2_offdiag, a3_a, a3_b, a3_c);
#endif
						
		// c * (M_2 (x) M_3) 
#if M2_TRIDIAG && M3_TRIDIAG
		tmp += MUL(mul, 
			MUL_HELPER(left_m2_offdiag, right_m2_offdiag, m3_a, m3_b, m3_c));        
#elif M2_TRIDIAG
		tmp += MUL(mul, 
			MUL(MUL(e_vecU, left_m2_offdiag), MUL(m3_b, upper_m)) +
			MUL(MUL(e_vecL, right_m2_offdiag), MUL(m3_b, lower_m)));
#endif

#undef MUL_HELPER

		// Generate the right-hand side vector
 		if(0 <= elem3_id) {
			_var data;
		   	data  = MUL(e_vec1, elem1[d]);
			data += MUL(e_vec2, elem2[d]);
			data += MUL(e_vec3, elem3[d]);

			GSTORE(data - tmp, d, g);
		} else if(0 <= elem2_id) {
			_var data;
		   	data  = MUL(e_vec1, elem1[d]);
			data += MUL(e_vec2, elem2[d]);

			GSTORE(data - tmp, d, g);
		} else {
			_var data;
			data  = MUL(e_vec1, elem1[d]);

			GSTORE(data - tmp, d, g);
		}  
	}
				
#else

	const __global _var *elemU = _f+(elemU_id*L1_LDF2+gid1)*L1_LDF3;
	const __global _var *elem1 = _f+(elem1_id*L1_LDF2+gid1)*L1_LDF3;
	const __global _var *elem2 = _f+(elem2_id*L1_LDF2+gid1)*L1_LDF3;
	const __global _var *elem3 = _f+(elem3_id*L1_LDF2+gid1)*L1_LDF3;
	const __global _var *elemL = _f+(elemL_id*L1_LDF2+gid1)*L1_LDF3;
	
#if M1_TRIDIAG || M2_TRIDIAG
	const __global _var *elemUu = _f + (elemU_id*L1_LDF2+gid1-1)*L1_LDF3;
	const __global _var *elemUl = _f + (elemU_id*L1_LDF2+gid1+1)*L1_LDF3;
	const __global _var *elemLu = _f + (elemL_id*L1_LDF2+gid1-1)*L1_LDF3;
	const __global _var *elemLl = _f + (elemL_id*L1_LDF2+gid1+1)*L1_LDF3;
#endif
	
	__global _var *g = _g + (gid0*L2_LDF2+gid1)*L2_LDF3;
	
#if M1_TRIDIAG
	_var_global_array a3_diag, a3_offdiag;
	 get_coef_matrix_components(_matXp2, &a3_diag, &a3_offdiag, 0, 0);
#endif
	
#if M3_TRIDIAG
	_var_global_array m3_diag, m3_offdiag;
	 get_coef_matrix_components(_matXp2, 0, 0, &m3_diag,  &m3_offdiag);
#else
	_var_global_array m3_diag;
	 get_coef_matrix_components(_matXp2, 0, 0, &m3_diag, 0);
#endif
	
	const int begin = gid2 * LX_STAGE_21_WG_SIZE;
	const int jump = LX_STAGE_21_WG_PER_VECTOR * LX_STAGE_21_WG_SIZE;
	
	for(int d = begin+local_id; d < N3; d+= jump) {
	
#if M1_TRIDIAG || M2_TRIDIAG
		// Load the upper section from the upper element
		_var upper_um = 0.0;
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
		_var upper_uu = 0.0, upper_ul = 0.0;
#endif
		if(0 <= elemU_id && 0 <= gid1-1) {
			upper_um = elemUu[d];
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
			l3_vector_load_helper(&upper_uu, &upper_um, &upper_ul, elemUu, 
				d, local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		}
#endif
		
		// Load the middle section from the upper element
		_var upper_mm = 0.0;
#if M1_TRIDIAG || M3_TRIDIAG
		_var upper_mu = 0.0, upper_ml = 0.0;
#endif
		if(0 <= elemU_id) {
			upper_mm = elemU[d];
#if M1_TRIDIAG || M3_TRIDIAG
			l3_vector_load_helper(&upper_mu, &upper_mm, &upper_ml, elemU, d, 
				local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		}

#if M1_TRIDIAG || M2_TRIDIAG
		// Load the lower section from the upper element
		_var upper_lm = 0.0;
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
		_var upper_lu = 0.0, upper_ll = 0.0;
#endif
		if(0 <= elemU_id && gid1+1 < N2) {
			upper_lm = elemUl[d];
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
			l3_vector_load_helper(&upper_lu, &upper_lm, &upper_ll, elemUl, 
				d, local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		}
#endif
		
#if M1_TRIDIAG || M2_TRIDIAG
		// Load the upper section from the lower element
		_var lower_um = 0.0;
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
		_var lower_uu = 0.0, lower_ul = 0.0;
#endif
		if(0 <= elemL_id && 0 <= gid1-1) {
			lower_um = elemLu[d];
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
			l3_vector_load_helper(&lower_uu, &lower_um, &lower_ul, elemLu, 
				d, local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		}
#endif
		
		// Load the middle section from the lower element
		_var lower_mm = 0.0;
#if M1_TRIDIAG || M3_TRIDIAG
		_var lower_mu = 0.0, lower_ml = 0.0;
#endif
		if(0 <= elemL_id) {
			lower_mm = elemL[d];
#if M1_TRIDIAG || M3_TRIDIAG
			l3_vector_load_helper(&lower_mu, &lower_mm, &lower_ml, elemL, d, 
				local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		}

#if M1_TRIDIAG || M2_TRIDIAG
		// Load the lower section from the lower element
		_var lower_lm = 0.0;
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
		_var lower_lu = 0.0, lower_ll = 0.0;
#endif
		if(0 <= elemL_id && gid1+1 < N2) {
			lower_lm = elemLl[d];
#if M3_TRIDIAG || (M1_TRIDIAG && M2_TRIDIAG)
			l3_vector_load_helper(&lower_lu, &lower_lm, &lower_ll, elemLl, 
				d, local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		}
#endif
		
		// Load A_3
#if M1_TRIDIAG
		const _var a3_b = GLOAD(d, a3_diag);
		_var a3_a, a3_c;
		l3_matrix_load_helper(&a3_a, &a3_c, a3_offdiag, d, 
			local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
		
		// Load M_3
		const _var m3_b = GLOAD(d, m3_diag);
#if M3_TRIDIAG
		_var m3_a, m3_c;
		l3_matrix_load_helper(&m3_a, &m3_c, m3_offdiag, d, 
			local_id, LX_STAGE_21_WG_SIZE, fwork);
#endif
			
		_var tmp = 0.0;

#define MUL_HELPER_123(left_offdiag, right_offdiag, a2, b2, c2, a3, b3, c3) ( \
		MUL(MUL(e_vecU, left_offdiag), ( 	\
			MUL(a2, 					\
				MUL(a3, upper_uu) + 	\
				MUL(b3, upper_um) + 	\
				MUL(c3, upper_ul)) + 	\
			MUL(b2, 					\
				MUL(a3, upper_mu) + 	\
				MUL(b3, upper_mm) + 	\
				MUL(c3, upper_ml)) + 	\
			MUL(c2, 					\
				MUL(a3, upper_lu) + 	\
				MUL(b3, upper_lm) +		\
				MUL(c3, upper_ll)))) +	\
		MUL(MUL(e_vecL, right_offdiag), (	\
			MUL(a2, 					\
				MUL(a3, lower_uu) + 	\
				MUL(b3, lower_um) +		\
				MUL(c3, lower_ul)) +	\
			MUL(b2, 					\
				MUL(a3, lower_mu) + 	\
				MUL(b3, lower_mm) +		\
				MUL(c3, lower_ml)) +	\
			MUL(c2, 					\
				MUL(a3, lower_lu) + 	\
				MUL(b3, lower_lm) +		\
				MUL(c3, lower_ll)))) )
				
#define MUL_HELPER_1X3(left_offdiag, right_offdiag, b2, a3, b3, c3) ( \
		MUL(MUL(e_vecU, left_offdiag), ( \
			MUL(b2, 					\
				MUL(a3, upper_mu) + 	\
				MUL(b3, upper_mm) + 	\
				MUL(c3, upper_ml)))) +	\
		MUL(MUL(e_vecL, right_offdiag), (	\
			MUL(b2, 					\
				MUL(a3, lower_mu) + 	\
				MUL(b3, lower_mm) +		\
				MUL(c3, lower_ml)))) )
				
#define MUL_HELPER_12X(left_offdiag, right_offdiag, a2, b2, c2, b3) ( \
		MUL(MUL(e_vecU, left_offdiag), ( \
			MUL(a2, 					\
				MUL(b3, upper_um)) + 	\
			MUL(b2, 					\
				MUL(b3, upper_mm)) + 	\
			MUL(c2, 					\
				MUL(b3, upper_lm)))) +	\
		MUL(MUL(e_vecL, right_offdiag), (	\
			MUL(a2, 					\
				MUL(b3, lower_um)) +	\
			MUL(b2, 					\
				MUL(b3, lower_mm)) +	\
			MUL(c2, 					\
				MUL(b3, lower_lm)))) )
				
#define MUL_HELPER_1XX(left_offdiag, right_offdiag, b2, b3) ( \
		MUL(MUL(e_vecU, left_offdiag), ( \
			MUL(b2, 					\
				MUL(b3, upper_mm)))) +	\
		MUL(MUL(e_vecL, right_offdiag), (	\
			MUL(b2, 					\
				MUL(b3, lower_mm)))) )
		
		/*
         * A_1 (x) M_2 (x) M_3
         */ 
		
#if M2_TRIDIAG && M3_TRIDIAG
		tmp += MUL_HELPER_123(left_a1_offdiag, right_a1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, m3_a, m3_b, m3_c);
#elif M2_TRIDIAG
		tmp += MUL_HELPER_12X(left_a1_offdiag, right_a1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, m3_b);
#elif M3_TRIDIAG
		tmp += MUL_HELPER_1X3(left_a1_offdiag, right_a1_offdiag, 
			m2_diag, m3_a, m3_b, m3_c);
#else
		tmp += MUL_HELPER_1XX(left_a1_offdiag, right_a1_offdiag, m2_diag, m3_b);
#endif
		
		/*
         * M_1 (x) A_2 (x) M_3
         */ 
		
#if M1_TRIDIAG
#if M3_TRIDIAG
		tmp += MUL_HELPER_123(left_m1_offdiag, right_m1_offdiag, 
			left_a2_offdiag, a2_diag, right_a2_offdiag, m3_a, m3_b, m3_c);
#else
		tmp += MUL_HELPER_12X(left_m1_offdiag, right_m1_offdiag, 
			left_a2_offdiag, a2_diag, right_a2_offdiag, m3_b);
#endif
#endif
		/*
         * M_1 (x) M_2 (x) A_3
         */
		
#if M1_TRIDIAG
#if M2_TRIDIAG
		tmp += MUL_HELPER_123(left_m1_offdiag, right_m1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, a3_a, a3_b, a3_c);
#else
		tmp += MUL_HELPER_1X3(left_m1_offdiag, right_m1_offdiag, 
			m2_diag, a3_a, a3_b, a3_c);
#endif
#endif
		
		/*
         * c * (M_1 (x) M_2 (x) M_3)
         */
		
#if M1_TRIDIAG
#if M2_TRIDIAG && M3_TRIDIAG
		tmp += MUL(_ch, MUL_HELPER_123(left_m1_offdiag, right_m1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, m3_a, m3_b, m3_c));
#elif M2_TRIDIAG
		tmp += MUL(_ch, MUL_HELPER_12X(left_m1_offdiag, right_m1_offdiag, 
			left_m2_offdiag, m2_diag, right_m2_offdiag, m3_b));
#elif M3_TRIDIAG
		tmp += MUL(_ch, MUL_HELPER_1X3(left_m1_offdiag, right_m1_offdiag, 
			m2_diag, m3_a, m3_b, m3_c));
#else
		tmp += MUL(_ch, MUL_HELPER_1XX(left_m1_offdiag, right_m1_offdiag, 
			m2_diag, m3_b));
#endif
#endif
		
					
#undef MUL_HELPER_123
#undef MUL_HELPER_1X3
#undef MUL_HELPER_12X
#undef MUL_HELPER_1XX

		// Generate the right-hand side vector
		if(0 <= elem3_id) {
			_var data;
		   	data  = MUL(e_vec1, elem1[d]);
			data += MUL(e_vec2, elem2[d]);
			data += MUL(e_vec3, elem3[d]);

			g[d] = data - tmp;
		} else if(0 <= elem2_id) {
			_var data;
		   	data  = MUL(e_vec1, elem1[d]);
			data += MUL(e_vec2, elem2[d]);

			g[d] = data - tmp;
		} else if(0 <= elem1_id) {
			_var data;
			data  = MUL(e_vec1, elem1[d]);

			g[d] = data - tmp;

		}
 
	}

#endif
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


/* The second half of the second stage, step A. Performs the first step of the
 * recursive summation. Each solution vector is multiplied by the corresponding
 * eigenvector component and the result of each partial sum is stored into a
 * temporary buffer. 
 * Arguments:
 *  _v			v-vector
 *  _tmp		Temporary buffer ()
 *  _guide		Guiding information, i.e., what each work group should do
 *  _eigen		Eigenvector components
 *  _i			Recursion index
 *  _
 * Work groups:
 *  get_num_groups(0)	The total number of partial sums
 *  get_num_groups(1) 	If LEVEL2 is enabled, then this is the number of level
 * 						2 subproblem. Otherwise, it is equal to N2.
 *  get_num_groups(2)  	Number of work groups per N3-vector
 * 						  = LX_STAGE_12A_WG_PER_VECTOR
 */
__attribute__((reqd_work_group_size(LX_STAGE_22A_WG_SIZE, 1, 1)))
__kernel void lx_stage_22a(
#if LEVEL2
		__global       _var_t *_v,
#else
		__global       _var   *_v,
#endif
		__global       _var   *_tmp,
		_stage12a_guide       *_guide,
		__global const _var   *_eigen,
						int    _i, 
                        int    _max_sum_size) {

	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);

	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1);
	
#if 1 < LX_STAGE_12A_WG_PER_VECTOR
	const int gid2 = get_group_id(2);
#else
	const int gid2 = 0;
#endif

	// Determine the index of the vector sum. At this point, it is (incorrectly)
	// assumed that each sum is of the size 4^i. This divergency is handled 
	// later.
	const int s_id = gid0 / DIVCEIL(_max_sum_size, LX_STAGE_Y2A_MAX_SUM_SIZE);
	
	// Calculate the index of the partial sum (trying to avoid mod-operations)
	const int ps_id = gid0 % DIVCEIL(_max_sum_size, LX_STAGE_Y2A_MAX_SUM_SIZE);

#if LX_SHARED_ISOLATED_ACCESS

	__local int iwork[4];

	// Loads guiding information
	if(local_id < 4)
		iwork[local_id] = GET_STAGE22A_GUIDE_ELEM(_guide, s_id, local_id, _i);
	barrier(CLK_LOCAL_MEM_FENCE);

	const int upper_bound = iwork[0];
	const int lower_bound = iwork[1];
	const int eigen_id = iwork[2];
	const int eigen_num = iwork[3];

#else

	const int upper_bound = GET_STAGE22A_GUIDE_ELEM(_guide, s_id, 0, _i);
	const int lower_bound = GET_STAGE22A_GUIDE_ELEM(_guide, s_id, 1, _i);
	const int eigen_id = GET_STAGE22A_GUIDE_ELEM(_guide, s_id, 2, _i);
	const int eigen_num = GET_STAGE22A_GUIDE_ELEM(_guide, s_id, 3, _i);
	
#endif

	// Calculate partial sum boundaries
	const int s_begin = ps_id * LX_STAGE_Y2A_MAX_SUM_SIZE;
	const int s_end = 
		min(lower_bound - upper_bound, (ps_id+1)*LX_STAGE_Y2A_MAX_SUM_SIZE);
	const int s_size = s_end - s_begin;
	
	// Return if the size of the partial sum is zero
	if(s_size <= 0)
		return;

#if LEVEL2
	
	// Calculates the location of the v-vector
	_var_global_array v = SUB_GLOBAL_ARRAY(
		GLOBAL_ARRAY(_v, L3_LDF1*L3_LDF2*L3_LDF3), 
			(gid1*L3_LDF2+upper_bound)*L3_LDF3);
	
	// Calculates the location of the temporary memory buffer
	__global _var *tmp = 
		_tmp + (gid1*TMP_PER_L2+gid0)*L2_LDF3;
	
#else

	__global _var *v = _v + (upper_bound*L2_LDF2+gid1)*L2_LDF3;
	
	__global _var *tmp = _tmp + (gid0*L1_LDF2+gid1)*L1_LDF3;

#endif	

	const int begin = gid2 * LX_STAGE_22A_WG_SIZE;
	const int jump = LX_STAGE_22A_WG_PER_VECTOR * LX_STAGE_22A_WG_SIZE;

	// Computes the u-vector (u = (R^~ W (x) I_N3)v)
	for(int d = begin+local_id; D*d < N3; d += jump) {
		_varD res = 0.0;
		for(int i = s_begin; i < s_end; i++) {
			_var eigen =
				GET_EIGEN(_eigen, eigen_id+i, eigen_num);
#if LEVEL2
			res += MULD(eigen, GVLOADD(i*L3_LDF3/D+d, v));
#else
			res += MULD(eigen, VLOADD(i*L2_LDF2*L2_LDF3/D+d, v));
#endif
		}
		VSTOREDD(res, d, tmp);
	}
}


__attribute__((reqd_work_group_size(LX_STAGE_22C_WG_SIZE, 1, 1)))
__kernel void lx_stage_22c(
		__global       _var   *_f,
		__global       _var   *_tmp,
		_stage22c_guide       *_guide,
						int    _i
		) {

	const int local_id = get_local_id(0);
	const int local_size = get_local_size(0);

	const int gid0 = get_group_id(0);
	const int gid1 = get_group_id(1);

#if 1 < LX_STAGE_22C_WG_PER_VECTOR
	const int gid2 = get_group_id(2);
#else
	const int gid2 = 0;
#endif
	
#if LX_SHARED_ISOLATED_ACCESS
	
	/*
	 * Load guiding information
	 */
	
	__local int iwork[3];
	if(local_id < 2)
		iwork[local_id] = GET_STAGE22C_GUIDE_ELEM(_guide, gid0, local_id, _i);
	barrier(CLK_LOCAL_MEM_FENCE);

	const int update_loc = iwork[0];
	const int sum_loc = iwork[1];
	
#else // if LX_SHARED_ISOLATED_ACCESS ... ***

	const int update_loc = GET_STAGE22C_GUIDE_ELEM(_guide, gid0, 0, _i);
	const int sum_loc = GET_STAGE22C_GUIDE_ELEM(_guide, gid0, 1, _i);
	
#endif

#if LEVEL2
	__global _var *f =  _f + (gid1*L2_LDF2+update_loc)*L2_LDF3;
	
	__global _var *tmp = 
		_tmp + (gid1*TMP_PER_L2+sum_loc)*L2_LDF3;
	
#else

	__global _var *f = _f + (update_loc*L1_LDF2+gid1)*L1_LDF3;
	
	__global _var *tmp = _tmp + (sum_loc*L1_LDF2+gid1)*L1_LDF3;

#endif	

	const int begin = gid2 * LX_STAGE_22C_WG_SIZE;
	const int jump = LX_STAGE_22C_WG_PER_VECTOR * LX_STAGE_22C_WG_SIZE;

	for(int d = begin+local_id; D*d < N3; d += jump)
		VSTOREDD(VLOADD(d, tmp), d, f);
	
}