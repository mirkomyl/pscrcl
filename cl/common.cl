/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 *
 * This includes constants and macros common for all OpenCL source files.
 *
 * Precompiler parameters:
 * 
 * DOUBLE:			Enable double precision
 * COMPLEX:			Enable complex numbers
 * AMD_FP64:		Enable AMD double precision support (not tested)
 * D:				Vector type width (1, 2, 4)
 * HILODOUBLE:		Store the high and low bits separately when using local 
 * 					memory and double precision
 *
 * Data types:
 * 	_var_t:						Basic data type (float/double)
 * 	_var2_t:					Basic 2-vector (float2/double2)
 * 	_var:						Data type (real => float/double; complex =>
 * 								float2/double2)
 * 	_var2:						2-component vector data type (real => float2/
 * 								double2; complex => float2)
 * 	VLOADD(idx, buff):			Returns buff[idx], normal storage method
 * 	VSTOREDD(data, idx, buff):	Sets buff[idx] = data, normal storage method
 *
 * Vector manipulation and data type specific macros/functions:
 *	VAR2_S0(x):					Returns x.s0
 *	VAR2_S1(x):					Returns x.s1
 *	SET_VAR2_S0(x,y):			Sets y.s0 = x
 *	SET_VAR2_S1(x,y):			Sets y.s1 = x
 *	MUL(x,y):					Returns x*y
 *	DIV(x,y):					Returns x/y
 *
 *	_var_global_array:			Global memory buffer handle
 *	GLOAD(idx, buff):			Returns buff[idx]
 *	GSTORE(data, idx, buff):	Sets buff[idx] = data
 *	GLOAD2(idx, buff):			Returns buff[idx] (2-component vector)
 *	GSTORE2(data, idx, buff):	Sets buff[idx] = data (2-component vector)
 *
 *	GVLOADD(idx, buff):			Returns buff[idx] (D-component vector)
 *	GVSTOREDD(data, idx, buff):	Sets buff[idx] = data (D-component vector)
 *	MULD(x,y):					Returns x*y (scalar, D-component vector)
 *	DIVD(x,y):					Returns x/y (D-component vector, scalar)
 *	GLOBAL_ARRAY(buff, offset):	Creates a global memory buffer handle
 *	SUB_GLOBAL_ARRAY(buff, \
 *		idx):					Creates a sub global memory buffer handle
 *
 *	_var_local_array:			Local memory buffer handle
 *	LLOAD(idx, buff):			Returns buff[idx]
 *	LSTORE(data, idx, buff):	Sets buff[idx] = data
 *	LSTORE2(data, idx, buff):	Sets buff[idx] = data (2-component vector)
 *	ALLOC_LOCAL_MEM(name, \
 *		tmp_name, size): 		Creates a local memory buffer handle
 *	SUB_LOCAL_ARRAY(buff, idx):	Creates a sub local memory buffer handle
 *
 * PSCR related macros:
 *	PSCR_GET_EIGEN(buff,vec_id,comp,n,i):	TODO
 *	PSCR_GET_GUIDE2(buff,g_id,comp,n,i):	TODO
 *	PSCR_GET_GUIDE3(buff,g_id,comp,n,i):	TODO
 *	SLB(i,k): 								TODO
 *	PSCR_GET_BOUND(buff,index,k,i):			TODO
 *
 */

//
// Coefficient matrix buffer strides
//

#if M1_TRIDIAG
#define X1_BUFFER_SIZE		4*L1_MAT_LDF
#else
#define X1_BUFFER_SIZE		3*L1_MAT_LDF
#endif
#define A1_DIAG_STRIDE		0*L1_MAT_LDF
#define A1_CODIAG_STRIDE	1*L1_MAT_LDF
#define M1_DIAG_STRIDE		2*L1_MAT_LDF
#define M1_CODIAG_STRIDE	3*L1_MAT_LDF

#if M2_TRIDIAG
#define X2_BUFFER_SIZE		4*L2_MAT_LDF
#else
#define X2_BUFFER_SIZE		3*L2_MAT_LDF
#endif
#define A2_DIAG_STRIDE		0*L2_MAT_LDF
#define A2_CODIAG_STRIDE	1*L2_MAT_LDF
#define M2_DIAG_STRIDE		2*L2_MAT_LDF
#define M2_CODIAG_STRIDE	3*L2_MAT_LDF

#if M3_TRIDIAG
#define X3_BUFFER_SIZE		4*L3_MAT_LDF
#else
#define X3_BUFFER_SIZE		3*L3_MAT_LDF
#endif
#define A3_DIAG_STRIDE		0*L3_MAT_LDF
#define A3_CODIAG_STRIDE	1*L3_MAT_LDF
#define M3_DIAG_STRIDE		2*L3_MAT_LDF
#define M3_CODIAG_STRIDE	3*L3_MAT_LDF


//
// Global math macros
//

#define POW4(a) (1<<(2*(a)))
#define POW2(a) (1<<(a))
#define LOG4(a) ((int)(log((_var_t)a)/log((_var_t)4)))
#define LOG2(a) ((int)log2((_var_t)a))

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define K_2(n) (LOG2(n)+1)
#define K_4(n) (LOG4(n)+1)

// a / 2^x
#define DIVBYPOW2(a,x) ((a) >> (x))

// a / 4^x
#define DIVBYPOW4(a,x) ((a) >> (2*(x))

// a % 2^x
#define MODBYPOW2(a, x) a & ((1U << x) - 1)

// // a % 4^x
#define MODBYPOW4(a, x) a & ((1U << (2*x)) - 1)

// TODO: Make sure that the GPU is made my NVIDIA!
//#define NVIDIA 1
//#define NVIDIA_SHFL 1

// Ceil of a / b
int DIVCEIL(int a, int b) {
	return a % b != 0 ? a / b + 1 : a / b;
}

int NEXTPOW2(int v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

//
// Precision and vector data types
//

#if DOUBLE

#if AMD_FP64
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef double _var_t;
typedef double2 _var2_t;

#if NEWTON_DIV
union invType {
	double f;
	ulong i;
	struct {
		uint e1;
		uint e2;
	} ii;
};
double inv(union invType b) {
	union invType md, x;
	
	md.i = (b.i & 0x000fffffffffffff) | 0xbfe0000000000000;
	
	x.f = (48.0/17.0) + (32.0/17.0) * md.f;
	
	x.f = x.f + x.f * (1.0 + md.f * x.f);
	x.f = x.f + x.f * (1.0 + md.f * x.f);
	x.f = x.f + x.f * (1.0 + md.f * x.f);
	x.f = x.f + x.f * (1.0 + md.f * x.f);

	x.ii.e2 += 0x3fe00000 - (b.ii.e2 & 0x7ff00000);
	x.i |= b.i & 0x8000000000000000;
	
	return x.f;
}
#define INV(b) inv((union invType)(b))
#else
#define INV(b) (1.0/(b))
#endif

#if COMPLEX
typedef double2 _var;
typedef double4 _var2;

#if D == 1
typedef _var _varD;
#define VLOADD(a, b) ((b)[a])
#define VSTOREDD(a, b, c) (c[b] = a)
#else
#error "Invalid D"
#endif

#if NVIDIA_SHFL
_var shfl(_var value, int line) {
	const uint bound = 0x1f;
	
	uint real_lo2, real_hi2;
	asm volatile("shfl.idx.b32 %0, %1, %2, %3;" : "=r"(real_lo2) : "r"(as_uint2(value.x).x), "r"(line), "r"(bound));
	asm volatile("shfl.idx.b32 %0, %1, %2, %3;" : "=r"(real_hi2) : "r"(as_uint2(value.x).y), "r"(line), "r"(bound));

	uint imag_lo2, imag_hi2;
	asm volatile("shfl.idx.b32 %0, %1, %2, %3;" : "=r"(imag_lo2) : "r"(as_uint2(value.y).x), "r"(line), "r"(bound));
	asm volatile("shfl.idx.b32 %0, %1, %2, %3;" : "=r"(imag_hi2) : "r"(as_uint2(value.y).y), "r"(line), "r"(bound));

	//real_lo2 = as_uint2(value.x).x;
	real_hi2 = as_uint2(value.x).y;
	imag_lo2 = as_uint2(value.y).x;
	imag_hi2 = as_uint2(value.y).y;

	_var ret;
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(ret.x) : "r"(real_lo2), "r"(real_hi2));
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(ret.y) : "r"(imag_lo2), "r"(imag_hi2));
	return ret;
}
#endif


#else // !COMPLEX

typedef double _var;
typedef double2 _var2;

#if D == 1
typedef _var _varD;
#define VLOADD(a, b) ((b)[a])
#define VSTOREDD(a, b, c) ((c)[b] = a)
#elif D == 2
typedef _var2 _varD;
#define VLOADD(a, b) vload2(a, b)
#define VSTOREDD(a, b, c) vstore2(a, b, c)
#else
#error "Invalid D"
#endif

_var shfl(_var value, int line) {
	uint lo, hi;
	asm volatile("mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(value));

	uint lo2, hi2;
	asm volatile("shfl.mode.b32 %0, %1, %2, 32;" : "=r"(lo2) : "r"(lo), "r"(line));
	asm volatile("shfl.mode.b32 %0, %1, %2, 32;" : "=r"(hi2) : "r"(hi), "r"(line));
	
	_var x;
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo2), "r"(hi2));
	return x;
}

#endif // COMPLEX else !COMPLEX

#else // !DOUBLE

typedef float _var_t;
typedef float2 _var2_t;

#define INV(b) (1.0/(b))

#if COMPLEX
typedef float2 _var;
typedef float4 _var2;

#if D == 1
typedef _var _varD;
#define VLOADD(a, b) ((b)[a])
#define VSTOREDD(a, b, c) ((c)[b] = a)
#elif D == 2
typedef _var2 _varD;
#define VLOADD(a, b) vload4(a, (_var_t*)b)
#define VSTOREDD(a, b, c) vstore4(a, b, (_var_t*)c)
#else
#error "Invalid D"
#endif

#if NVIDIA_SHFL
_var shfl(_var value, int line) {
	_var x;
	asm volatile("shfl.mode.b32 %0, %1, %2, 32;" : "=r"(x.s0) : "r"(value.s0), "r"(line));
	asm volatile("shfl.mode.b32 %0, %1, %2, 32;" : "=r"(x.s0) : "r"(value.s1), "r"(line));
	return x;
}
#endif

#else // !COMPLEX

typedef float _var;
typedef float2 _var2;

#if D == 1
typedef _var _varD;
#define VLOADD(a, b) ((b)[a])
#define VSTOREDD(a, b, c) ((c)[b] = a)
#elif D == 2
typedef _var2 _varD;
#define VLOADD(a, b) vload2(a, b)
#define VSTOREDD(a, b, c) vstore2(a, b, c)
#elif D == 4
typedef float4 _varD;
#define VLOADD(a, b) vload4(a, b)
#define VSTOREDD(a, b, c) vstore4(a, b, c)
#else
#error "Invalid D"
#endif

#if NVIDIA_SHFL
_var shfl(_var value, int line) {
	_var x;
	asm volatile("shfl.mode.b32 %0, %1, %2, 32;" : "=r"(x) : "r"(value), "r"(line));
	return x;
}
#endif

#endif // COMPLEX else !COMPEX

#endif // DOUBLE else !DOUBLE

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

//
// Vector manipulation and data type specific macros/functions
//

#if COMPLEX

_var VAR2_S0(_var2 x) {
	return ((_var) (x.s0, x.s1));
}
_var VAR2_S1(_var2 x) {
	return ((_var) (x.s2, x.s3));
}
void SET_VAR2_S0(_var x, _var2* y) {
	(*y).s0 = x.s0;
	(*y).s1 = x.s1;
}
void SET_VAR2_S1(_var x, _var2* y) {
	(*y).s2 = x.s0;
	(*y).s3 = x.s1;
}
_var MUL(_var a, _var b) {
	return ((_var)(a.s0*b.s0 - a.s1*b.s1, a.s0*b.s1 + a.s1*b.s0));
}
_var DIV(_var a, _var b) {
	return ((_var)(((_var)(a.s0*b.s0 + a.s1*b.s1, - a.s0*b.s1 + a.s1*b.s0))*INV(b.s0*b.s0 + b.s1*b.s1)));
}

typedef struct {
	__global _var_t *real;
	__global _var_t *imag;
} _var_global_array;

_var GLOAD(int idx, _var_global_array buff) {
	return ((_var) ((buff).real[idx], (buff).imag[idx]));
}

void GSTORE(_var data, int idx, _var_global_array buff) {
	buff.real[idx] = data.s0;
	buff.imag[idx] = data.s1;
}
_var2 GLOAD2(int idx, _var_global_array buff) {
	_var2_t real = vload2(idx, buff.real);
	_var2_t imag = vload2(idx, buff.imag);
	return ((_var2)(real.s0, imag.s0, real.s1, imag.s1));
}

void GSTORE2(_var2 data, int idx, _var_global_array buff) {
	vstore2((_var2_t)(data.s0, data.s2), idx, buff.real);
	vstore2((_var2_t)(data.s1, data.s3), idx, buff.imag);
}

#if DOUBLE || D == 1
#define GVLOADD(idx, buff) GLOAD(idx, buff)
#define GVSTOREDD(data, idx, buff) GSTORE(data, idx, buff)
#define MULD(a, b) MUL(a, b)
#define DIVD(a, b) DIV(a, b)
#elif D == 2
#define GVLOADD(idx, buff) GLOAD2(idx, buff)
#define GVSTOREDD(data, idx, buff) GSTORE2(data, idx, buff)
_varD MULD(_var a, _varD b) {
	return ((_varD)(MUL(a, VAR2_S0(b)), MUL(a, VAR2_S1(b))));
}
_varD DIVD(_varD a, _var b) {
	return ((_varD)(DIV(VAR2_S0(a), b), DIV(VAR2_S1(a), b)));
}
#endif

_var_global_array GLOBAL_ARRAY(__global _var_t *buff, int offset) {
	_var_global_array tmp;
	tmp.real = buff;
	tmp.imag = buff + offset;
	return tmp;
}

_var_global_array SUB_GLOBAL_ARRAY(_var_global_array buff, int offset) {
	_var_global_array tmp;
	tmp.real = buff.real + offset;
	tmp.imag = buff.imag + offset;
	return tmp;
}

#if DOUBLE && HILODOUBLE

//
// Store double in two parts
//

typedef struct {
	__local uint *real_lo;
	__local uint *real_hi;
	__local uint *imag_lo;
	__local uint *imag_hi;
} _var_local_array;

_var LLOAD(int idx, _var_local_array buff) {
#if NVIDIA
	_var x;
	uint real_hi = (buff).real_hi[idx];
	uint real_lo = (buff).real_lo[idx];
	uint imag_hi = (buff).imag_hi[idx];
	uint imag_lo = (buff).imag_lo[idx];
	
	asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x.s0) : "r"(real_lo), "r"(real_hi));
	asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x.s1) : "r"(imag_lo), "r"(imag_hi));
	return x;
#else
	return ((_var) (
			as_double((((ulong)(buff).real_hi[idx]) << 32) |
					((ulong)(buff).real_lo[idx])),
			as_double((((ulong)(buff).imag_hi[idx]) << 32) |
					((ulong)(buff).imag_lo[idx]))));
#endif
}

void LSTORE(_var data, int idx, _var_local_array buff) {
	buff.real_lo[idx] = as_uint2(data.x).x;
	buff.real_hi[idx] = as_uint2(data.x).y;
	buff.imag_lo[idx] = as_uint2(data.y).x;
	buff.imag_hi[idx] = as_uint2(data.y).y;
}
void LSTORE2(_var2 data, int idx, _var_local_array buff) {
	vstore2((uint2)((uint)as_uint2(data.s0).x,
			(uint)as_uint2(data.s2).x), idx, buff.real_lo);
	vstore2((uint2)((uint)as_uint2(data.s0).y,
			(uint)as_uint2(data.s2).y), idx, buff.real_hi);
	vstore2((uint2)((uint)as_uint2(data.s1).x,
			(uint)as_uint2(data.s3).x), idx, buff.imag_lo);
	vstore2((uint2)((uint)as_uint2(data.s1).y,
			(uint)as_uint2(data.s3).y), idx, buff.imag_hi);
}

#define ALLOC_LOCAL_MEM(name, tmp_name, size) 	\
	__local int tmp_name[4*(size)];				\
	_var_local_array name;						\
	name.real_lo = tmp_name + 0*size;			\
	name.real_hi = tmp_name + 1*size;			\
	name.imag_lo = tmp_name + 2*size;			\
	name.imag_hi = tmp_name + 3*size

_var_local_array SUB_LOCAL_ARRAY(_var_local_array buff, int idx) {
	_var_local_array tmp;
	tmp.real_lo = buff.real_lo + idx;
	tmp.real_hi = buff.real_hi + idx;
	tmp.imag_lo = buff.imag_lo + idx;
	tmp.imag_hi = buff.imag_hi + idx;
	return tmp;
}

#else // !HILODOUBLE
typedef struct {
	__local _var_t *real;
	__local _var_t *imag;
} _var_local_array;
_var LLOAD(int idx, _var_local_array buff) {
	return ((_var) ((buff).real[idx], (buff).imag[idx]));
}
void LSTORE(_var data, int idx, _var_local_array buff) {
	buff.real[idx] = data.s0;
	buff.imag[idx] = data.s1;
}
void LSTORE2(_var2 data, int idx, _var_local_array buff) {
	vstore2((_var2_t)(data.s0, data.s2), idx, buff.real);
	vstore2((_var2_t)(data.s1, data.s3), idx, buff.imag);
}

#define ALLOC_LOCAL_MEM(name, tmp_name, size)	\
	__local _var_t tmp_name[2*(size)];			\
	_var_local_array name;						\
	name.real = tmp_name;						\
	name.imag = tmp_name + size

_var_local_array SUB_LOCAL_ARRAY(_var_local_array buff, int offset) {
	_var_local_array sub_buff;
	sub_buff.real = buff.real+offset;
	sub_buff.imag = buff.imag+offset;
	return sub_buff;
}

#endif // HILODOUBLE else !HILODOUBLE

//
// Real numbers
//
#else // !COMPLEX

#define VAR2_S0(x) ((x).s0)
#define VAR2_S1(x) ((x).s1)
void SET_VAR2_S0(_var x, _var2* y) {(*y).s0 = x; }
void SET_VAR2_S1(_var x, _var2* y) {(*y).s1 = x; }
#define MUL(a, b) ((a)*(b))
#define MULD(a, b) ((a)*(b))

#if NEWTON_DIV
_var DIV(_var a, _var b) {
	return a * INV(b);
}
#define DIVD(a, b) DIV(a, b)
#else
#define DIV(a, b) ((a)/(b))
#define DIVD(a, b) ((a)/(b))
#endif

typedef __global _var_t* _var_global_array;
#define GLOAD(idx, buff) ((buff)[idx])
#define GSTORE(data, idx, buff) ((buff)[idx] = (data))
#define GLOAD2(idx, buff) vload2(idx, buff)
#define GSTORE2(data, idx, buff) vstore2(data, idx, buff)

#if D == 1
#define GVLOADD(idx, buff) GLOAD(idx, buff)
#define GVSTOREDD(data, idx, buff) GSTORE(data, idx, buff)
#elif D == 2
#define GVLOADD(idx, buff) GLOAD2(idx, buff)
#define GVSTOREDD(data, idx, buff) GSTORE2(data, idx, buff)
#elif D == 4
#define GVLOADD(idx, buff) vstore4(idx, buff)
#define GVSTOREDD(data, idx, buff) vstore4(data, idx, buff)
#endif

#define GLOBAL_ARRAY(buff, offset) (buff)
#define SUB_GLOBAL_ARRAY(buff, idx) ((buff)+(idx))

//
// Store double in two parts
//
#if DOUBLE && HILODOUBLE
typedef struct {
	__local uint* lo;
	__local uint* hi;
} _var_local_array;
double LLOAD(int idx, _var_local_array buff) {
#if NVIDIA
	_var x;
	uint hi = (buff).hi[idx];
	uint lo = (buff).lo[idx];
	
	asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi));
	return x;
#else
	return as_double((((ulong)(buff).hi[idx]) << 32) |
			((ulong)(buff).lo[idx]));
#endif
}
void LSTORE(_var data, int idx, _var_local_array buff) {
	(buff).lo[idx] = as_uint2(data).x;
	(buff).hi[idx] = as_uint2(data).y;
}
void LSTORE2(_var2 data, int idx, _var_local_array buff) {
	vstore2((uint2)(
			(uint)as_uint2(data.x).x,
			(uint)as_uint2(data.y).x), idx, buff.lo);
	vstore2((uint2)(
			(uint)as_uint2(data.x).y,
			(uint)as_uint2(data.y).y), idx, buff.hi);
}

#define ALLOC_LOCAL_MEM(name, tmp_name, size)	\
	__local int tmp_name[2*size];				\
	_var_local_array name;						\
	name.lo = tmp_name;							\
	name.hi = tmp_name + size

_var_local_array SUB_LOCAL_ARRAY(_var_local_array buff, int idx) {
	_var_local_array sub_buff;
	sub_buff.hi = buff.hi+idx;
	sub_buff.lo = buff.lo+idx;
	return sub_buff;
}

#else // !HILODOUBLE
#define LLOAD(idx, buff) ((buff)[idx])
#define LSTORE(data, idx, buff) ((buff)[idx] = data)
#define LSTORE2(data, idx, buff) vstore2(data, idx, buff)
typedef __local _var_t* _var_local_array;

#define ALLOC_LOCAL_MEM(name, tmp_name, size)	\
	__local _var_t name[size];

#define SUB_LOCAL_ARRAY(buff, idx) ((buff)+(idx))

#endif // END !HILODOUBLE

#endif

_var2 MUL2(_var a, _var2 b) {
	return ((_var2)(MUL(a, VAR2_S0(b)), MUL(a, VAR2_S1(b))));
}
_var2 DIV2(_var2 a, _var b) {
	return ((_var2)(MUL(VAR2_S0(a), b), MUL(VAR2_S1(a), b)));
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*
 * Returns the coefficient matrix component vectors.
 * Arguments:
 *   _packet		A buffer which contains the coefficient matrix components
 *   _a3_diag		Returns a pointer to a buffer containing the diagonal
 *   		  		elements of the matrix A3.
 *   _a3_offdiag	Returns a pointer to a buffer containing the off-diagonal
 *   				elements of the matrix A3.
 *   _m3_diag		Returns a pointer to a buffer containing the diagonal
 *   				elements of the matrix M3.
 *   _m3_offdiag	Returns a pointer to a buffer containing the off-diagonal
 *   				elements of the matrix M3.
 */
void get_coef_matrix_components(
		__global _var_t   *_packed,
		_var_global_array *_a3_diag,
		_var_global_array *_a3_offdiag,
		_var_global_array *_m3_diag,
		_var_global_array *_m3_offdiag) {

	_var_global_array diags = GLOBAL_ARRAY(_packed, X3_BUFFER_SIZE);

	if (_a3_diag)
		*_a3_diag = SUB_GLOBAL_ARRAY(diags, A3_DIAG_STRIDE);
	if (_a3_offdiag)
		*_a3_offdiag = SUB_GLOBAL_ARRAY(diags, A3_CODIAG_STRIDE);
	if (_m3_diag)
		*_m3_diag = SUB_GLOBAL_ARRAY(diags, M3_DIAG_STRIDE);
#if M3_TRIDIAG
	if (_m3_offdiag)
		*_m3_offdiag = SUB_GLOBAL_ARRAY(diags, M3_CODIAG_STRIDE);
#endif
}
