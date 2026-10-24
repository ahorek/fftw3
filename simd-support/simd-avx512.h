/*
 * Copyright (c) 2003, 2007-11 Matteo Frigo
 * Copyright (c) 2003, 2007-11 Massachusetts Institute of Technology
 *
 * AVX-512 support implemented by Romain Dolbeau.
 * Romain Dolbeau hereby places his modifications in the public domain.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#if defined(FFTW_LDOUBLE) || defined(FFTW_QUAD)
#error "AVX-512 vector instructions only works in single or double precision"
#endif

#ifdef FFTW_SINGLE
#  define DS(d,s) s /* single-precision option */
#  define SUFF(name) name ## _ps
#  define SCAL(x) x ## f
#else /* !FFTW_SINGLE */
#  define DS(d,s) d /* double-precision option */
#  define SUFF(name) name ## _pd
#  define SCAL(x) x
#endif /* FFTW_SINGLE */

#define SIMD_SUFFIX  _avx512  /* for renaming */
#define VL DS(4, 8)        /* SIMD complex vector length */
#define SIMD_VSTRIDE_OKA(x) ((x) == 2)
#define SIMD_STRIDE_OKPAIR SIMD_STRIDE_OK

#if defined(__GNUC__) && !defined(__AVX512F__) /* sanity check */
#error "compiling simd-avx512.h without avx-512f support"
#endif

#if !defined(HAVE_AVX2)
#warning "You should probably enable AVX2 with --enable-avx2 for AVX-512"
#endif

#include <immintrin.h>

typedef DS(__m512d, __m512) V;

#define VLIT(re, im) DS(SUFF(_mm512_setr)(im, re, im, re, im, re, im, re),SUFF(_mm512_setr)(im, re, im, re, im, re, im, re, im, re, im, re, im, re, im, re))
#define VLIT1(val) SUFF(_mm512_set1)(val)
#define LDK(x) x
#define DVK(var, val) V var = VLIT1(val)
#define VZERO SUFF(_mm512_setzero)()

#define VDUPL(x) DS(_mm512_movedup_pd(x),_mm512_moveldup_ps(x))
#define VDUPH(x) DS(_mm512_unpackhi_pd(x, x),_mm512_movehdup_ps(x))
#define FLIP_RI(x) SUFF(_mm512_shuffle)(x, x, DS(0x55,0xB1))
#define VCONJ(x) SUFF(_mm512_fmsubadd)(VZERO, VZERO, x)
static inline V VBYI(V x)
{
     return FLIP_RI(VCONJ(x));
}

#define VADD(a,b) SUFF(_mm512_add)(a,b)
#define VSUB(a,b) SUFF(_mm512_sub)(a,b)
#define VMUL(a,b) SUFF(_mm512_mul)(a,b)
#define VFMA(a, b, c)  SUFF(_mm512_fmadd)(a, b, c)
#define VFMS(a, b, c)  SUFF(_mm512_fmsub)(a, b, c)
#define VFNMS(a, b, c) SUFF(_mm512_fnmadd)(a, b, c)
#define VFMAI(b, c)    SUFF(_mm512_fmaddsub)(VLIT1(1.), c, FLIP_RI(b))
#define VFNMSI(b, c)   SUFF(_mm512_fmsubadd)(VLIT1(1.), c, FLIP_RI(b))
#define VFMACONJ(b,c)  SUFF(_mm512_fmsubadd)(VLIT1(1.), c, b)
#define VFMSCONJ(b,c)  SUFF(_mm512_fmsubadd)(VLIT1(-1.), c, b)
#define VFNMSCONJ(b,c) SUFF(_mm512_fmaddsub)(VLIT1(1.), c, b)

static inline V LDA(const R *x, INT ivs, const R *aligned_like) {
  (void)aligned_like; /* UNUSED */
  (void)ivs; /* UNUSED */
  return SUFF(_mm512_loadu)(x);
}
static inline void STA(R *x, V v, INT ovs, const R *aligned_like) {
  (void)aligned_like; /* UNUSED */
  (void)ovs; /* UNUSED */
  SUFF(_mm512_storeu)(x, v);
}

/*
 * Strided loads/stores.
 *
 * The original implementation used _mm512_i32gather / _mm512_i32scatter
 * for every LD/ST.  Gathers are slow on most AVX-512 parts (and much
 * slower still with the "Downfall"/GDS microcode mitigation on Intel),
 * and scatters are very slow on every AVX-512 implementation.  LD/ST
 * are used by all n1, n2, q1 and t1u codelets, i.e. by the leaf
 * transforms of essentially every plan.
 *
 * The data accessed is not really "gathered": each element is a
 * contiguous complex number (64 bits in single, 128 bits in double), so
 * the vector can be assembled from VL plain loads with a few cheap
 * inserts, and stored with VL plain stores.
 *
 * Only AVX-512F instructions are used (no DQ/BW/VL requirement).
 *
 * WARNING: the extra_iter hack in dftw-direct.c depends upon the store
 * of the lowest element occurring after the stores of the higher ones
 * (see simd-avx2.h), so all ST variants below store element 0 last.
 */

#ifdef FFTW_SINGLE

#define LOADL(addr, val) _mm_loadl_pi(val, (const __m64 *)(addr))
#define LOADH(addr, val) _mm_loadh_pi(val, (const __m64 *)(addr))
#define STOREL(addr, val) _mm_storel_pi((__m64 *)(addr), val)
#define STOREH(addr, val) _mm_storeh_pi((__m64 *)(addr), val)

static inline __m256 LD4u(const R *x, INT ivs)
{
     __m128 l = LOADL(x, _mm_undefined_ps());
     __m128 h = LOADL(x + 2 * ivs, _mm_undefined_ps());
     l = LOADH(x + ivs, l);
     h = LOADH(x + 3 * ivs, h);
     return _mm256_insertf128_ps(_mm256_castps128_ps256(l), h, 1);
}

static inline V LDu(const R *x, INT ivs, const R *aligned_like)
{
     (void)aligned_like; /* UNUSED */
     return _mm512_castpd_ps(
	  _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_castps_pd(LD4u(x, ivs))),
			     _mm256_castps_pd(LD4u(x + 4 * ivs, ivs)), 1));
}

static inline void STu(R *x, V v, INT ovs, const R *aligned_like)
{
     (void)aligned_like; /* UNUSED */
     __m128 q3 = _mm512_extractf32x4_ps(v, 3);
     __m128 q2 = _mm512_extractf32x4_ps(v, 2);
     __m128 q1 = _mm512_extractf32x4_ps(v, 1);
     __m128 q0 = _mm512_castps512_ps128(v);
     STOREH(x + 7 * ovs, q3);
     STOREL(x + 6 * ovs, q3);
     STOREH(x + 5 * ovs, q2);
     STOREL(x + 4 * ovs, q2);
     STOREH(x + 3 * ovs, q1);
     STOREL(x + 2 * ovs, q1);
     STOREH(x + ovs, q0);
     STOREL(x, q0);
}

#else /* !FFTW_SINGLE */

static inline __m256d LD2u(const R *x, INT ivs)
{
     return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(x)),
				 _mm_loadu_pd(x + ivs), 1);
}

static inline V LDu(const R *x, INT ivs, const R *aligned_like)
{
     (void)aligned_like; /* UNUSED */
     return _mm512_insertf64x4(_mm512_castpd256_pd512(LD2u(x, ivs)),
			       LD2u(x + 2 * ivs, ivs), 1);
}

static inline void STu(R *x, V v, INT ovs, const R *aligned_like)
{
     (void)aligned_like; /* UNUSED */
     /* vextractf32x4 to memory is a plain store, no shuffle uop */
     __m512 w = _mm512_castpd_ps(v);
     _mm_storeu_ps((float *)(x + 3 * ovs), _mm512_extractf32x4_ps(w, 3));
     _mm_storeu_ps((float *)(x + 2 * ovs), _mm512_extractf32x4_ps(w, 2));
     _mm_storeu_ps((float *)(x + ovs), _mm512_extractf32x4_ps(w, 1));
     _mm_storeu_pd(x, _mm512_castpd512_pd128(v));
}

#endif /* FFTW_SINGLE */

#define LD LDu
#define ST STu


#ifdef FFTW_SINGLE

/* n2fv/n2bv: v0 holds outputs x[0..1], v1 holds x[2..3] (complex),
   element i at x + i*ovs.  Interleave so each i is one 128-bit store. */
#define STM2(x, v, ovs, a) /* no-op */
static inline void STN2(R *x, V v0, V v1, INT ovs)
{
     V x0 = _mm512_shuffle_ps(v0, v1, _MM_SHUFFLE(1, 0, 1, 0));
     V x1 = _mm512_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 2, 3, 2));
     /* x0 block j -> element 2j, x1 block j -> element 2j+1 */
     _mm_storeu_ps(x + 7 * ovs, _mm512_extractf32x4_ps(x1, 3));
     _mm_storeu_ps(x + 6 * ovs, _mm512_extractf32x4_ps(x0, 3));
     _mm_storeu_ps(x + 5 * ovs, _mm512_extractf32x4_ps(x1, 2));
     _mm_storeu_ps(x + 4 * ovs, _mm512_extractf32x4_ps(x0, 2));
     _mm_storeu_ps(x + 3 * ovs, _mm512_extractf32x4_ps(x1, 1));
     _mm_storeu_ps(x + 2 * ovs, _mm512_extractf32x4_ps(x0, 1));
     _mm_storeu_ps(x + 1 * ovs, _mm512_castps512_ps128(x1));
     _mm_storeu_ps(x, _mm512_castps512_ps128(x0));
}

/* n2sv: 4 real vectors (x[0], x[1], x[2], x[3]), 16 elements each at
   stride ovs.  4x4 transpose inside each 128-bit block, then 16
   128-bit stores instead of 4 x 16-element scatters. */
#define STM4(x, v, ovs, a) /* no-op */
#define STN4(x, v0, v1, v2, v3, ovs)					\
{									\
     V xxx0, xxx1, xxx2, xxx3;						\
     V yyy0, yyy1, yyy2, yyy3;						\
     xxx0 = _mm512_unpacklo_ps(v0, v2);					\
     xxx1 = _mm512_unpackhi_ps(v0, v2);					\
     xxx2 = _mm512_unpacklo_ps(v1, v3);					\
     xxx3 = _mm512_unpackhi_ps(v1, v3);					\
     yyy0 = _mm512_unpacklo_ps(xxx0, xxx2);				\
     yyy1 = _mm512_unpackhi_ps(xxx0, xxx2);				\
     yyy2 = _mm512_unpacklo_ps(xxx1, xxx3);				\
     yyy3 = _mm512_unpackhi_ps(xxx1, xxx3);				\
     _mm_storeu_ps(x + 15 * ovs, _mm512_extractf32x4_ps(yyy3, 3));	\
     _mm_storeu_ps(x + 14 * ovs, _mm512_extractf32x4_ps(yyy2, 3));	\
     _mm_storeu_ps(x + 13 * ovs, _mm512_extractf32x4_ps(yyy1, 3));	\
     _mm_storeu_ps(x + 12 * ovs, _mm512_extractf32x4_ps(yyy0, 3));	\
     _mm_storeu_ps(x + 11 * ovs, _mm512_extractf32x4_ps(yyy3, 2));	\
     _mm_storeu_ps(x + 10 * ovs, _mm512_extractf32x4_ps(yyy2, 2));	\
     _mm_storeu_ps(x +  9 * ovs, _mm512_extractf32x4_ps(yyy1, 2));	\
     _mm_storeu_ps(x +  8 * ovs, _mm512_extractf32x4_ps(yyy0, 2));	\
     _mm_storeu_ps(x +  7 * ovs, _mm512_extractf32x4_ps(yyy3, 1));	\
     _mm_storeu_ps(x +  6 * ovs, _mm512_extractf32x4_ps(yyy2, 1));	\
     _mm_storeu_ps(x +  5 * ovs, _mm512_extractf32x4_ps(yyy1, 1));	\
     _mm_storeu_ps(x +  4 * ovs, _mm512_extractf32x4_ps(yyy0, 1));	\
     _mm_storeu_ps(x +  3 * ovs, _mm512_castps512_ps128(yyy3));		\
     _mm_storeu_ps(x +  2 * ovs, _mm512_castps512_ps128(yyy2));		\
     _mm_storeu_ps(x +  1 * ovs, _mm512_castps512_ps128(yyy1));		\
     _mm_storeu_ps(x,             _mm512_castps512_ps128(yyy0));	\
}

#else /* !FFTW_SINGLE */

/* n2fv/n2bv: v0 holds complex x[0..1], v1 holds x[2..3]; for each
   element i write (v0[i], v1[i]) as one 256-bit store. */
#define STM2(x, v, ovs, a) /* no-op */
static inline void STN2(R *x, V v0, V v1, INT ovs)
{
     const __m512i i01 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
     const __m512i i23 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
     V a = _mm512_permutex2var_pd(v0, i01, v1); /* elements 0,1 */
     V b = _mm512_permutex2var_pd(v0, i23, v1); /* elements 2,3 */
     _mm256_storeu_pd(x + 3 * ovs, _mm512_extractf64x4_pd(b, 1));
     _mm256_storeu_pd(x + 2 * ovs, _mm512_castpd512_pd256(b));
     _mm256_storeu_pd(x + 1 * ovs, _mm512_extractf64x4_pd(a, 1));
     _mm256_storeu_pd(x, _mm512_castpd512_pd256(a));
}

/* n2sv: 4 real vectors (x[0], x[1], x[2], x[3]), 8 elements each at
   stride ovs -> 8 256-bit stores instead of 4 x 8-element scatters. */
#define STM4(x, v, ovs, a) /* no-op */
#define STN4(x, v0, v1, v2, v3, ovs)					\
{									\
     const __m512i iii0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);	\
     const __m512i iii1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15); \
     V xxx0, xxx1, xxx2, xxx3;						\
     V yyy0, yyy1, yyy2, yyy3;						\
     xxx0 = _mm512_unpacklo_pd(v0, v1); /* (v0,v1)[2j]   */		\
     xxx1 = _mm512_unpackhi_pd(v0, v1); /* (v0,v1)[2j+1] */		\
     xxx2 = _mm512_unpacklo_pd(v2, v3);					\
     xxx3 = _mm512_unpackhi_pd(v2, v3);					\
     yyy0 = _mm512_permutex2var_pd(xxx0, iii0, xxx2); /* elts 0,2 */	\
     yyy1 = _mm512_permutex2var_pd(xxx1, iii0, xxx3); /* elts 1,3 */	\
     yyy2 = _mm512_permutex2var_pd(xxx0, iii1, xxx2); /* elts 4,6 */	\
     yyy3 = _mm512_permutex2var_pd(xxx1, iii1, xxx3); /* elts 5,7 */	\
     _mm256_storeu_pd(x + 7 * ovs, _mm512_extractf64x4_pd(yyy3, 1));	\
     _mm256_storeu_pd(x + 6 * ovs, _mm512_extractf64x4_pd(yyy2, 1));	\
     _mm256_storeu_pd(x + 5 * ovs, _mm512_castpd512_pd256(yyy3));	\
     _mm256_storeu_pd(x + 4 * ovs, _mm512_castpd512_pd256(yyy2));	\
     _mm256_storeu_pd(x + 3 * ovs, _mm512_extractf64x4_pd(yyy1, 1));	\
     _mm256_storeu_pd(x + 2 * ovs, _mm512_extractf64x4_pd(yyy0, 1));	\
     _mm256_storeu_pd(x + 1 * ovs, _mm512_castpd512_pd256(yyy1));	\
     _mm256_storeu_pd(x,           _mm512_castpd512_pd256(yyy0));	\
}

#endif /* FFTW_SINGLE */


static inline V VZMUL(V tx, V sr)
{
     /* V tr = VDUPL(tx); */
     /* V ti = VDUPH(tx); */
     /* tr = VMUL(sr, tr); */
     /* sr = VBYI(sr); */
     /* return VFMA(ti, sr, tr); */
     return SUFF(_mm512_fmaddsub)(sr, VDUPL(tx), VMUL(FLIP_RI(sr), VDUPH(tx)));
}

static inline V VZMULJ(V tx, V sr)
{
     /* V tr = VDUPL(tx); */
     /* V ti = VDUPH(tx); */
     /* tr = VMUL(sr, tr); */
     /* sr = VBYI(sr); */
     /* return VFNMS(ti, sr, tr); */
     return SUFF(_mm512_fmsubadd)(sr, VDUPL(tx), VMUL(FLIP_RI(sr), VDUPH(tx)));
}

static inline V VZMULI(V tx, V sr)
{
     V tr = VDUPL(tx);
     V ti = VDUPH(tx);
     ti = VMUL(ti, sr);
     sr = VBYI(sr);
     return VFMS(tr, sr, ti);
  /* return SUFF(_mm512_addsub)(SUFF(_mm512_fnmadd)(sr, VDUPH(tx), VZERO), VMUL(FLIP_RI(sr), VDUPL(tx))); */
}

static inline V VZMULIJ(V tx, V sr)
{
     /* V tr = VDUPL(tx); */
     /* V ti = VDUPH(tx); */
     /* ti = VMUL(ti, sr); */
     /* sr = VBYI(sr); */
     /* return VFMA(tr, sr, ti); */
     return SUFF(_mm512_fmaddsub)(sr, VDUPH(tx), VMUL(FLIP_RI(sr), VDUPL(tx)));
}

/* twiddle storage #1: compact, slower */
#define DEFVTW1(v, x) {TW_CEXP, v, x}
#define VTW1(v,x) CONCAT2(REPEAT_, VL)(DEFVTW1, v, x)
#define TWVL1 (VL)

static inline V BYTW1(const R *t, V sr)
{
     return VZMUL(LDA(t, 2, t), sr);
}

static inline V BYTWJ1(const R *t, V sr)
{
     return VZMULJ(LDA(t, 2, t), sr);
}

/* twiddle storage #2: twice the space, faster (when in cache) */
#define DEFVTW2_COS(v, x) {TW_COS, v, x}, {TW_COS, v, x}
#define DEFVTW2_SIN(v, x) {TW_SIN, v, -x}, {TW_SIN, v, x}
#define VTW2(v,x) CONCAT2(REPEAT_, VL)(DEFVTW2_COS, v, x),      \
          CONCAT2(REPEAT_, VL)(DEFVTW2_SIN, v, x)
#define TWVL2 (2*VL)

static inline V BYTW2(const R *t, V sr)
{
     const V *twp = (const V *)t;
     V si = FLIP_RI(sr);
     V tr = twp[0], ti = twp[1];
/*      V tr = LD(t, 2, t), ti = LD(t + VL, 2, t + VL); */
     return VFMA(tr, sr, VMUL(ti, si));
}

static inline V BYTWJ2(const R *t, V sr)
{
     const V *twp = (const V *)t;
     V si = FLIP_RI(sr);
     V tr = twp[0], ti = twp[1];
/*      V tr = LD(t, 2, t), ti = LD(t + VL, 2, t + VL); */
     return VFNMS(ti, si, VMUL(tr, sr));
}

/* twiddle storage #3 */
#define VTW3(v,x) VTW1(v,x)
#define TWVL3 TWVL1

/* twiddle storage for split arrays */
#define DEFVTWS_COS(v, x) {TW_COS, 2*(v), x}, {TW_COS, 2*(v)+1, x}
#define DEFVTWS_SIN(v, x) {TW_SIN, 2*(v), x}, {TW_SIN, 2*(v)+1, x}
#define VTWS(v,x) CONCAT2(REPEAT_, VL)(DEFVTWS_COS, v, x),      \
          CONCAT2(REPEAT_, VL)(DEFVTWS_SIN, v, x)
#define TWVLS (2*VL)

#define VLEAVE _mm256_zeroupper

#include "simd-common.h"
