// Package fixed implements deterministic signed fixed-point arithmetic.
//
// The package provides three formats without choosing a default. [Q32] stores
// Q32.32 in an int64, with resolution 2⁻³² and range
// [-2³¹, 2³¹ - 2⁻³²]. [Q16] stores Q16.16 in an int32, with resolution 2⁻¹⁶
// and range [-2¹⁵, 2¹⁵ - 2⁻¹⁶]. [Q48] stores Q48.16 in an int64, with
// resolution 2⁻¹⁶ and range [-2⁴⁷, 2⁴⁷ - 2⁻¹⁶]; it is the accumulator for Q16
// products. Each operation produces the same bits on every supported
// architecture.
//
// # Overflow
//
// An overflow saturates to the minimum or maximum of the selected format.
// [SaturationCount] reports saturation events for diagnostics. The counter does
// not affect fixed-point values or operation results.
//
// Div panics for a zero divisor in every format. [Q32FromRatio],
// [Q16FromRatio], and [Q48FromRatio] panic for a zero denominator. Sqrt panics
// for a negative input in every format.
//
// Saturated addition is not associative near the range limits. Do not reorder
// an accumulation. Use enough numeric headroom to prevent saturation.
//
// # Rounding
//
// [Q32.Mul] and [Q48.Mul] floor the 128-bit product when they convert the
// result to their format. [Q16.Mul] floors the exact 62-bit product when it
// converts the result to Q16.16. All three operations match an arithmetic
// right shift. [Q48.MulAdd16] floors the exact Q16 product to Q48.16 and then
// adds it.
//
// Div and FromRatio truncate toward zero in every format. Sqrt floors its
// result in every format. Round and MustParse round to the nearest
// representable value in every format. An exact half rounds away from zero.
//
// # Formats and conversion
//
// A widening conversion is exact and never saturates. A narrowing conversion
// floors to the target grid and saturates outside the target range.
// [Q16.ToQ32], [Q16.ToQ48], and [Q48.ToQ32] widen. [Q32.ToQ16], [Q32.ToQ48],
// and [Q48.ToQ16] narrow. Q16 and Q48 share one fraction grid, so [Q48.ToQ16]
// only saturates and [Q32.ToQ48] only floors. [Q48.ToQ32] saturates when the
// integer part does not fit 31 bits.
//
// For any Q16 values a and b, a.Mul(b) equals a.ToQ32().Mul(b.ToQ32()).ToQ16()
// and also equals Q48Zero().MulAdd16(a, b).ToQ16(). The same identity does not
// hold for Div because division truncates toward zero and narrowing floors.
//
// # Accumulation
//
// A Q16 product has at most 32 integer bits. [Q48.MulAdd16] stores it with 16
// bits of headroom, so a sum of up to 2¹⁶ products of full-range Q16 values
// cannot saturate. Narrow the sum with [Q48.ToQ16] only when the value is
// stored, and check [SaturationCount] when the term count is not bounded.
//
// # Construction and text
//
// [Q32], [Q16], and [Q48] are opaque. Construct Q32 values with [Q32FromInt],
// [Q32FromRatio], [Q32MustParse], or [Q32FromRaw]. Q16 and Q48 have the same
// four constructors under their own prefixes. The package does not accept float
// values because a computed float can contain architecture-dependent bits.
//
// String returns the exact canonical decimal form in every format. For every
// value q, parsing q.String() with the constructor for its format returns q.
//
// # Angles
//
// Angles use turns. [Q32One] is a full revolution. The fractional bits of a Q32
// map directly to the circle, so range reduction does not use pi and does not
// round.
//
// [SinTurns] and [CosTurns] accept every Q32 value. They never panic or saturate,
// and their results stay in [-Q32One, Q32One]. They use a 1024-interval
// quarter-wave table. Table entries round to nearest, and linear interpolation
// floors. The maximum absolute error is 2⁻²⁰.
//
// [Atan2Turns] returns an angle in (-1/2, 1/2] turns. It reduces the input to a
// ratio in [0, 1], truncates that ratio to Q32.32, and uses a 1024-interval
// table. Table entries round to nearest, and linear interpolation floors.
// Octant reconstruction is exact.
//
// # Vectors and rotations
//
// [Vec2.LenSq] composes scalar multiplication and addition, so it can saturate
// even when the length fits in Q32.32. [Vec2.Len] computes the length with a
// 128-bit intermediate and saturates only when the final length is out of
// range. [Vec2.Normalize] scales the components before it squares them, so
// intermediate underflow cannot turn a nonzero vector into the zero vector.
//
// [Rot] stores a rotation as its sine and cosine. The zero Rot is invalid. Use
// [RotIdentity] or [RotFromTurns] to construct a rotation. Repeated composition
// can introduce rounding drift; [Rot.Normalize] restores unit length.
//
// # Batch operations
//
// [BatchAdd16], [BatchSub16], [BatchMul16], and [BatchClamp16] apply one
// operation across whole slices of Q16. Their results and saturation counts
// are identical to a loop over the scalar methods; the counter receives one
// update per call with the number of saturated elements. The destination may
// be the same slice as a source; any other overlap is undefined.
// [BatchQ32FromQ16] and [BatchQ16FromQ32] move whole slices across the format
// boundary and follow the conversion rules of [Q16.ToQ32] and [Q32.ToQ16].
//
// Every build runs the scalar kernels. A build with GOEXPERIMENT=simd on Go
// 1.27 or later selects vector kernels at package initialization on amd64 with
// AVX2 and on arm64. [BatchPath] reports the active family. The selection
// changes speed, never bits.
//
// # Compatibility contract
//
// The three raw representations, their conversions, saturation rules, and
// rounding rules are part of the public contract. An independent implementation
// must reproduce these rules before it exchanges raw values with this package.
// The trigonometric raw outputs are also part of the contract. A compatible
// implementation may use a different representation, but it must produce the
// same output bits.
//
// # Dependencies
//
// The portable files import only math, math/bits, and sync/atomic. The math
// import provides hardware seeds; exact integer comparisons close every
// result, so floating point never decides a bit. Files behind the
// goexperiment.simd build tag also import unsafe and simd/archsimd; no default
// build reaches them. The unsafe import is confined to batch16_raw.go, which
// reinterprets a Q16 or Q32 slice as the raw words the vector loads take and
// asserts the layout at compile time.
package fixed
