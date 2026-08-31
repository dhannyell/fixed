// Package fixed implements deterministic Q32.32 fixed-point arithmetic.
//
// [Q32] stores one signed value in an int64. The high 32 bits contain the signed
// integer part. The low 32 bits contain the fraction. The resolution is 2⁻³².
// The range is [-2³¹, 2³¹ - 2⁻³²]. Each operation produces the same bits on
// every supported architecture.
//
// # Overflow
//
// An overflow saturates to [MinValue] or [MaxValue]. [SaturationCount] reports
// saturation events for diagnostics. The counter does not affect Q32 values or
// operation results.
//
// [Q32.Div] panics for a zero divisor. [FromRatio] panics for a zero denominator.
// [Q32.Sqrt] panics for a negative input.
//
// Saturated addition is not associative near the range limits. Do not reorder
// an accumulation. Use enough numeric headroom to prevent saturation.
//
// # Rounding
//
// [Q32.Mul] floors the 128-bit product when it converts the result to Q32.32.
// This operation matches an arithmetic right shift. [Q32.Div] and [FromRatio]
// truncate toward zero. [Q32.Sqrt] floors its result.
//
// [Q32.Round] and [MustParse] round to the nearest representable value. An exact
// half rounds away from zero.
//
// # Construction and text
//
// [Q32] is opaque. Construct values with [FromInt], [FromRatio], [MustParse], or
// [FromRaw]. The package does not accept float values because a computed float
// can contain architecture-dependent bits.
//
// [Q32.String] returns the exact canonical decimal form. For every q,
// MustParse(q.String()) == q.
//
// # Angles
//
// Angles use turns. [One] is a full revolution. The fractional bits of a Q32 map
// directly to the circle, so range reduction does not use pi and does not
// round.
//
// [SinTurns] and [CosTurns] accept every Q32 value. They never panic or saturate,
// and their results stay in [-One, One]. They use a 1024-interval quarter-wave
// table. Table entries round to nearest, and linear interpolation floors. The
// maximum absolute error is 2⁻²⁰.
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
// # Compatibility contract
//
// The raw representation, saturation rules, and rounding rules are part of the
// public contract. An independent implementation must reproduce these rules
// before it exchanges raw values with this package. The trigonometric raw
// outputs are also part of the contract. A compatible implementation may use a
// different representation, but it must produce the same output bits.
//
// # Dependencies
//
// The fixed package imports only math, math/bits, and sync/atomic. The math
// import provides hardware seeds; exact integer comparisons close every
// result, so floating point never decides a bit.
package fixed
