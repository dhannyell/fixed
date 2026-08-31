// Package fixed implements deterministic signed fixed-point arithmetic.
//
// The package provides two formats without choosing a default. [Q32] stores
// Q32.32 in an int64, with resolution 2⁻³² and range
// [-2³¹, 2³¹ - 2⁻³²]. [Q16] stores Q16.16 in an int32, with resolution 2⁻¹⁶
// and range [-2¹⁵, 2¹⁵ - 2⁻¹⁶]. Each operation produces the same bits on every
// supported architecture.
//
// # Overflow
//
// An overflow saturates to the minimum or maximum of the selected format.
// [SaturationCount] reports saturation events for diagnostics. The counter does
// not affect fixed-point values or operation results.
//
// [Q32.Div] and [Q16.Div] panic for a zero divisor. [Q32FromRatio] and
// [Q16FromRatio] panic for a zero denominator. [Q32.Sqrt] and [Q16.Sqrt] panic
// for a negative input.
//
// Saturated addition is not associative near the range limits. Do not reorder
// an accumulation. Use enough numeric headroom to prevent saturation.
//
// # Rounding
//
// [Q32.Mul] floors the 128-bit product when it converts the result to Q32.32.
// [Q16.Mul] floors the exact 62-bit product when it converts the result to
// Q16.16. Both operations match an arithmetic right shift.
//
// [Q32.Div], [Q16.Div], [Q32FromRatio], and [Q16FromRatio] truncate toward
// zero. [Q32.Sqrt] and [Q16.Sqrt] floor their results. [Q32.Round], [Q16.Round],
// [Q32MustParse], and [Q16MustParse] round to the nearest representable value.
// An exact half rounds away from zero.
//
// # Formats and conversion
//
// [Q16.ToQ32] widens exactly and never saturates. [Q32.ToQ16] floors to the
// Q16.16 grid and saturates outside the Q16 range. For any Q16 values a and b,
// a.Mul(b) equals a.ToQ32().Mul(b.ToQ32()).ToQ16(). The same identity does not
// hold for Div because division truncates toward zero and narrowing floors.
//
// # Construction and text
//
// [Q32] and [Q16] are opaque. Construct Q32 values with [Q32FromInt],
// [Q32FromRatio], [Q32MustParse], or [Q32FromRaw]. Construct Q16 values with
// [Q16FromInt], [Q16FromRatio], [Q16MustParse], or [Q16FromRaw]. The package
// does not accept float values because a computed float can contain
// architecture-dependent bits.
//
// [Q32.String] and [Q16.String] return exact canonical decimal forms. For every
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
// # Compatibility contract
//
// The two raw representations, their conversions, saturation rules, and
// rounding rules are part of the public contract. An independent implementation
// must reproduce these rules before it exchanges raw values with this package.
// The trigonometric raw outputs are also part of the contract. A compatible
// implementation may use a different representation, but it must produce the
// same output bits.
//
// # Dependencies
//
// The fixed package imports only math, math/bits, and sync/atomic. The math
// import provides hardware seeds; exact integer comparisons close every
// result, so floating point never decides a bit.
package fixed
