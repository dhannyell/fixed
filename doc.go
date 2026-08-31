// Package fixed implements deterministic Q32.32 fixed-point arithmetic.
//
// [Q] stores one signed value in an int64. The high 32 bits contain the signed
// integer part. The low 32 bits contain the fraction. The resolution is 2⁻³².
// The range is [-2³¹, 2³¹ - 2⁻³²]. Each operation produces the same bits on
// every supported architecture.
//
// # Overflow
//
// An overflow saturates to [MinValue] or [MaxValue]. [SaturationCount] reports
// saturation events for diagnostics. The counter does not affect Q values or
// operation results.
//
// [Q.Div] panics for a zero divisor. [FromRatio] panics for a zero denominator.
// [Q.Sqrt] panics for a negative input.
//
// Saturated addition is not associative near the range limits. Do not reorder
// an accumulation. Use enough numeric headroom to prevent saturation.
//
// # Rounding
//
// [Q.Mul] floors the 128-bit product when it converts the result to Q32.32.
// This operation matches an arithmetic right shift. [Q.Div] and [FromRatio]
// truncate toward zero. [Q.Sqrt] floors its result.
//
// [Q.Round] and [MustParse] round to the nearest representable value. An exact
// half rounds away from zero.
//
// # Construction and text
//
// [Q] is opaque. Construct values with [FromInt], [FromRatio], [MustParse], or
// [FromRaw]. The package does not accept float values because a computed float
// can contain architecture-dependent bits.
//
// [Q.String] returns the exact canonical decimal form. For every q,
// MustParse(q.String()) == q.
//
// # Angles
//
// Angles are turns: [One] is a full revolution. The fractional bits of a Q map
// directly onto the circle, so range reduction is a bit mask and never rounds.
// Pi stays out of the reduction entirely.
//
// [SinTurns] and [CosTurns] are periodic by contract. Every Q is a valid
// input. They never panic and never saturate. Results stay in [-One, One].
// The kernel is a quarter-wave table with linear interpolation. The maximum
// absolute error is 2⁻²⁰.
//
// [Atan2Turns] recovers an angle from a vector. The result stays in
// (-1/2, 1/2] turns. [Rot] stores a rotation as its sine and cosine, so
// composition and application need no trig evaluation.
//
// # Compatibility contract
//
// The raw representation, saturation rules, and rounding rules are part of the
// public contract. An independent implementation must reproduce these rules
// before it exchanges raw values with this package. The table literals in
// trig_table.go are part of this contract: an independent kernel must embed
// the same table.
//
// # Dependencies
//
// Production files import only math/bits and sync/atomic.
package fixed
