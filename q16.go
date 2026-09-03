package fixed

import "math/bits"

// Q16 stores an opaque signed Q16.16 value. Its zero value is 0.
type Q16 struct {
	raw int32
}

const (
	q16RawHalf = 1 << 15
	q16RawOne  = 1 << 16
	q16RawMin  = -1 << 31
	q16RawMax  = 1<<31 - 1
)

// Q16FromInt returns i as a Q16 value. It saturates outside [-2¹⁵, 2¹⁵-1].
func Q16FromInt(i int) Q16 {
	if i > 1<<15-1 {
		saturationEvents.Add(1)
		return Q16{raw: q16RawMax}
	}
	if i < -1<<15 {
		saturationEvents.Add(1)
		return Q16{raw: q16RawMin}
	}
	return Q16{raw: int32(i) << 16}
}

// Q16FromRatio returns num/den truncated toward zero.
// It saturates on overflow. It panics when den is zero.
func Q16FromRatio(num, den int) Q16 {
	if den == 0 {
		panicDivZero()
	}
	n, d := magnitude(int64(num)), magnitude(int64(den))
	neg := (num < 0) != (den < 0)
	hi, lo := n>>48, n<<16
	if hi >= d {
		return q16SaturatedQuotient(neg)
	}
	quo, _ := bits.Div64(hi, lo, d)
	if neg {
		if quo > 1<<31 {
			return q16SaturatedQuotient(true)
		}
		return Q16{raw: int32(-int64(quo))} // A magnitude of 1<<31 converts to the minimum.
	}
	if quo > uint64(q16RawMax) {
		return q16SaturatedQuotient(false)
	}
	return Q16{raw: int32(quo)}
}

// Q16FromRaw returns the Q16 value with the specified signed bit pattern.
func Q16FromRaw(raw int32) Q16 {
	return Q16{raw: raw}
}

// Q16Zero returns the fixed-point value 0.
func Q16Zero() Q16 { return Q16{} }

// Q16One returns the fixed-point value 1.
func Q16One() Q16 { return Q16{q16RawOne} }

// Q16Half returns the fixed-point value 1/2.
func Q16Half() Q16 { return Q16{q16RawHalf} }

// Q16MinValue returns the smallest representable Q16 value, -2¹⁵.
func Q16MinValue() Q16 { return Q16{q16RawMin} }

// Q16MaxValue returns the largest representable Q16 value, 2¹⁵ - 2⁻¹⁶.
func Q16MaxValue() Q16 { return Q16{q16RawMax} }

// Raw returns the signed Q16.16 bit pattern of q.
func (q Q16) Raw() int32 {
	return q.raw
}

// Add returns q+o. It saturates on overflow.
func (q Q16) Add(o Q16) Q16 { return q16Saturate(int64(q.raw) + int64(o.raw)) }

// Sub returns q-o. It saturates on overflow.
func (q Q16) Sub(o Q16) Q16 { return q16Saturate(int64(q.raw) - int64(o.raw)) }

// Mul returns q*o. It floors the product to Q16.16 and saturates on overflow.
func (q Q16) Mul(o Q16) Q16 {
	return q16Saturate((int64(q.raw) * int64(o.raw)) >> 16)
}

// Div returns q/o truncated toward zero. It saturates on overflow.
// It panics when o is zero.
func (q Q16) Div(o Q16) Q16 {
	if o.raw == 0 {
		panicDivZero()
	}
	return q16Saturate((int64(q.raw) << 16) / int64(o.raw))
}

// Sqrt returns floor(sqrt(q)). It panics when q is negative.
// The result cannot overflow.
func (q Q16) Sqrt() Q16 {
	if q.raw < 0 {
		panic("fixed: square root of a negative value")
	}
	// The 47-bit radicand has a root of at most 24 bits.
	return Q16{raw: int32(isqrt128(0, uint64(q.raw)<<16))}
}

// Neg returns -q. Neg of the minimum saturates to the maximum.
func (q Q16) Neg() Q16 { return q16Saturate(-int64(q.raw)) }

// Abs returns the magnitude of q. Abs of the minimum saturates to the maximum.
func (q Q16) Abs() Q16 {
	if q.raw >= 0 {
		return q
	}
	return q.Neg()
}

// Cmp returns -1 when q < o, 0 when q == o, and 1 when q > o.
func (q Q16) Cmp(o Q16) int {
	if q.raw < o.raw {
		return -1
	}
	if q.raw > o.raw {
		return 1
	}
	return 0
}

// Less reports whether q < o.
func (q Q16) Less(o Q16) bool { return q.raw < o.raw }

// Greater reports whether q > o.
func (q Q16) Greater(o Q16) bool { return q.raw > o.raw }

// Eq reports whether q == o.
func (q Q16) Eq(o Q16) bool { return q.raw == o.raw }

// Min returns the smaller of q and o.
func (q Q16) Min(o Q16) Q16 {
	if o.raw < q.raw {
		return o
	}
	return q
}

// Max returns the larger of q and o.
func (q Q16) Max(o Q16) Q16 {
	if o.raw > q.raw {
		return o
	}
	return q
}

// Clamp returns q limited to [lo, hi]. It requires lo <= hi.
func (q Q16) Clamp(lo, hi Q16) Q16 {
	if q.raw < lo.raw {
		return lo
	}
	if q.raw > hi.raw {
		return hi
	}
	return q
}

// Floor returns the largest integer multiple of Q16One not above q.
func (q Q16) Floor() Q16 { return Q16{raw: q.raw &^ (q16RawOne - 1)} }

// Ceil returns the smallest integer multiple of Q16One not below q.
// It saturates when the result is outside the Q16 range.
func (q Q16) Ceil() Q16 {
	f := q.raw &^ int32(q16RawOne-1)
	if f == q.raw {
		return q
	}
	return q16Saturate(int64(f) + q16RawOne)
}

// Round returns the nearest integer multiple of Q16One. An exact half rounds
// away from zero. Round saturates when the result is outside the Q16 range.
func (q Q16) Round() Q16 {
	if q.raw >= 0 {
		return q16Saturate((int64(q.raw) + q16RawHalf) &^ (q16RawOne - 1))
	}
	m := (-int64(q.raw) + q16RawHalf) &^ int64(q16RawOne-1)
	return q16Saturate(-m)
}

// Int returns the integer part truncated toward zero.
func (q Q16) Int() int { return int(int64(q.raw) / q16RawOne) }

// ToQ32 returns q widened to Q32.32. The conversion is exact and never saturates.
func (q Q16) ToQ32() Q32 { return Q32{raw: int64(q.raw) << 16} }

// ToQ48 returns q widened to Q48.16. Both types share one fraction grid, so
// the conversion is a sign extension. It never saturates.
func (q Q16) ToQ48() Q48 { return Q48{raw: int64(q.raw)} }

// q16SaturatedQuotient handles a quotient outside the Q16 range.
func q16SaturatedQuotient(neg bool) Q16 {
	saturationEvents.Add(1)
	if neg {
		return Q16{raw: q16RawMin}
	}
	return Q16{raw: q16RawMax}
}

// q16Saturate clamps a widened raw value to the Q16 range.
func q16Saturate(v int64) Q16 {
	if v > q16RawMax {
		saturationEvents.Add(1)
		return Q16{raw: q16RawMax}
	}
	if v < q16RawMin {
		saturationEvents.Add(1)
		return Q16{raw: q16RawMin}
	}
	return Q16{raw: int32(v)}
}
