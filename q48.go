package fixed

import "math/bits"

// Q48 stores an opaque signed Q48.16 value. Its zero value is 0.
// Q48 shares the fraction grid of Q16, so it accumulates Q16 products with
// 16 bits of integer headroom.
type Q48 struct {
	raw int64
}

const (
	q48RawHalf = 1 << 15
	q48RawOne  = 1 << 16
	q48RawMin  = -1 << 63
	q48RawMax  = 1<<63 - 1
)

// Q48FromInt returns i as a Q48 value. It saturates outside [-2⁴⁷, 2⁴⁷-1].
// The integer range exceeds a 32-bit int, so the argument is an int64.
func Q48FromInt(i int64) Q48 {
	if i > 1<<47-1 {
		saturationEvents.Add(1)
		return Q48{raw: q48RawMax}
	}
	if i < -1<<47 {
		saturationEvents.Add(1)
		return Q48{raw: q48RawMin}
	}
	return Q48{raw: i << 16}
}

// Q48FromRatio returns num/den truncated toward zero. It saturates on
// overflow. It panics when den is zero.
func Q48FromRatio(num, den int64) Q48 {
	if den == 0 {
		panicDivZero()
	}
	return q48DivMag(magnitude(num), magnitude(den), (num < 0) != (den < 0))
}

// Q48FromRaw returns the Q48 value with the specified signed bit pattern.
func Q48FromRaw(raw int64) Q48 { return Q48{raw: raw} }

// Q48Zero returns the fixed-point value 0.
func Q48Zero() Q48 { return Q48{} }

// Q48One returns the fixed-point value 1.
func Q48One() Q48 { return Q48{q48RawOne} }

// Q48Half returns the fixed-point value 1/2.
func Q48Half() Q48 { return Q48{q48RawHalf} }

// Q48MinValue returns the smallest representable Q48 value, -2⁴⁷.
func Q48MinValue() Q48 { return Q48{q48RawMin} }

// Q48MaxValue returns the largest representable Q48 value, 2⁴⁷ - 2⁻¹⁶.
func Q48MaxValue() Q48 { return Q48{q48RawMax} }

// Raw returns the signed Q48.16 bit pattern of q.
func (q Q48) Raw() int64 { return q.raw }

// Add returns q+o. It saturates on overflow.
func (q Q48) Add(o Q48) Q48 {
	res := q.raw + o.raw
	if (q.raw >= 0) == (o.raw >= 0) && (res >= 0) != (q.raw >= 0) {
		saturationEvents.Add(1)
		return Q48{raw: q48RawMax ^ (q.raw >> 63)}
	}
	return Q48{raw: res}
}

// Sub returns q-o. It saturates on overflow.
func (q Q48) Sub(o Q48) Q48 {
	res := q.raw - o.raw
	if (q.raw >= 0) != (o.raw >= 0) && (res >= 0) != (q.raw >= 0) {
		saturationEvents.Add(1)
		return Q48{raw: q48RawMax ^ (q.raw >> 63)}
	}
	return Q48{raw: res}
}

// Mul returns q*o. It floors the product to Q48.16 and saturates on overflow.
func (q Q48) Mul(o Q48) Q48 {
	hi, lo := bits.Mul64(uint64(q.raw), uint64(o.raw))

	if q.raw < 0 {
		hi -= uint64(o.raw)
	}
	if o.raw < 0 {
		hi -= uint64(q.raw)
	}

	res := int64(hi<<48 | lo>>16)
	if int64(hi)>>16 != res>>63 {
		saturationEvents.Add(1)
		// A negative product flips every bit of Max to give Min.
		return Q48{raw: q48RawMax ^ (int64(hi) >> 63)}
	}
	return Q48{raw: res}
}

// MulAdd16 returns q + a*b. The Q16 product is exact in 64 bits. The sum
// floors it to the Q48.16 grid and saturates on overflow.
func (q Q48) MulAdd16(a, b Q16) Q48 {
	return q.Add(Q48{raw: (int64(a.raw) * int64(b.raw)) >> 16})
}

// Mul16 returns q*f for a Q16 factor. It floors the product to Q48.16 and
// saturates on overflow; the bits equal q.Mul(f.ToQ48()). The sign
// corrections are branch-free so the method stays inside the inlining budget.
func (q Q48) Mul16(f Q16) Q48 {
	o := int64(f.raw)
	hi, lo := bits.Mul64(uint64(q.raw), uint64(o))
	hi -= uint64(o) & uint64(q.raw>>63)
	hi -= uint64(q.raw) & uint64(o>>63)
	// The result fits when bits 63..79 of the product agree, and they are
	// bits 15..31 of hi.
	if uint64(int64(hi)>>15+1) > 1 {
		saturationEvents.Add(1)
		return Q48{raw: q48RawMax ^ (int64(hi) >> 63)}
	}
	return Q48{raw: int64(hi<<48 | lo>>16)}
}

// Div returns q/o truncated toward zero. It saturates on overflow.
// It panics when o is zero.
func (q Q48) Div(o Q48) Q48 {
	if o.raw == 0 {
		panicDivZero()
	}
	return q48DivMag(magnitude(q.raw), magnitude(o.raw), (q.raw < 0) != (o.raw < 0))
}

// Sqrt returns floor(sqrt(q)). It panics when q is negative.
// The result cannot overflow.
func (q Q48) Sqrt() Q48 {
	if q.raw < 0 {
		panic("fixed: square root of a negative value")
	}
	// The 80-bit radicand has a root of at most 40 bits.
	return Q48{raw: isqrtQ48(q.raw)}
}

// Neg returns -q. Neg of MinValue saturates to MaxValue.
func (q Q48) Neg() Q48 {
	if q.raw == q48RawMin {
		saturationEvents.Add(1)
		return Q48{raw: q48RawMax}
	}
	return Q48{raw: -q.raw}
}

// Abs returns the magnitude of q. Abs of MinValue saturates to MaxValue.
func (q Q48) Abs() Q48 {
	if q.raw >= 0 {
		return q
	}
	return q.Neg()
}

// Cmp returns -1 when q < o, 0 when q == o, and 1 when q > o.
func (q Q48) Cmp(o Q48) int {
	if q.raw < o.raw {
		return -1
	}
	if q.raw > o.raw {
		return 1
	}
	return 0
}

// Less reports whether q < o.
func (q Q48) Less(o Q48) bool { return q.raw < o.raw }

// Greater reports whether q > o.
func (q Q48) Greater(o Q48) bool { return q.raw > o.raw }

// Eq reports whether q == o.
func (q Q48) Eq(o Q48) bool { return q.raw == o.raw }

// Min returns the smaller of q and o.
func (q Q48) Min(o Q48) Q48 {
	if o.raw < q.raw {
		return o
	}
	return q
}

// Max returns the larger of q and o.
func (q Q48) Max(o Q48) Q48 {
	if o.raw > q.raw {
		return o
	}
	return q
}

// Clamp returns q limited to [lo, hi]. It requires lo <= hi.
func (q Q48) Clamp(lo, hi Q48) Q48 {
	if q.raw < lo.raw {
		return lo
	}
	if q.raw > hi.raw {
		return hi
	}
	return q
}

// Floor returns the largest integer multiple of Q48One not above q.
func (q Q48) Floor() Q48 { return Q48{raw: q.raw &^ (q48RawOne - 1)} }

// Ceil returns the smallest integer multiple of Q48One not below q.
// It saturates when the result is outside the Q48 range.
func (q Q48) Ceil() Q48 {
	f := q.raw &^ (q48RawOne - 1)
	if f == q.raw {
		return q
	}
	// The next integer is outside the Q48 range.
	if f == (1<<47-1)<<16 {
		saturationEvents.Add(1)
		return Q48{raw: q48RawMax}
	}
	return Q48{raw: f + q48RawOne}
}

// Round returns the nearest integer multiple of Q48One. An exact half rounds
// away from zero. Round saturates when the result is outside the Q48 range.
func (q Q48) Round() Q48 {
	if q.raw >= 0 {
		// The nearest integer is outside the Q48 range.
		if q.raw > q48RawMax-q48RawHalf {
			saturationEvents.Add(1)
			return Q48{raw: q48RawMax}
		}
		return Q48{raw: (q.raw + q48RawHalf) &^ (q48RawOne - 1)}
	}
	// The magnitude cannot exceed 2⁶³, so the result cannot pass MinValue.
	rounded := (magnitude(q.raw) + q48RawHalf) &^ uint64(q48RawOne-1)
	return Q48{raw: -int64(rounded)} // A magnitude of 1<<63 converts to MinValue.
}

// Int returns the integer part truncated toward zero. The integer part does
// not fit a 32-bit int, so the result is an int64 on every architecture.
func (q Q48) Int() int64 {
	if q.raw >= 0 {
		return q.raw >> 16
	}
	return -int64(magnitude(q.raw) >> 16)
}

// ToQ16 saturates q to the Q16 range. Both types share one fraction grid,
// so no rounding occurs.
func (q Q48) ToQ16() Q16 { return q16Saturate(q.raw) }

// ToQ32 returns q as a Q32.32 value. The fraction grid gets finer, so no
// rounding occurs. The integer range shrinks, so it saturates outside the
// Q32 range.
func (q Q48) ToQ32() Q32 {
	// The shift is exact only when the integer part fits 31 bits plus sign.
	if top := q.raw >> 47; top != 0 && top != -1 {
		saturationEvents.Add(1)
		return Q32{raw: q32RawMax ^ (q.raw >> 63)}
	}
	return Q32{raw: q.raw << 16}
}

// q48DivMag returns the signed Q48.16 quotient of magnitudes n and d. It
// truncates toward zero and saturates. The caller must reject d == 0.
func q48DivMag(n, d uint64, neg bool) Q48 {
	hi, lo := n>>48, n<<16
	if hi >= d {
		saturationEvents.Add(1)
		if neg {
			return Q48{raw: q48RawMin}
		}
		return Q48{raw: q48RawMax}
	}
	quo, _ := bits.Div64(hi, lo, d)
	if neg {
		if quo > 1<<63 {
			saturationEvents.Add(1)
			return Q48{raw: q48RawMin}
		}
		return Q48{raw: -int64(quo)} // A quotient of 1<<63 converts to MinValue.
	}
	if quo > q48RawMax {
		saturationEvents.Add(1)
		return Q48{raw: q48RawMax}
	}
	return Q48{raw: int64(quo)}
}
