package fixed

import (
	"math"
	"math/bits"
)

// Q32 stores an opaque signed Q32.32 value. Its zero value is 0.
type Q32 struct {
	raw int64
}

const (
	q32RawHalf = 1 << 31
	q32RawOne  = 1 << 32
	q32RawMin  = -1 << 63
	q32RawMax  = 1<<63 - 1
)

// Q32FromInt returns i as a Q32 value. It saturates outside [-2³¹, 2³¹-1].
func Q32FromInt(i int) Q32 {
	if i > 1<<31-1 {
		saturationEvents.Add(1)
		return Q32{raw: q32RawMax}
	}
	if i < -1<<31 {
		saturationEvents.Add(1)
		return Q32{raw: q32RawMin}
	}
	return Q32{raw: int64(i) << 32}
}

// Q32FromRatio returns num/den truncated toward zero. It saturates on overflow.
// It panics when den is zero.
func Q32FromRatio(num, den int) Q32 {
	if den == 0 {
		panicDivZero()
	}
	return divMag(magnitude(int64(num)), magnitude(int64(den)), (num < 0) != (den < 0))
}

// Q32FromRaw returns the Q32 value with the specified signed bit pattern.
func Q32FromRaw(raw int64) Q32 { return Q32{raw: raw} }

// Q32Zero returns the fixed-point value 0.
func Q32Zero() Q32 { return Q32{} }

// Q32One returns the fixed-point value 1.
func Q32One() Q32 { return Q32{q32RawOne} }

// Q32Half returns the fixed-point value 1/2.
func Q32Half() Q32 { return Q32{q32RawHalf} }

// Q32MinValue returns the smallest representable Q32 value, -2³¹.
func Q32MinValue() Q32 { return Q32{q32RawMin} }

// Q32MaxValue returns the largest representable Q32 value, 2³¹ - 2⁻³².
func Q32MaxValue() Q32 { return Q32{q32RawMax} }

// Raw returns the signed Q32.32 bit pattern of q.
func (q Q32) Raw() int64 {
	return q.raw
}

// Add returns q+o. It saturates on overflow.
func (q Q32) Add(o Q32) Q32 {
	res := q.raw + o.raw
	if (q.raw >= 0) == (o.raw >= 0) && (res >= 0) != (q.raw >= 0) {
		saturationEvents.Add(1)
		if q.raw >= 0 {
			return Q32{raw: q32RawMax}
		}
		return Q32{raw: q32RawMin}
	}
	return Q32{raw: res}
}

// Sub returns q-o. It saturates on overflow.
func (q Q32) Sub(o Q32) Q32 {
	res := q.raw - o.raw
	if (q.raw >= 0) != (o.raw >= 0) && (res >= 0) != (q.raw >= 0) {
		saturationEvents.Add(1)
		if q.raw >= 0 {
			return Q32{raw: q32RawMax}
		}
		return Q32{raw: q32RawMin}
	}
	return Q32{raw: res}
}

// Mul returns q*o. It floors the product to Q32.32 and saturates on overflow.
func (q Q32) Mul(o Q32) Q32 {
	hi, lo := bits.Mul64(uint64(q.raw), uint64(o.raw))

	if q.raw < 0 {
		hi -= uint64(o.raw)
	}
	if o.raw < 0 {
		hi -= uint64(q.raw)
	}

	res := int64(hi<<32 | lo>>32)
	if int64(hi)>>32 != res>>63 {
		saturationEvents.Add(1)
		if (q.raw >= 0) == (o.raw >= 0) {
			return Q32{raw: q32RawMax}
		}
		return Q32{raw: q32RawMin}
	}
	return Q32{raw: res}
}

// Div returns q/o truncated toward zero. It saturates on overflow.
// It panics when o is zero.
func (q Q32) Div(o Q32) Q32 {
	if o.raw == 0 {
		panicDivZero()
	}
	n, d := magnitude(q.raw), magnitude(o.raw)
	neg := (q.raw < 0) != (o.raw < 0)
	hi, lo := n>>32, n<<32
	if hi >= d {
		return saturatedQuotient(neg)
	}
	quo, _ := bits.Div64(hi, lo, d)
	return signedQuotient(quo, neg)
}

// Sqrt returns floor(sqrt(q)). It panics when q is negative.
// The result cannot overflow.
func (q Q32) Sqrt() Q32 {
	if q.raw < 0 {
		panic("fixed: square root of a negative value")
	}
	// The 96-bit radicand has a root of at most 48 bits.
	return Q32{raw: int64(isqrt128(uint64(q.raw)>>32, uint64(q.raw)<<32))}
}

// Neg returns -q. Neg of MinValue saturates to MaxValue.
func (q Q32) Neg() Q32 {
	if q.raw == q32RawMin {
		saturationEvents.Add(1)
		return Q32{raw: q32RawMax}
	}
	return Q32{raw: -q.raw}
}

// Abs returns the magnitude of q. Abs of MinValue saturates to MaxValue.
func (q Q32) Abs() Q32 {
	if q.raw >= 0 {
		return q
	}
	return q.Neg()
}

// Cmp returns -1 when q < o, 0 when q == o, and 1 when q > o.
func (q Q32) Cmp(o Q32) int {
	if q.raw < o.raw {
		return -1
	}
	if q.raw > o.raw {
		return 1
	}
	return 0
}

// Less reports whether q < o.
func (q Q32) Less(o Q32) bool { return q.raw < o.raw }

// Greater reports whether q > o.
func (q Q32) Greater(o Q32) bool { return q.raw > o.raw }

// Eq reports whether q == o.
func (q Q32) Eq(o Q32) bool { return q.raw == o.raw }

// Min returns the smaller of q and o.
func (q Q32) Min(o Q32) Q32 {
	if o.raw < q.raw {
		return o
	}
	return q
}

// Max returns the larger of q and o.
func (q Q32) Max(o Q32) Q32 {
	if o.raw > q.raw {
		return o
	}
	return q
}

// Clamp returns q limited to [lo, hi]. It requires lo <= hi.
func (q Q32) Clamp(lo, hi Q32) Q32 {
	if q.raw < lo.raw {
		return lo
	}
	if q.raw > hi.raw {
		return hi
	}
	return q
}

// Floor returns the largest integer multiple of One not above q.
func (q Q32) Floor() Q32 { return Q32{raw: q.raw &^ (q32RawOne - 1)} }

// Ceil returns the smallest integer multiple of One that is not below q.
// It saturates when the result is outside the Q32 range.
func (q Q32) Ceil() Q32 {
	f := q.raw &^ (q32RawOne - 1)
	if f == q.raw {
		return q
	}
	// The next integer is outside the Q32 range.
	if f == (1<<31-1)<<32 {
		saturationEvents.Add(1)
		return Q32{raw: q32RawMax}
	}
	return Q32{raw: f + q32RawOne}
}

// Round returns the nearest integer multiple of One. An exact half rounds
// away from zero. Round saturates when the result is outside the Q32 range.
func (q Q32) Round() Q32 {
	if q.raw >= 0 {
		// The nearest integer is outside the Q32 range.
		if q.raw > q32RawMax-q32RawHalf {
			saturationEvents.Add(1)
			return Q32{raw: q32RawMax}
		}
		return Q32{raw: (q.raw + q32RawHalf) &^ (q32RawOne - 1)}
	}
	// The magnitude cannot exceed 2⁶³, so the result cannot pass MinValue.
	rounded := (magnitude(q.raw) + q32RawHalf) &^ uint64(q32RawOne-1)
	return Q32{raw: -int64(rounded)} // A magnitude of 1<<63 converts to MinValue.
}

// Int returns the integer part truncated toward zero.
func (q Q32) Int() int {
	if q.raw >= 0 {
		return int(q.raw >> 32)
	}
	return int(-int64(magnitude(q.raw) >> 32))
}

// ToQ16 floors q to the Q16.16 grid and saturates outside the Q16 range.
// Flooring matches the rounding of Mul.
func (q Q32) ToQ16() Q16 { return q16Saturate(q.raw >> 16) }

func magnitude(v int64) uint64 {
	// Branchless absolute value; MinInt64 maps to 2⁶³ unchanged.
	m := uint64(v >> 63)
	return (uint64(v) ^ m) - m
}

func panicDivZero() {
	panic("fixed: division by zero")
}

// saturatedQuotient handles a quotient at or beyond 2³², off the hot path.
func saturatedQuotient(neg bool) Q32 {
	saturationEvents.Add(1)
	if neg {
		return Q32{raw: q32RawMin}
	}
	return Q32{raw: q32RawMax}
}

// signedQuotient applies the sign and the two output saturation.
func signedQuotient(quo uint64, neg bool) Q32 {
	if neg {
		if quo > 1<<63 {
			saturationEvents.Add(1)
			return Q32{raw: q32RawMin}
		}
		return Q32{raw: -int64(quo)} // A quotient of 1<<63 converts to MinValue.
	}
	if quo > q32RawMax {
		saturationEvents.Add(1)
		return Q32{raw: q32RawMax}
	}
	return Q32{raw: int64(quo)}
}

// divMag returns the signed Q32.32 quotient of magnitudes n and d. It
// truncates toward zero and saturates. The caller must reject d == 0.
func divMag(n, d uint64, neg bool) Q32 {
	hi, lo := n>>32, n<<32
	if hi >= d {
		return saturatedQuotient(neg)
	}
	quo, _ := bits.Div64(hi, lo, d)
	return signedQuotient(quo, neg)
}

// isqrt128 returns floor(sqrt(hi·2⁶⁴+lo)). A hardware square root only
// seeds the answer; exact 128-bit integer checks settle the floor, so
// platform float rounding can never reach the result.
func isqrt128(hi, lo uint64) uint64 {
	if hi == 0 && lo == 0 {
		return 0
	}

	// Near the top of the range the root can equal hi, which the Newton
	// division below cannot take. Two decrements at most resolve it.
	if hi >= ^uint64(0)-1 {
		r := ^uint64(0)
		for {
			pHi, pLo := bits.Mul64(r, r)
			if pHi < hi || (pHi == hi && pLo <= lo) {
				return r
			}
			r--
		}
	}

	f := math.Sqrt(float64(hi)*0x1p64 + float64(lo))
	var x uint64
	if f >= 0x1p64 {
		x = ^uint64(0)
	} else {
		x = uint64(f)
	}

	// A root beyond 52 bits outruns float precision: the seed can be off
	// by thousands of units. One Newton round collapses that to a few.
	if hi >= 1<<40 {
		if x <= hi {
			x = hi + 1
		}
		q, _ := bits.Div64(hi, lo, x)
		sum, carry := bits.Add64(x, q, 0)
		x = sum>>1 | carry<<63
	}

	for {
		pHi, pLo := bits.Mul64(x, x)
		if pHi > hi || (pHi == hi && pLo > lo) {
			x--
			continue
		}
		break
	}
	for x < ^uint64(0) {
		y := x + 1
		pHi, pLo := bits.Mul64(y, y)
		if pHi > hi || (pHi == hi && pLo > lo) {
			break
		}
		x = y
	}
	return x
}
