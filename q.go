package fixed

import (
	"math"
	"math/bits"
	"sync/atomic"
)

// Q stores an opaque signed Q32.32 value. Its zero value is 0.
type Q struct {
	raw int64
}

const (
	rawHalf = 1 << 31
	rawOne  = 1 << 32
	rawMin  = -1 << 63
	rawMax  = 1<<63 - 1
)

// FromInt returns i as a Q value. It saturates outside [-2³¹, 2³¹-1].
func FromInt(i int) Q {
	if i > 1<<31-1 {
		saturationEvents.Add(1)
		return Q{raw: rawMax}
	}
	if i < -1<<31 {
		saturationEvents.Add(1)
		return Q{raw: rawMin}
	}
	return Q{raw: int64(i) << 32}
}

// FromRatio returns num/den truncated toward zero. It saturates on overflow.
// It panics when den is zero.
func FromRatio(num, den int) Q {
	if den == 0 {
		panicDivZero()
	}
	return divMag(magnitude(int64(num)), magnitude(int64(den)), (num < 0) != (den < 0))
}

// FromRaw returns the Q value with the specified signed bit pattern.
func FromRaw(raw int64) Q { return Q{raw: raw} }

// Zero returns the fixed-point value 0.
func Zero() Q { return Q{} }

// One returns the fixed-point value 1.
func One() Q { return Q{rawOne} }

// Half returns the fixed-point value 1/2.
func Half() Q { return Q{rawHalf} }

// MinValue returns the smallest representable Q value, -2³¹.
func MinValue() Q { return Q{rawMin} }

// MaxValue returns the largest representable Q value, 2³¹ - 2⁻³².
func MaxValue() Q { return Q{rawMax} }

// Raw returns the signed Q32.32 bit pattern of q.
func (q Q) Raw() int64 {
	return q.raw
}

// Add returns q+o. It saturates on overflow.
func (q Q) Add(o Q) Q {
	res := q.raw + o.raw
	if (q.raw >= 0) == (o.raw >= 0) && (res >= 0) != (q.raw >= 0) {
		saturationEvents.Add(1)
		if q.raw >= 0 {
			return Q{raw: rawMax}
		}
		return Q{raw: rawMin}
	}
	return Q{raw: res}
}

// Sub returns q-o. It saturates on overflow.
func (q Q) Sub(o Q) Q {
	res := q.raw - o.raw
	if (q.raw >= 0) != (o.raw >= 0) && (res >= 0) != (q.raw >= 0) {
		saturationEvents.Add(1)
		if q.raw >= 0 {
			return Q{raw: rawMax}
		}
		return Q{raw: rawMin}
	}
	return Q{raw: res}
}

// Mul returns q*o. It floors the product to Q32.32 and saturates on overflow.
func (q Q) Mul(o Q) Q {
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
			return Q{raw: rawMax}
		}
		return Q{raw: rawMin}
	}
	return Q{raw: res}
}

// Div returns q/o truncated toward zero. It saturates on overflow.
// It panics when o is zero.
func (q Q) Div(o Q) Q {
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
func (q Q) Sqrt() Q {
	if q.raw < 0 {
		panic("fixed: square root of a negative value")
	}
	// The 96-bit radicand has a root of at most 48 bits.
	return Q{raw: int64(isqrt128(uint64(q.raw)>>32, uint64(q.raw)<<32))}
}

// Neg returns -q. Neg of MinValue saturates to MaxValue.
func (q Q) Neg() Q {
	if q.raw == rawMin {
		saturationEvents.Add(1)
		return Q{raw: rawMax}
	}
	return Q{raw: -q.raw}
}

// Abs returns the magnitude of q. Abs of MinValue saturates to MaxValue.
func (q Q) Abs() Q {
	if q.raw >= 0 {
		return q
	}
	return q.Neg()
}

// Cmp returns -1 when q < o, 0 when q == o, and 1 when q > o.
func (q Q) Cmp(o Q) int {
	if q.raw < o.raw {
		return -1
	}
	if q.raw > o.raw {
		return 1
	}
	return 0
}

// Less reports whether q < o.
func (q Q) Less(o Q) bool { return q.raw < o.raw }

// Eq reports whether q == o.
func (q Q) Eq(o Q) bool { return q.raw == o.raw }

// Min returns the smaller of q and o.
func (q Q) Min(o Q) Q {
	if o.raw < q.raw {
		return o
	}
	return q
}

// Max returns the larger of q and o.
func (q Q) Max(o Q) Q {
	if o.raw > q.raw {
		return o
	}
	return q
}

// Clamp returns q limited to [lo, hi]. It requires lo <= hi.
func (q Q) Clamp(lo, hi Q) Q {
	if q.raw < lo.raw {
		return lo
	}
	if q.raw > hi.raw {
		return hi
	}
	return q
}

// Floor returns the largest integer multiple of One not above q.
func (q Q) Floor() Q { return Q{raw: q.raw &^ (rawOne - 1)} }

// Ceil returns the smallest integer multiple of One that is not below q.
// It saturates when the result is outside the Q range.
func (q Q) Ceil() Q {
	f := q.raw &^ (rawOne - 1)
	if f == q.raw {
		return q
	}
	// The next integer is outside the Q range.
	if f == (1<<31-1)<<32 {
		saturationEvents.Add(1)
		return Q{raw: rawMax}
	}
	return Q{raw: f + rawOne}
}

// Round returns the nearest integer multiple of One. An exact half rounds
// away from zero. Round saturates when the result is outside the Q range.
func (q Q) Round() Q {
	if q.raw >= 0 {
		// The nearest integer is outside the Q range.
		if q.raw > rawMax-rawHalf {
			saturationEvents.Add(1)
			return Q{raw: rawMax}
		}
		return Q{raw: (q.raw + rawHalf) &^ (rawOne - 1)}
	}
	// The magnitude cannot exceed 2⁶³, so the result cannot pass MinValue.
	rounded := (magnitude(q.raw) + rawHalf) &^ uint64(rawOne-1)
	return Q{raw: -int64(rounded)} // A magnitude of 1<<63 converts to MinValue.
}

// Int returns the integer part truncated toward zero.
func (q Q) Int() int {
	if q.raw >= 0 {
		return int(q.raw >> 32)
	}
	return int(-int64(magnitude(q.raw) >> 32))
}

func magnitude(v int64) uint64 {
	// Branchless absolute value; MinInt64 maps to 2⁶³ unchanged.
	m := uint64(v >> 63)
	return (uint64(v) ^ m) - m
}

func panicDivZero() {
	panic("fixed: division by zero")
}

// saturatedQuotient handles a quotient at or beyond 2³², off the hot path.
func saturatedQuotient(neg bool) Q {
	saturationEvents.Add(1)
	if neg {
		return Q{raw: rawMin}
	}
	return Q{raw: rawMax}
}

// signedQuotient applies the sign and the two output saturations.
func signedQuotient(quo uint64, neg bool) Q {
	if neg {
		if quo > 1<<63 {
			saturationEvents.Add(1)
			return Q{raw: rawMin}
		}
		return Q{raw: -int64(quo)} // A quotient of 1<<63 converts to MinValue.
	}
	if quo > rawMax {
		saturationEvents.Add(1)
		return Q{raw: rawMax}
	}
	return Q{raw: int64(quo)}
}

// divMag returns the signed Q32.32 quotient of magnitudes n and d. It
// truncates toward zero and saturates. The caller must reject d == 0.
func divMag(n, d uint64, neg bool) Q {
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

// saturationEvents records diagnostic data. It does not affect Q values or
// operation results.
var saturationEvents atomic.Uint64

// SaturationCount reports the number of saturation events since the last reset.
func SaturationCount() uint64 { return saturationEvents.Load() }

// ResetSaturationCount zeroes the saturation counter.
func ResetSaturationCount() { saturationEvents.Store(0) }
