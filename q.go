package fixed

import (
	"math/bits"
	"sync/atomic"
)

type Q struct {
	raw int64
}

const (
	rawHalf = 1 << 31
	rawOne  = 1 << 32
	rawMin  = -1 << 63
	rawMax  = 1<<63 - 1
)

// Constructors

// FromInt returns the fixed-point value of i, saturating to MinValue or MaxValue.
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

// MustParse returns the value of a decimal literal,
// rounded to the nearest representable value, half away from zero.
// Values outside the range saturate. MustParse panics on malformed inputs.
func MustParse(s string) Q {
	rest := s
	neg := len(rest) > 0 && rest[0] == '-'
	if neg {
		rest = rest[1:]
	}
	intStr, fracStr := rest, ""
	for i := 0; i < len(rest); i++ {
		if rest[i] == '.' {
			intStr, fracStr = rest[:i], rest[i+1:]
			if fracStr == "" {
				panic("fixed: malformed literal: " + s)
			}
			break
		}
	}
	if intStr == "" {
		panic("fixed: malformed literal: " + s)
	}
	if len(fracStr) > 19 {
		panic("fixed: literal has more than 19 fraction digits: " + s)
	}

	var intPart uint64
	for i := 0; i < len(intStr); i++ {
		d := intStr[i] - '0'
		if d > 9 {
			panic("fixed: malformed literal: " + s)
		}
		if intPart <= 1<<31 {
			intPart = intPart*10 + uint64(d)
		}
	}

	var rawFrac uint64
	if fracStr != "" {
		var frac, den uint64 = 0, 1
		for i := 0; i < len(fracStr); i++ {
			d := fracStr[i] - '0'
			if d > 9 {
				panic("fixed: malformed literal: " + s)
			}
			frac = frac*10 + uint64(d)
			den *= 10
		}
		hi, lo := frac>>32, frac<<32
		quo, rem := bits.Div64(hi, lo, den)
		if den-rem <= rem { // half away from zero rounds up
			quo++
		}
		rawFrac = quo
	}

	if intPart > 1<<31 {
		saturationEvents.Add(1)
		if neg {
			return Q{raw: rawMin}
		}
		return Q{raw: rawMax}
	}
	rawU := intPart<<32 + rawFrac
	if neg {
		if rawU > 1<<63 {
			saturationEvents.Add(1)
			return Q{raw: rawMin}
		}
		return Q{raw: -int64(rawU)}
	}
	if rawU > rawMax {
		saturationEvents.Add(1)
		return Q{raw: rawMax}
	}
	return Q{raw: int64(rawU)}
}

// FromRatio returns num/den truncated toward zero, saturating to
// MinValue or MaxValue on overflow. FromRatio panics when den is zero.
func FromRatio(num, den int) Q {
	if den == 0 {
		panic("fixed: division by zero")
	}
	return divMag(magnitude(int64(num)), magnitude(int64(den)), (num < 0) != (den < 0))
}

// FromRaw returns the value whose bit pattern is raw. Use it for
// serialization, checksums and the GPU mirror.
func FromRaw(raw int64) Q { return Q{raw: raw} }

// GET/SETTERS

// Zero returns the fixed-point value 0.
func Zero() Q { return Q{} }

// One returns the fixed-point value 1.
func One() Q { return Q{rawOne} }

// Half returns the half fixed-point value an int64 can store.
func Half() Q { return Q{rawHalf} }

// MinValue returns the most negative int64 representable value.
func MinValue() Q { return Q{rawMin} }

// MaxValue returns the largest int64 representable value.
func MaxValue() Q { return Q{rawMax} }

// Raw returns the current raw value of Q.
func (q Q) Raw() int64 {
	return q.raw
}

// Arithmetic

// Add returns a sum of two numbers, saturating to MinValue or MaxValue on overflow.
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

// Sub returns the subtraction of two numbers, saturating to MinValue or MaxValue on overflow.
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

// Mul returns the result of two int64 variables multiplication using a 128bit intermediate product variable.
// Saturating to MinValue or MaxValue on overflow.
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

// Div returns the quotient truncated toward zero, saturating to
// MinValue or MaxValue on overflow. Div panics when input is zero.
func (q Q) Div(o Q) Q {
	if o.raw == 0 {
		panic("fixed: division by zero")
	}
	return divMag(magnitude(q.raw), magnitude(o.raw), (q.raw < 0) != (o.raw < 0))
}

// Sqrt returns the square root, floored to the nearest representable value.
// Panics when q is negative. Never saturates, the result of a 96-bit radicand fits in 48 bits.
func (q Q) Sqrt() Q {
	if q.raw < 0 {
		panic("fixed: square root of a negative value")
	}
	hi := uint64(q.raw) >> 32
	lo := uint64(q.raw) << 32

	var root, rem uint64
	for range 64 {
		rem = rem<<2 | hi>>62
		hi = hi<<2 | lo>>62
		lo <<= 2
		root <<= 1
		if t := root<<1 | 1; rem >= t {
			rem -= t
			root |= 1
		}
	}
	return Q{raw: int64(root)}
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

// Comparison

// Cmp returns -1 when q < o, 0 when q == o and 1 when q > o.
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

// Integers

// Floor returns the largest integer multiple of One not above q.
func (q Q) Floor() Q { return Q{raw: q.raw &^ (rawOne - 1)} }

// Ceil returns the smallest integer multiple of One not below q,
// saturating to MaxValue when the result leaves the range.
func (q Q) Ceil() Q {
	f := q.raw &^ (rawOne - 1)
	if f == q.raw {
		return q
	}
	if f == (1<<31-1)<<32 { // the top integer; one more leaves the range
		saturationEvents.Add(1)
		return Q{raw: rawMax}
	}
	return Q{raw: f + rawOne}
}

// Round returns the nearest integer multiple of One, half away from
// zero, saturating to MaxValue when the result leaves the range.
func (q Q) Round() Q {
	if q.raw >= 0 {
		if q.raw > rawMax-rawHalf { // rounds past the top of the range
			saturationEvents.Add(1)
			return Q{raw: rawMax}
		}
		return Q{raw: (q.raw + rawHalf) &^ (rawOne - 1)}
	}
	// |q| never rounds below MinValue: the magnitude tops out at 2⁶³.
	rounded := (magnitude(q.raw) + rawHalf) &^ uint64(rawOne-1)
	return Q{raw: -int64(rounded)} // rounded == 1<<63 lands exactly on MinValue
}

// Int returns the integer part truncated toward zero.
func (q Q) Int() int {
	if q.raw >= 0 {
		return int(q.raw >> 32)
	}
	return int(-int64(magnitude(q.raw) >> 32))
}

// String returns the exact decimal form of q, such as "-6.25".
// Presentation only; use Raw for exact transport.
func (q Q) String() string {
	n := magnitude(q.raw)
	intPart := n >> 32
	frac := n & (rawOne - 1)

	var buf []byte
	if q.raw < 0 {
		buf = append(buf, '-')
	}
	var digits [10]byte
	i := len(digits)
	for {
		i--
		digits[i] = byte('0' + intPart%10)
		intPart /= 10
		if intPart == 0 {
			break
		}
	}
	buf = append(buf, digits[i:]...)
	if frac != 0 {
		buf = append(buf, '.')
		// Each ×10 shifts one binary digit out; 2⁻³² ends after 32 digits.
		for frac != 0 {
			frac *= 10
			buf = append(buf, byte('0'+frac>>32))
			frac &= rawOne - 1
		}
	}
	return string(buf)
}

// Helpers

// Clamp limits q to [lo, hi]. Precondition: lo <= hi.
func (q Q) Clamp(lo, hi Q) Q {
	if q.raw < lo.raw {
		return lo
	}
	if q.raw > hi.raw {
		return hi
	}
	return q
}

func magnitude(v int64) uint64 {
	if v < 0 {
		return -uint64(v)
	}
	return uint64(v)
}

// divMag returns n·2³²/d with the given sign, truncated toward zero and
// saturating on overflow. Callers guard d == 0.
func divMag(n, d uint64, neg bool) Q {
	hi, lo := n>>32, n<<32
	if hi >= d {
		saturationEvents.Add(1)
		if neg {
			return Q{raw: rawMin}
		}
		return Q{raw: rawMax}
	}
	quo, _ := bits.Div64(hi, lo, d)
	if neg {
		if quo > 1<<63 {
			saturationEvents.Add(1)
			return Q{raw: rawMin}
		}
		return Q{raw: -int64(quo)} // quo == 1<<63 lands exactly on MinValue
	}
	if quo > rawMax {
		saturationEvents.Add(1)
		return Q{raw: rawMax}
	}
	return Q{raw: int64(quo)}
}

// Telemetry

// saturationEvents counts saturation events for dev telemetry. Not
// simulation state: it never enters checksums, snapshots or replay.
var saturationEvents atomic.Uint64

// SaturationCount reports the number of saturation events since the last reset.
func SaturationCount() uint64 { return saturationEvents.Load() }

// ResetSaturationCount zeroes the saturation counter.
func ResetSaturationCount() { saturationEvents.Store(0) }
