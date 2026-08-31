package fixed

import "math/bits"

// SinTurns returns the sine of t, where One is a full revolution. It
// uses only the fractional part of t, accepts every Q32 value, and returns
// a value in [-One, One] without saturation.
//
// The 1024-interval quarter-wave table rounds entries to nearest. Linear
// interpolation floors. The maximum absolute error is 2⁻²⁰.
func SinTurns(t Q32) Q32 {
	return Q32{raw: sinFrac(uint32(t.raw))}
}

// CosTurns returns the cosine of t in turns. It uses the same rules as
// SinTurns with a quarter-turn shift.
func CosTurns(t Q32) Q32 {
	return Q32{raw: sinFrac(uint32(t.raw) + 1<<30)}
}

// Atan2Turns returns the angle of (x, y) in (-1/2, 1/2] turns.
// Atan2Turns(Zero(), Zero()) returns Zero. It never panics or saturates.
// The unit ratio is truncated to Q32.32, table entries round to nearest,
// and linear interpolation floors. The maximum absolute error is 2⁻²⁰ turn.
func Atan2Turns(y, x Q32) Q32 {
	ay, ax := magnitude(y.raw), magnitude(x.raw)
	if ay == 0 && ax == 0 {
		return Q32{}
	}
	num, den := ay, ax
	if num > den {
		num, den = den, num
	}
	// Unit ratio num/den on the Q32.32 grid via 128-bit division.
	// num <= den keeps the quotient inside [0, 2³²]: no saturation.
	quo, _ := bits.Div64(num>>32, num<<32, den)
	a := atanLerp(quo)
	if ay > ax {
		a = 1<<30 - a
	}
	if x.raw < 0 {
		a = 1<<31 - a
	}
	if y.raw < 0 {
		a = -a
	}
	return Q32{raw: a}
}

// atanLerp evaluates atan on the unit ratio r in [0, 2³²] with the
// 1024-interval table.
func atanLerp(r uint64) int64 {
	idx := r >> 22
	v := atanUnit[idx]
	if rem := r & (1<<22 - 1); rem != 0 {
		// atan is monotonic, so the delta is non-negative.
		d := uint64(atanUnit[idx+1] - v)
		v += int64(d * rem >> 22)
	}
	return v
}

// sinFrac evaluates sine on a 32-bit turn fraction. Converting a raw
// Q32 value to uint32 performs exact range reduction in two's complement.
func sinFrac(u uint32) int64 {
	quad := u >> 30
	pos := u & (1<<30 - 1)
	if quad&1 == 1 {
		pos = 1<<30 - pos
	}
	idx := pos >> 20
	v := sineQuarter[idx]
	if rem := uint64(pos) & (1<<20 - 1); rem != 0 {
		// The quarter is monotonic, so the delta is non-negative.
		d := uint64(sineQuarter[idx+1] - v)
		v += int64(d * rem >> 20)
	}
	if quad >= 2 {
		v = -v
	}
	return v
}
