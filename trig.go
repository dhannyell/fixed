package fixed

import "math/bits"

// SinTurns returns the sine of the angle t, measured in turns: One()
// is a full revolution. The function is periodic by contract — only
// the fractional part of t is used — so every Q is a valid input; it
// never panics and never saturates. The result stays in [-One, One].
//
// The kernel is the 1024-interval quarter-wave table of trig_table.go
// with linear interpolation: table entries round to nearest, the
// interpolation floors, and the maximum absolute error is 2⁻²⁰.
func SinTurns(t Q) Q {
	return Q{raw: sinFrac(uint32(t.raw))}
}

// CosTurns returns the cosine of the angle t in turns. It shares the
// contract and the kernel of SinTurns through the quarter-turn shift.
func CosTurns(t Q) Q {
	return Q{raw: sinFrac(uint32(t.raw) + 1<<30)}
}

// Atan2Turns returns the angle of the vector (x, y) in (-1/2, 1/2]
// turns. Atan2Turns(Zero(), Zero()) returns Zero, matching the float
// convention. It never panics and never saturates; the maximum
// absolute error is 2⁻²⁰ turn.
func Atan2Turns(y, x Q) Q {
	ay, ax := magnitude(y.raw), magnitude(x.raw)
	if ay == 0 && ax == 0 {
		return Q{}
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
	return Q{raw: a}
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

// sinFrac evaluates sine on the 32-bit turn fraction. The truncating
// uint32 conversion above equals t.Sub(t.Floor()) in two's complement,
// so the range reduction is exact.
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
