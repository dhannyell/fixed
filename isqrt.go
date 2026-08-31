package fixed

import (
	"math"
	"math/bits"
)

// isqrt64 returns floor(sqrt(n)) with an integer final decision.
func isqrt64(n uint64) uint64 {
	if n == 0 {
		return 0
	}
	seed := math.Sqrt(float64(n))
	const maxRoot = uint64(1<<32 - 1)
	var x uint64
	if seed >= float64(1<<32) {
		x = maxRoot
	} else {
		x = uint64(seed)
	}
	for x*x > n {
		x--
	}
	for x < maxRoot {
		y := x + 1
		if y*y > n {
			break
		}
		x = y
	}
	return x
}

// isqrt128 returns floor(sqrt(hi·2⁶⁴+lo)). A hardware square root only
// seeds the answer; exact 128-bit integer checks settle the floor, so
// platform float rounding can never reach the result.
func isqrt128(hi, lo uint64) uint64 {
	if hi == 0 {
		return isqrt64(lo)
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
