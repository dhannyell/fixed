package fixed

import "math/bits"

// Batch functions with a Q48 result accumulate or scale Q16 data. As in
// batch16.go, the scalar kernels specify the bits and the saturation count.

// dot16Lanes is the number of partial sums in BatchDot16. Element i joins
// partial i mod dot16Lanes; the partials then reduce as a balanced tree. A
// vector kernel with this many 64-bit lanes per block reproduces the order
// for free.
const dot16Lanes = 8

// BatchDot16 returns the sum of a[i]*b[i] as Q48. Each product is exact and
// floored to the shared grid; each addition saturates. Both slices must share
// one length.
//
// The summation order is fixed: element i joins partial sum i mod 8, and the
// eight partials then reduce as a balanced tree. Without saturation this equals
// a loop over [Q48.MulAdd16]. With saturation the order decides the result, so
// every kernel follows this one order and the bits match on every host.
func BatchDot16(a, b []Q16) Q48 {
	batchLens2(len(a), len(b))
	r, events := kernels.dot16(a, b)
	if events != 0 {
		saturationEvents.Add(events)
	}
	return r
}

// BatchQ48Mul16 stores q[i]*f[i] into dst. It floors each product to Q48.16
// and saturates on overflow, as [Q48.Mul16] does. All three slices must share
// one length. dst may be the same slice as q; no other overlap is allowed.
func BatchQ48Mul16(dst, q []Q48, f []Q16) {
	if len(q) != len(f) || len(dst) != len(q) {
		panic("fixed: mismatched slice lengths")
	}
	if events := kernels.q48Mul16(dst, q, f); events != 0 {
		saturationEvents.Add(events)
	}
}

// q48AddSat is the saturating add on raw values with an event flag.
func q48AddSat(x, y int64) (int64, bool) {
	r := x + y
	if (x >= 0) == (y >= 0) && (r >= 0) != (x >= 0) {
		return q48RawMax ^ (x >> 63), true
	}
	return r, false
}

// dot16Scalar is the canonical order every dot kernel must reproduce.
func dot16Scalar(a, b []Q16) (Q48, uint64) {
	b = b[:len(a)]
	var partial [dot16Lanes]int64
	var events uint64
	for i := range a {
		p := (int64(a[i].raw) * int64(b[i].raw)) >> 16
		r, ovf := q48AddSat(partial[i%dot16Lanes], p)
		if ovf {
			events++
		}
		partial[i%dot16Lanes] = r
	}
	return dot16Reduce(partial, &events), events
}

// dot16Reduce folds the partials as a balanced tree: (0+1)+(2+3) and so on.
func dot16Reduce(p [dot16Lanes]int64, events *uint64) Q48 {
	for width := dot16Lanes; width > 1; width /= 2 {
		for j := range width / 2 {
			r, ovf := q48AddSat(p[2*j], p[2*j+1])
			if ovf {
				*events++
			}
			p[j] = r
		}
	}
	return Q48{raw: p[0]}
}

// q48Mul16Raw is Q48.Mul with a Q16 operand and a local overflow flag.
func q48Mul16Raw(q int64, f int32) (int64, bool) {
	o := int64(f)
	hi, lo := bits.Mul64(uint64(q), uint64(o))
	if q < 0 {
		hi -= uint64(o)
	}
	if o < 0 {
		hi -= uint64(q)
	}
	res := int64(hi<<48 | lo>>16)
	if int64(hi)>>16 != res>>63 {
		return q48RawMax ^ (int64(hi) >> 63), true
	}
	return res, false
}

func q48Mul16Scalar(dst, q []Q48, f []Q16) uint64 {
	dst = dst[:len(q)]
	f = f[:len(q)]
	var events uint64
	for i := range q {
		r, ovf := q48Mul16Raw(q[i].raw, f[i].raw)
		if ovf {
			events++
		}
		dst[i] = Q48{raw: r}
	}
	return events
}
