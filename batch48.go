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
	i := 0
	for ; i+dot16Lanes <= len(a); i += dot16Lanes {
		var e uint64
		partial[0], e = dot16Accumulate(partial[0], a[i].raw, b[i].raw)
		events += e
		partial[1], e = dot16Accumulate(partial[1], a[i+1].raw, b[i+1].raw)
		events += e
		partial[2], e = dot16Accumulate(partial[2], a[i+2].raw, b[i+2].raw)
		events += e
		partial[3], e = dot16Accumulate(partial[3], a[i+3].raw, b[i+3].raw)
		events += e
		partial[4], e = dot16Accumulate(partial[4], a[i+4].raw, b[i+4].raw)
		events += e
		partial[5], e = dot16Accumulate(partial[5], a[i+5].raw, b[i+5].raw)
		events += e
		partial[6], e = dot16Accumulate(partial[6], a[i+6].raw, b[i+6].raw)
		events += e
		partial[7], e = dot16Accumulate(partial[7], a[i+7].raw, b[i+7].raw)
		events += e
	}
	for ; i < len(a); i++ {
		var e uint64
		lane := i & (dot16Lanes - 1)
		partial[lane], e = dot16Accumulate(partial[lane], a[i].raw, b[i].raw)
		events += e
	}
	return dot16Reduce(partial, &events), events
}

func dot16Accumulate(sum int64, a, b int32) (int64, uint64) {
	r, ovf := q48AddSat(sum, (int64(a)*int64(b))>>16)
	if ovf {
		return r, 1
	}
	return r, 0
}

// dot16Reduce folds the partials as a balanced tree: (0+1)+(2+3) and so on.
func dot16Reduce(p [dot16Lanes]int64, events *uint64) Q48 {
	var e uint64
	p[0], e = dot16Add(p[0], p[1])
	*events += e
	p[1], e = dot16Add(p[2], p[3])
	*events += e
	p[2], e = dot16Add(p[4], p[5])
	*events += e
	p[3], e = dot16Add(p[6], p[7])
	*events += e
	p[0], e = dot16Add(p[0], p[1])
	*events += e
	p[1], e = dot16Add(p[2], p[3])
	*events += e
	p[0], e = dot16Add(p[0], p[1])
	*events += e
	return Q48{raw: p[0]}
}

func dot16Add(x, y int64) (int64, uint64) {
	r, ovf := q48AddSat(x, y)
	if ovf {
		return r, 1
	}
	return r, 0
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
