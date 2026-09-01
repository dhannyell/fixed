package fixed

// Batch functions apply one Q16 operation across whole slices. The scalar
// kernels in this file define the result bits and the saturation count. Every
// vector kernel on every architecture must match them exactly.

// batchKernels holds the kernels chosen for this CPU. Each kernel returns the
// number of saturated elements; the exported wrapper publishes the count.
type batchKernels struct {
	add func(dst, a, b []Q16) uint64
	mul func(dst, a, b []Q16) uint64
	// addWrap records no events, so it returns nothing. It is the speed
	// ceiling that the saturating add is measured against.
	addWrap func(dst, a, b []Q16)
}

// kernels is chosen once, at package initialization. Tests swap it to run the
// parity grid over every path on one machine.
var kernels = selectKernels()

// batchLens panics unless the three slices share one length.
func batchLens(dst, a, b []Q16) {
	if len(a) != len(b) || len(dst) != len(a) {
		panic("fixed: mismatched slice lengths")
	}
}

// Add16 stores a[i]+b[i] into dst. Each element saturates on overflow.
// All three slices must share one length; dst may alias a or b.
func Add16(dst, a, b []Q16) {
	batchLens(dst, a, b)
	if events := kernels.add(dst, a, b); events != 0 {
		saturationEvents.Add(events)
	}
}

// Mul16 stores a[i]*b[i] into dst. It floors each product to Q16.16 and
// saturates on overflow. All three slices must share one length.
func Mul16(dst, a, b []Q16) {
	batchLens(dst, a, b)
	if events := kernels.mul(dst, a, b); events != 0 {
		saturationEvents.Add(events)
	}
}

// Add16Wrap stores a[i]+b[i] into dst with two's-complement wraparound.
func Add16Wrap(dst, a, b []Q16) {
	batchLens(dst, a, b)
	kernels.addWrap(dst, a, b)
}

// add16WrapScalar is the oracle for every wrapping add kernel.
func add16WrapScalar(dst, a, b []Q16) {
	for i := range a {
		dst[i] = Q16{raw: a[i].raw + b[i].raw}
	}
}

// Mul16Wrap stores a[i]*b[i] into dst with two's-complement wraparound.
func Mul16Wrap(dst, a, b []Q16) {
	batchLens(dst, a, b)
	for i := range a {
		dst[i] = Q16{raw: int32((int64(a[i].raw) * int64(b[i].raw)) >> 16)}
	}
}

// Clamp16 stores a[i] limited to [lo, hi] into dst. It requires lo <= hi.
// A clamp is not an overflow, so it records no saturation event.
func Clamp16(dst, a []Q16, lo, hi Q16) {
	if len(dst) != len(a) {
		panic("fixed: mismatched slice lengths")
	}
	for i := range a {
		v := a[i].raw
		if v < lo.raw {
			v = lo.raw
		} else if v > hi.raw {
			v = hi.raw
		}
		dst[i] = Q16{raw: v}
	}
}

// add16Scalar is the oracle for every add kernel.
func add16Scalar(dst, a, b []Q16) uint64 {
	var events uint64
	for i := range a {
		v := int64(a[i].raw) + int64(b[i].raw)
		if v > q16RawMax {
			v = q16RawMax
			events++
		} else if v < q16RawMin {
			v = q16RawMin
			events++
		}
		dst[i] = Q16{raw: int32(v)}
	}
	return events
}

// mul16Scalar is the oracle for every mul kernel.
func mul16Scalar(dst, a, b []Q16) uint64 {
	var events uint64
	for i := range a {
		v := (int64(a[i].raw) * int64(b[i].raw)) >> 16
		if v > q16RawMax {
			v = q16RawMax
			events++
		} else if v < q16RawMin {
			v = q16RawMin
			events++
		}
		dst[i] = Q16{raw: int32(v)}
	}
	return events
}

// The axpyClamp16 kernels all compute the same expression, in the shape of a
// constraint solver inner loop:
//
//	x[i] = clamp(x[i]+m[i]*(v[i]+bias), lo, hi)
//
// The three forms differ only in how they traverse memory. Each publishes its
// own saturation events, so a caller compares them through SaturationCount.

// axpyClampPerCall walks the slices once and calls the scalar Q16 methods.
func axpyClampPerCall(x, m, v []Q16, bias, lo, hi Q16) {
	for i := range x {
		x[i] = x[i].Add(m[i].Mul(v[i].Add(bias))).Clamp(lo, hi)
	}
}

// axpyClampMultipass runs four batch primitives over scratch buffers. It shows
// the memory traffic that a one-operation-per-pass API forces.
// biasVec must hold bias in every element; tmp must match x in length.
func axpyClampMultipass(x, m, v, biasVec, tmp []Q16, lo, hi Q16) {
	Add16(tmp, v, biasVec)
	Mul16(tmp, m, tmp)
	Add16(x, x, tmp)
	Clamp16(x, x, lo, hi)
}

// axpyClampFused walks the slices once and keeps every intermediate in a
// register. It is the scalar shape that a fused vector kernel must match.
func axpyClampFused(x, m, v []Q16, bias, lo, hi Q16) {
	var events uint64
	for i := range x {
		s := int64(v[i].raw) + int64(bias.raw)
		if s > q16RawMax {
			s = q16RawMax
			events++
		} else if s < q16RawMin {
			s = q16RawMin
			events++
		}
		p := (int64(m[i].raw) * s) >> 16
		if p > q16RawMax {
			p = q16RawMax
			events++
		} else if p < q16RawMin {
			p = q16RawMin
			events++
		}
		t := int64(x[i].raw) + p
		if t > q16RawMax {
			t = q16RawMax
			events++
		} else if t < q16RawMin {
			t = q16RawMin
			events++
		}
		r := int32(t)
		if r < lo.raw {
			r = lo.raw
		} else if r > hi.raw {
			r = hi.raw
		}
		x[i] = Q16{raw: r}
	}
	if events != 0 {
		saturationEvents.Add(events)
	}
}
