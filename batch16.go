package fixed

// Batch functions apply one Q16 operation across whole slices. The scalar
// kernels in this file define the result bits and the saturation count. Every
// vector kernel on every architecture must match them exactly.

// batchKernels holds the kernels chosen for this CPU. Each kernel returns the
// number of saturated elements; the exported wrapper publishes the count.
type batchKernels struct {
	add func(dst, a, b []Q16) uint64
	mul func(dst, a, b []Q16) uint64
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
