package fixed

// Batch functions apply one Q16 operation across whole slices. Scalar kernels
// specify the result bits and saturation count. Architecture-specific kernels
// must match them exactly.
//
// Aliasing: dst may be the same slice as a source, so an operation can run in
// place. Any other overlap is undefined, because the scalar and the vector
// kernels read and write in different block sizes and would disagree.

// batchKernels holds the kernels chosen for this CPU. A kernel that can
// saturate returns the number of saturated elements; the exported wrapper
// publishes the count.
type batchKernels struct {
	path       string
	add        func(dst, a, b []Q16) uint64
	sub        func(dst, a, b []Q16) uint64
	mul        func(dst, a, b []Q16) uint64
	clamp      func(dst, a []Q16, lo, hi Q16)
	q32FromQ16 func(dst []Q32, a []Q16)
	q16FromQ32 func(dst []Q16, a []Q32) uint64
}

// kernels holds the dispatch table selected at package initialization.
var kernels = selectKernels()

func batchLens(dst, a, b []Q16) {
	if len(a) != len(b) || len(dst) != len(a) {
		panic("fixed: mismatched slice lengths")
	}
}

func batchLens2(dst, a int) {
	if dst != a {
		panic("fixed: mismatched slice lengths")
	}
}

// BatchPath reports the kernel family behind the batch functions on this
// host: "scalar", "avx2" or "neon". Every path produces the same bits.
func BatchPath() string { return kernels.path }

// BatchAdd16 stores a[i]+b[i] into dst. Each element saturates on overflow.
// All three slices must share one length. dst may be the same slice as a or b;
// no other overlap is allowed.
func BatchAdd16(dst, a, b []Q16) {
	batchLens(dst, a, b)
	if events := kernels.add(dst, a, b); events != 0 {
		saturationEvents.Add(events)
	}
}

// BatchSub16 stores a[i]-b[i] into dst. Each element saturates on overflow.
// All three slices must share one length. dst may be the same slice as a or b;
// no other overlap is allowed.
func BatchSub16(dst, a, b []Q16) {
	batchLens(dst, a, b)
	if events := kernels.sub(dst, a, b); events != 0 {
		saturationEvents.Add(events)
	}
}

// BatchMul16 stores a[i]*b[i] into dst. It floors each product to Q16.16 and
// saturates on overflow. All three slices must share one length. dst may be
// the same slice as a or b; no other overlap is allowed.
func BatchMul16(dst, a, b []Q16) {
	batchLens(dst, a, b)
	if events := kernels.mul(dst, a, b); events != 0 {
		saturationEvents.Add(events)
	}
}

// BatchClamp16 stores a[i] limited to [lo, hi] into dst. It requires lo <= hi.
// A clamp is not an overflow, so it records no saturation event. Both slices
// must share one length. dst may be the same slice as a; no other overlap is
// allowed.
func BatchClamp16(dst, a []Q16, lo, hi Q16) {
	batchLens2(len(dst), len(a))
	kernels.clamp(dst, a, lo, hi)
}

// BatchQ32FromQ16 stores a[i] converted to Q32 into dst. The conversion is exact
// and never saturates. Both slices must share one length.
func BatchQ32FromQ16(dst []Q32, a []Q16) {
	batchLens2(len(dst), len(a))
	kernels.q32FromQ16(dst, a)
}

// BatchQ16FromQ32 stores a[i] floored to the Q16.16 grid into dst. Each element
// saturates outside the Q16 range. Both slices must share one length.
func BatchQ16FromQ32(dst []Q16, a []Q32) {
	batchLens2(len(dst), len(a))
	if events := kernels.q16FromQ32(dst, a); events != 0 {
		saturationEvents.Add(events)
	}
}

// q16SaturateWide clamps an out-of-range widened value. It is the cold tail of
// the batch kernels; the hot loop keeps one comparison.
//
//go:noinline
func q16SaturateWide(v int64) int32 {
	if v > q16RawMax {
		return q16RawMax
	}
	return q16RawMin
}

func add16Scalar(dst, a, b []Q16) uint64 {
	// Reslicing to one length lets the compiler drop the bounds checks that
	// the hot loop would otherwise repeat on every element.
	dst = dst[:len(a)]
	b = b[:len(a)]
	var events uint64
	for i := range a {
		v := int64(a[i].raw) + int64(b[i].raw)
		r := int32(v)
		if int64(r) != v {
			r = q16SaturateWide(v)
			events++
		}
		dst[i] = Q16{raw: r}
	}
	return events
}

func sub16Scalar(dst, a, b []Q16) uint64 {
	dst = dst[:len(a)]
	b = b[:len(a)]
	var events uint64
	for i := range a {
		v := int64(a[i].raw) - int64(b[i].raw)
		r := int32(v)
		if int64(r) != v {
			r = q16SaturateWide(v)
			events++
		}
		dst[i] = Q16{raw: r}
	}
	return events
}

func mul16Scalar(dst, a, b []Q16) uint64 {
	dst = dst[:len(a)]
	b = b[:len(a)]
	var events uint64
	for i := range a {
		v := (int64(a[i].raw) * int64(b[i].raw)) >> 16
		r := int32(v)
		if int64(r) != v {
			r = q16SaturateWide(v)
			events++
		}
		dst[i] = Q16{raw: r}
	}
	return events
}

func clamp16Scalar(dst, a []Q16, lo, hi Q16) {
	dst = dst[:len(a)]
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

func q32FromQ16Scalar(dst []Q32, a []Q16) {
	dst = dst[:len(a)]
	i := 0
	// Two elements per step keep the loop counter off the critical path of
	// the widening store, which is twice as wide as the load.
	for ; i+2 <= len(a); i += 2 {
		dst[i] = Q32{raw: int64(a[i].raw) << 16}
		dst[i+1] = Q32{raw: int64(a[i+1].raw) << 16}
	}
	for ; i < len(a); i++ {
		dst[i] = Q32{raw: int64(a[i].raw) << 16}
	}
}

func q16FromQ32Scalar(dst []Q16, a []Q32) uint64 {
	dst = dst[:len(a)]
	var events uint64
	for i := range a {
		v := a[i].raw >> 16
		r := int32(v)
		if int64(r) != v {
			r = q16SaturateWide(v)
			events++
		}
		dst[i] = Q16{raw: r}
	}
	return events
}
