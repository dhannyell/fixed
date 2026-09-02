//go:build goexperiment.simd && arm64

package fixed

import (
	"simd/archsimd"
	"unsafe"
)

// NEON is mandatory on ARMv8, so every kernel here runs on every arm64 CPU.
// There is no tier to choose.

// selectKernels returns the kernels for this CPU.
func selectKernels() batchKernels {
	return batchKernels{
		path:       "neon",
		add:        add16NEON,
		sub:        sub16NEON,
		mul:        mul16Scalar,
		clamp:      clamp16NEON,
		q32FromQ16: q32FromQ16Scalar,
		q16FromQ32: q16FromQ32Scalar,
	}
}

// rawInt32 reinterprets a Q16 slice as int32. Q16 is struct{raw int32}, so the
// layout is identical.
func rawInt32(s []Q16) []int32 {
	return unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
}

// vecLaneSum adds the four lanes of a counter vector. The counter holds one
// non-negative count per lane, so the sum cannot overflow.
func vecLaneSum(count archsimd.Int32x4) uint64 {
	var events uint64
	for lane := range 4 {
		events += uint64(count.GetElem(uint8(lane)))
	}
	return events
}

// add16NEON adds four lanes per step. NEON has a saturating 32-bit add, so the
// clamp is one instruction. Mask32x4 has no bitmask form, so the counter
// accumulates the compare result as a vector of 0 and -1 lanes.
func add16NEON(dst, a, b []Q16) uint64 {
	const lanes = 4
	rd, ra, rb := rawInt32(dst), rawInt32(a), rawInt32(b)
	count := archsimd.BroadcastInt32x4(0)

	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		x := archsimd.LoadInt32x4(ra[i:])
		y := archsimd.LoadInt32x4(rb[i:])
		s := x.AddSaturated(y)
		s.Store(rd[i:])
		// A lane saturated exactly when the wrapping sum differs from it.
		count = count.Sub(x.Add(y).NotEqual(s).ToInt32x4())
	}
	return vecLaneSum(count) + add16Scalar(dst[i:], a[i:], b[i:])
}

// sub16NEON subtracts four lanes per step with the saturating NEON difference.
func sub16NEON(dst, a, b []Q16) uint64 {
	const lanes = 4
	rd, ra, rb := rawInt32(dst), rawInt32(a), rawInt32(b)
	count := archsimd.BroadcastInt32x4(0)

	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		x := archsimd.LoadInt32x4(ra[i:])
		y := archsimd.LoadInt32x4(rb[i:])
		s := x.SubSaturated(y)
		s.Store(rd[i:])
		count = count.Sub(x.Sub(y).NotEqual(s).ToInt32x4())
	}
	return vecLaneSum(count) + sub16Scalar(dst[i:], a[i:], b[i:])
}

// clamp16NEON needs no overflow work: a clamp cannot leave the int32 range.
func clamp16NEON(dst, a []Q16, lo, hi Q16) {
	const lanes = 4
	rd, ra := rawInt32(dst), rawInt32(a)
	loV := archsimd.BroadcastInt32x4(lo.raw)
	hiV := archsimd.BroadcastInt32x4(hi.raw)
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		archsimd.LoadInt32x4(ra[i:]).Max(loV).Min(hiV).Store(rd[i:])
	}
	clamp16Scalar(dst[i:], a[i:], lo, hi)
}
