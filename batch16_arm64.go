//go:build goexperiment.simd && arm64

package fixed

import (
	"simd/archsimd"
	"unsafe"
)

// NEON is mandatory on ARMv8, so every kernel here runs on every arm64 CPU.
// There is no tier to choose.

// selectKernels returns the kernels for this CPU.
//
// Mul stays scalar here: NEON widens only the low half of a vector
// (MulWidenHi has no counterpart), and no arm64 machine measured this probe.
func selectKernels() batchKernels {
	return batchKernels{add: add16NEON, mul: mul16Scalar}
}

// rawInt32 reinterprets a Q16 slice as int32. Q16 is struct{raw int32}, so the
// layout is identical.
func rawInt32(s []Q16) []int32 {
	return unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
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

	var events uint64
	for lane := range lanes {
		events += uint64(count.GetElem(uint8(lane)))
	}
	return events + add16Scalar(dst[i:], a[i:], b[i:])
}
