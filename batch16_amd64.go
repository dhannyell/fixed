//go:build goexperiment.simd && amd64

package fixed

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

// The compiler emits vector instructions without a feature guard, at every
// GOAMD64 level. The runtime check below is the only thing that keeps this
// build off a CPU that lacks the instructions.

// selectKernels returns the kernels for this CPU.
//
// Inputs: archsimd.X86.AVX2() and archsimd.X86.AVX512(), both bool.
// Available adds: add16AVX2 (256-bit) and add16Scalar. Mul stays scalar until
// its vector kernel lands. An AVX-512 tier is not written yet.
func selectKernels() batchKernels {
	// TODO(human): order the tiers and leave room for an AVX-512 step.
	return batchKernels{add: add16Scalar, mul: mul16Scalar}
}

// rawInt32 reinterprets a Q16 slice as int32. Q16 is struct{raw int32}, so the
// layout is identical.
func rawInt32(s []Q16) []int32 {
	return unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
}

// add16AVX2 adds eight lanes per step. x86 has no saturating 32-bit add, so
// overflow is detected from the sign of (a^r)&(b^r) and blended in.
// It requires AVX2.
func add16AVX2(dst, a, b []Q16) uint64 {
	const lanes = 8
	rd, ra, rb := rawInt32(dst), rawInt32(a), rawInt32(b)
	zero := archsimd.BroadcastInt32x8(0)
	maxV := archsimd.BroadcastInt32x8(q16RawMax)

	var events uint64
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		x := archsimd.LoadInt32x8(ra[i:])
		y := archsimd.LoadInt32x8(rb[i:])
		r := x.Add(y)
		ovf := x.Xor(r).And(y.Xor(r)).Less(zero)
		// Arithmetic shift by 31 gives 0 for x>=0 and -1 for x<0, so the xor
		// picks q16RawMax on the high side and q16RawMin on the low side.
		sat := x.ShiftAllRight(31).Xor(maxV)
		sat.IfElse(ovf, r).Store(rd[i:])
		events += uint64(bits.OnesCount8(ovf.ToBits()))
	}
	return events + add16Scalar(dst[i:], a[i:], b[i:])
}
