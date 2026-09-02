//go:build goexperiment.simd && go1.27 && arm64

package fixed

import (
	"simd/archsimd"
)

// NEON is mandatory on ARMv8, so every kernel here runs on every arm64 CPU.
// There is no tier to choose.

// selectKernels returns the kernels for this CPU.
func selectKernels() batchKernels {
	return batchKernels{
		path:       "neon",
		add:        add16NEON,
		sub:        sub16NEON,
		mul:        mul16NEON,
		clamp:      clamp16NEON,
		q32FromQ16: q32FromQ16NEON,
		q16FromQ32: q16FromQ32NEON,
	}
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

// vecPackHalves merges two half-filled vectors, each carrying two results in
// its low lanes, into one vector of four. NEON narrows only two lanes at a
// time, so every 64-bit kernel ends here.
func vecPackHalves(lo, hi archsimd.Int32x4) archsimd.Int32x4 {
	// ConcatEven spreads the two halves to lanes 0 and 2; ConcatOdd does the
	// same for lanes 1 and 3; the transpose then closes the four.
	return lo.ConcatEven(hi).InterleaveEven(lo.ConcatOdd(hi))
}

// vecNarrowPairNEON takes two 64-bit halves already shifted down to the
// Q16.16 grid and returns their four elements saturated to Q16, plus the
// lanes that saturated as a vector of 0 and -1.
func vecNarrowPairNEON(lo, hi archsimd.Int64x2) (archsimd.Int32x4, archsimd.Int32x4) {
	maxV := archsimd.BroadcastInt64x2(q16RawMax)
	minV := archsimd.BroadcastInt64x2(q16RawMin)
	loOvf := maxV.Less(lo).Or(lo.Less(minV)).ToInt64x2()
	hiOvf := maxV.Less(hi).Or(hi.Less(minV)).ToInt64x2()

	// A saturating narrow is one instruction, and it clamps to exactly the
	// Q16 range. The overflow flags narrow through the same path: -1 and 0
	// both survive it unchanged.
	res := vecPackHalves(lo.SaturateToInt32(), hi.SaturateToInt32())
	ovf := vecPackHalves(loOvf.SaturateToInt32(), hiOvf.SaturateToInt32())
	return res, ovf
}

// mul16NEON multiplies four lanes per step. NEON widens only the low two
// lanes, so HiToLo brings the other two down for a second pass.
func mul16NEON(dst, a, b []Q16) uint64 {
	const lanes = 4
	rd, ra, rb := rawInt32(dst), rawInt32(a), rawInt32(b)
	count := archsimd.BroadcastInt32x4(0)

	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		x := archsimd.LoadInt32x4(ra[i:])
		y := archsimd.LoadInt32x4(rb[i:])
		lo := x.MulWidenLo(y).ShiftAllRight(16)
		hi := x.HiToLo().MulWidenLo(y.HiToLo()).ShiftAllRight(16)

		r, o := vecNarrowPairNEON(lo, hi)
		r.Store(rd[i:])
		count = count.Sub(o)
	}
	return vecLaneSum(count) + mul16Scalar(dst[i:], a[i:], b[i:])
}

// q32FromQ16NEON widens four elements per step. The sign extension and the
// shift are one instruction each; nothing can saturate.
func q32FromQ16NEON(dst []Q32, a []Q16) {
	const lanes = 4
	rd, ra := rawInt64(dst), rawInt32(a)
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		x := archsimd.LoadInt32x4(ra[i:])
		x.ExtendLo2ToInt64().ShiftAllLeft(16).Store(rd[i:])
		x.HiToLo().ExtendLo2ToInt64().ShiftAllLeft(16).Store(rd[i+2:])
	}
	q32FromQ16Scalar(dst[i:], a[i:])
}

// q16FromQ32NEON narrows four elements per step. A Q32 raw value has the same
// shape as a Q16 product, so the narrowing shares the multiply's tail.
func q16FromQ32NEON(dst []Q16, a []Q32) uint64 {
	const lanes = 4
	rd, ra := rawInt32(dst), rawInt64(a)
	count := archsimd.BroadcastInt32x4(0)

	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		lo := archsimd.LoadInt64x2(ra[i:]).ShiftAllRight(16)
		hi := archsimd.LoadInt64x2(ra[i+2:]).ShiftAllRight(16)

		r, o := vecNarrowPairNEON(lo, hi)
		r.Store(rd[i:])
		count = count.Sub(o)
	}
	return vecLaneSum(count) + q16FromQ32Scalar(dst[i:], a[i:])
}
