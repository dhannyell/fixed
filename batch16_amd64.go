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
func selectKernels() batchKernels {
	k := batchKernels{
		path:       "scalar",
		add:        add16Scalar,
		sub:        sub16Scalar,
		mul:        mul16Scalar,
		clamp:      clamp16Scalar,
		q32FromQ16: q32FromQ16Scalar,
		q16FromQ32: q16FromQ32Scalar,
	}
	// Tiers widen downward. An AVX-512 branch belongs above this one, guarded
	// by archsimd.X86.AVX512(), once a machine can measure it.
	if archsimd.X86.AVX2() {
		k.path = "avx2"
		k.add = add16AVX2
		k.sub = sub16AVX2
		k.mul = mul16AVX2
		k.clamp = clamp16AVX2
		k.q32FromQ16 = q32FromQ16AVX2
		k.q16FromQ32 = q16FromQ32AVX2
	}
	return k
}

// rawInt32 reinterprets a Q16 slice as int32. Q16 is struct{raw int32}, so the
// layout is identical.
func rawInt32(s []Q16) []int32 {
	return unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
}

// rawInt64 reinterprets a Q32 slice as int64. Q32 is struct{raw int64}, so the
// layout is identical.
func rawInt64(s []Q32) []int64 {
	return unsafe.Slice((*int64)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
}

// vecAddSat adds two vectors with Q16 saturation and returns the saturated
// lane count. x86 has no saturating 32-bit add, so overflow is read from the
// sign of (x^r)&(y^r). It requires AVX2.
func vecAddSat(x, y archsimd.Int32x8) (archsimd.Int32x8, uint64) {
	r := x.Add(y)
	ovf := x.Xor(r).And(y.Xor(r)).Less(archsimd.BroadcastInt32x8(0))
	// Shifting by 31 gives 0 for x>=0 and -1 for x<0, so the xor picks
	// q16RawMax on the high side and q16RawMin on the low side.
	sat := x.ShiftAllRight(31).Xor(archsimd.BroadcastInt32x8(q16RawMax))
	return sat.IfElse(ovf, r), uint64(bits.OnesCount8(ovf.ToBits()))
}

// vecSubSat subtracts two vectors with Q16 saturation and returns the
// saturated lane count. A difference overflows when the operands disagree in
// sign and the result takes the sign of y. It requires AVX2.
func vecSubSat(x, y archsimd.Int32x8) (archsimd.Int32x8, uint64) {
	r := x.Sub(y)
	ovf := x.Xor(y).And(x.Xor(r)).Less(archsimd.BroadcastInt32x8(0))
	sat := x.ShiftAllRight(31).Xor(archsimd.BroadcastInt32x8(q16RawMax))
	return sat.IfElse(ovf, r), uint64(bits.OnesCount8(ovf.ToBits()))
}

// vecNarrowPair takes two 64-bit halves already shifted down to the Q16.16
// grid and returns their eight elements saturated to Q16, interleaved as
// te0, to0, te1, to1 and so on, with the saturated lane count.
//
// A value fits Q16 when bits 47 through 63 of the pre-shift product all agree.
// The low dword of each shifted lane carries the candidate and the high dword
// carries the top 16 bits, so the whole test runs in 32-bit lanes and x86
// needs no 64-bit arithmetic shift. The two halves interleave through shifts
// and one or, because a mask register is AVX-512.
func vecNarrowPair(te, to archsimd.Uint64x4) (archsimd.Int32x8, uint64) {
	cand := te.ShiftAllLeft(32).ShiftAllRight(32).Or(to.ShiftAllLeft(32)).AsInt32x8()
	high := te.ShiftAllRight(32).Or(to.ShiftAllRight(32).ShiftAllLeft(32)).AsInt32x8()

	// The top 16 bits arrive zero extended; the shift pair restores the sign.
	sext := high.ShiftAllLeft(16).ShiftAllRight(16)
	ovf := sext.NotEqual(cand.ShiftAllRight(31))
	// Shifting by 31 gives 0 for a positive product and -1 for a negative one,
	// so the xor picks q16RawMax on the high side and q16RawMin on the low.
	sat := sext.ShiftAllRight(31).Xor(archsimd.BroadcastInt32x8(q16RawMax))
	return sat.IfElse(ovf, cand), uint64(bits.OnesCount8(ovf.ToBits()))
}

func add16AVX2(dst, a, b []Q16) uint64 {
	const lanes = 8
	rd, ra, rb := rawInt32(dst), rawInt32(a), rawInt32(b)
	var events uint64
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		r, e := vecAddSat(archsimd.LoadInt32x8(ra[i:]), archsimd.LoadInt32x8(rb[i:]))
		r.Store(rd[i:])
		events += e
	}
	return events + add16Scalar(dst[i:], a[i:], b[i:])
}

func sub16AVX2(dst, a, b []Q16) uint64 {
	const lanes = 8
	rd, ra, rb := rawInt32(dst), rawInt32(a), rawInt32(b)
	var events uint64
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		r, e := vecSubSat(archsimd.LoadInt32x8(ra[i:]), archsimd.LoadInt32x8(rb[i:]))
		r.Store(rd[i:])
		events += e
	}
	return events + sub16Scalar(dst[i:], a[i:], b[i:])
}

func mul16AVX2(dst, a, b []Q16) uint64 {
	const lanes = 8
	rd, ra, rb := rawInt32(dst), rawInt32(a), rawInt32(b)
	var events uint64
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		x := archsimd.LoadInt32x8(ra[i:])
		y := archsimd.LoadInt32x8(rb[i:])
		// MulWidenEven reads lanes 0, 2, 4 and 6, so the odd lanes move down
		// by 32 bits to take their turn.
		xOdd := x.AsUint64x4().ShiftAllRight(32).AsInt32x8()
		yOdd := y.AsUint64x4().ShiftAllRight(32).AsInt32x8()
		te := x.MulWidenEven(y).AsUint64x4().ShiftAllRight(16)
		to := xOdd.MulWidenEven(yOdd).AsUint64x4().ShiftAllRight(16)

		r, e := vecNarrowPair(te, to)
		r.Store(rd[i:])
		events += e
	}
	return events + mul16Scalar(dst[i:], a[i:], b[i:])
}

// clamp16AVX2 needs no overflow work: a clamp cannot leave the int32 range.
func clamp16AVX2(dst, a []Q16, lo, hi Q16) {
	const lanes = 8
	rd, ra := rawInt32(dst), rawInt32(a)
	loV := archsimd.BroadcastInt32x8(lo.raw)
	hiV := archsimd.BroadcastInt32x8(hi.raw)
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		archsimd.LoadInt32x8(ra[i:]).Max(loV).Min(hiV).Store(rd[i:])
	}
	clamp16Scalar(dst[i:], a[i:], lo, hi)
}

// q16NarrowOrder undoes the interleave of vecNarrowPair. Element j of the
// result reads lane q16NarrowOrder[j] of the interleaved vector.
var q16NarrowOrder = [8]uint32{0, 2, 4, 6, 1, 3, 5, 7}

// q32FromQ16AVX2 widens eight elements per step. The sign extension and the
// shift are one instruction each; nothing can saturate.
func q32FromQ16AVX2(dst []Q32, a []Q16) {
	const lanes = 8
	rd, ra := rawInt64(dst), rawInt32(a)
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		x := archsimd.LoadInt32x8(ra[i:])
		x.GetLo().ExtendToInt64().ShiftAllLeft(16).Store(rd[i:])
		x.GetHi().ExtendToInt64().ShiftAllLeft(16).Store(rd[i+4:])
	}
	q32FromQ16Scalar(dst[i:], a[i:])
}

// q16FromQ32AVX2 narrows eight elements per step. A Q32 raw value has the same
// shape as a Q16 product, so the narrowing shares the multiply's tail.
func q16FromQ32AVX2(dst []Q16, a []Q32) uint64 {
	const lanes = 8
	rd, ra := rawInt32(dst), rawInt64(a)
	order := archsimd.LoadUint32x8Array(&q16NarrowOrder)
	var events uint64
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		te := archsimd.LoadInt64x4(ra[i:]).AsUint64x4().ShiftAllRight(16)
		to := archsimd.LoadInt64x4(ra[i+4:]).AsUint64x4().ShiftAllRight(16)
		r, e := vecNarrowPair(te, to)
		r.Permute(order).Store(rd[i:])
		events += e
	}
	return events + q16FromQ32Scalar(dst[i:], a[i:])
}
