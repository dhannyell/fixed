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
	}
	return k
}

// rawInt32 reinterprets a Q16 slice as int32. Q16 is struct{raw int32}, so the
// layout is identical.
func rawInt32(s []Q16) []int32 {
	return unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
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

// vecNarrowQ16 shifts four 64-bit products down to Q16.16 and clamps them to
// the int32 range. It returns the clamped values and the saturated lane count.
func vecNarrowQ16(p archsimd.Int64x4) (archsimd.Int64x4, uint64) {
	// x86 has no arithmetic 64-bit shift below AVX-512. The identity
	// sra(v,n) == (srl(v,n) ^ k) - k, with k = 1<<(63-n), rebuilds the sign.
	k := archsimd.BroadcastInt64x4(int64(1) << 47)
	v := p.AsUint64x4().ShiftAllRight(16).AsInt64x4().Xor(k).Sub(k)

	maxV := archsimd.BroadcastInt64x4(q16RawMax)
	minV := archsimd.BroadcastInt64x4(q16RawMin)
	high, low := maxV.Less(v), v.Less(minV)
	v = maxV.IfElse(high, v)
	v = minV.IfElse(low, v)
	return v, uint64(bits.OnesCount8(high.Or(low).ToBits()))
}

// vecMulSat multiplies two vectors as Q16.16 with saturation and returns the
// saturated lane count. It requires AVX2.
func vecMulSat(x, y archsimd.Int32x8) (archsimd.Int32x8, uint64) {
	// MulWidenEven reads lanes 0, 2, 4 and 6, so the odd lanes move down by
	// 32 bits to take their turn.
	xOdd := x.AsUint64x4().ShiftAllRight(32).AsInt32x8()
	yOdd := y.AsUint64x4().ShiftAllRight(32).AsInt32x8()
	even, evenEvents := vecNarrowQ16(x.MulWidenEven(y))
	odd, oddEvents := vecNarrowQ16(xOdd.MulWidenEven(yOdd))

	// Each clamped value fits int32. Clearing the sign extension of the even
	// lanes and lifting the odd lanes into place makes the two disjoint.
	lo := even.AsUint64x4().ShiftAllLeft(32).ShiftAllRight(32)
	hi := odd.AsUint64x4().ShiftAllLeft(32)
	return lo.Or(hi).AsInt32x8(), evenEvents + oddEvents
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
		r, e := vecMulSat(archsimd.LoadInt32x8(ra[i:]), archsimd.LoadInt32x8(rb[i:]))
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
