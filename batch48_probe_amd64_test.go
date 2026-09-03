//go:build goexperiment.simd && go1.27 && amd64

package fixed

import (
	"math/bits"
	"simd/archsimd"
)

func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	mulAcc48Kernels = append(mulAcc48Kernels,
		mulAcc48Kernel{"avx2", mulAcc48AVX2},
		mulAcc48Kernel{"avx2gate", mulAcc48AVX2Gate},
		mulAcc48Kernel{"avx2wrap", mulAcc48AVX2Wrap})
	widen48Kernels = append(widen48Kernels, widen48Kernel{"avx2", q48FromQ16AVX2})
	narrow48Kernels = append(narrow48Kernels, narrow48Kernel{"avx2", q16FromQ48AVX2})
}

func mulAcc48AVX2(acc []Q48, a, b []Q16) uint64 {
	const lanes = 8
	racc, ra, rb := rawInt64Q48(acc), rawInt32(a), rawInt32(b)
	order := archsimd.LoadUint32x8Array(&q48SplitOrder)
	var events uint64
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		lo, hi := vecProducts48(archsimd.LoadInt32x8(ra[i:]), archsimd.LoadInt32x8(rb[i:]), order)
		r0, e0 := vecAddSat64(archsimd.LoadInt64x4(racc[i:]), lo)
		r1, e1 := vecAddSat64(archsimd.LoadInt64x4(racc[i+4:]), hi)
		r0.Store(racc[i:])
		r1.Store(racc[i+4:])
		events += e0 + e1
	}
	return events + mulAcc48Scalar(acc[i:], a[i:], b[i:])
}

// vecNearEdge reports whether any lane of x or y is outside (-2^62, 2^62).
// Such a lane has bits 63 and 62 different, so v^(v<<1) carries the answer
// in its sign, and one Or folds both vectors into one comparison.
func vecNearEdge(x, y archsimd.Int64x4) bool {
	e := x.Xor(x.ShiftAllLeft(1)).Or(y.Xor(y.ShiftAllLeft(1)))
	return archsimd.BroadcastInt64x4(0).Greater(e).ToBits() != 0
}

// mulAcc48AVX2Gate adds without saturation when no lane can overflow. A
// product after the shift fits 47 bits, so a lane inside (-2^62, 2^62) is
// safe. A block near the edge falls back to the saturating scalar path.
func mulAcc48AVX2Gate(acc []Q48, a, b []Q16) uint64 {
	const lanes = 8
	racc, ra, rb := rawInt64Q48(acc), rawInt32(a), rawInt32(b)
	order := archsimd.LoadUint32x8Array(&q48SplitOrder)
	var events uint64
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		acc0, acc1 := archsimd.LoadInt64x4(racc[i:]), archsimd.LoadInt64x4(racc[i+4:])
		if vecNearEdge(acc0, acc1) {
			events += mulAcc48Scalar(acc[i:i+lanes], a[i:i+lanes], b[i:i+lanes])
			continue
		}
		lo, hi := vecProducts48(archsimd.LoadInt32x8(ra[i:]), archsimd.LoadInt32x8(rb[i:]), order)
		acc0.Add(lo).Store(racc[i:])
		acc1.Add(hi).Store(racc[i+4:])
	}
	return events + mulAcc48Scalar(acc[i:], a[i:], b[i:])
}

// mulAcc48AVX2Wrap is the same kernel without saturation. It is not a
// candidate; it measures what the saturation costs in the vector path.
func mulAcc48AVX2Wrap(acc []Q48, a, b []Q16) uint64 {
	const lanes = 8
	racc, ra, rb := rawInt64Q48(acc), rawInt32(a), rawInt32(b)
	order := archsimd.LoadUint32x8Array(&q48SplitOrder)
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		lo, hi := vecProducts48(archsimd.LoadInt32x8(ra[i:]), archsimd.LoadInt32x8(rb[i:]), order)
		archsimd.LoadInt64x4(racc[i:]).Add(lo).Store(racc[i:])
		archsimd.LoadInt64x4(racc[i+4:]).Add(hi).Store(racc[i+4:])
	}
	return mulAcc48Scalar(acc[i:], a[i:], b[i:])
}

// q48FromQ16AVX2 is q32FromQ16AVX2 without the shift.
func q48FromQ16AVX2(dst []Q48, a []Q16) {
	const lanes = 8
	rd, ra := rawInt64Q48(dst), rawInt32(a)
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		x := archsimd.LoadInt32x8(ra[i:])
		x.GetLo().ExtendToInt64().Store(rd[i:])
		x.GetHi().ExtendToInt64().Store(rd[i+4:])
	}
	q48FromQ16Scalar(dst[i:], a[i:])
}

// vecClamp48ToQ16 clamps 64-bit lanes to the Q16 range and returns the lane
// count that moved. A Q48 raw value can hold garbage above bit 47, so the
// check must read all 64 bits; two native VPCMPGTQ do that.
func vecClamp48ToQ16(x archsimd.Int64x4) (archsimd.Uint64x4, uint64) {
	maxV := archsimd.BroadcastInt64x4(q16RawMax)
	minV := archsimd.BroadcastInt64x4(q16RawMin)
	over := x.Greater(maxV)
	under := minV.Greater(x)
	x = maxV.IfElse(over, x)
	x = minV.IfElse(under, x)
	return x.AsUint64x4(), uint64(bits.OnesCount8(over.Or(under).ToBits()))
}

// q16FromQ48AVX2 clamps in 64 bits, then packs through vecNarrowPair, which
// sees only in-range lanes and reports nothing.
func q16FromQ48AVX2(dst []Q16, a []Q48) uint64 {
	const lanes = 8
	rd, ra := rawInt32(dst), rawInt64Q48(a)
	order := archsimd.LoadUint32x8Array(&q16NarrowOrder)
	var events uint64
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		te, e0 := vecClamp48ToQ16(archsimd.LoadInt64x4(ra[i:]))
		to, e1 := vecClamp48ToQ16(archsimd.LoadInt64x4(ra[i+4:]))
		r, _ := vecNarrowPair(te, to)
		r.Permute(order).Store(rd[i:])
		events += e0 + e1
	}
	return events + q16FromQ48Scalar(dst[i:], a[i:])
}
