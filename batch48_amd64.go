//go:build goexperiment.simd && go1.27 && amd64

package fixed

import (
	"math/bits"
	"simd/archsimd"
)

// AVX2 has no 64-bit arithmetic shift and no 64x64 multiply. The kernels
// below build both from 32-bit halves, which is exact because every product
// here has one 32-bit operand.

// q48SplitOrder moves elements 0..3 into the even lanes and 4..7 into the odd
// lanes, so MulWidenEven yields products 0..3 and the odd pass yields 4..7 in
// the memory order of a Q48 slice.
var q48SplitOrder = [8]uint32{0, 4, 1, 5, 2, 6, 3, 7}

// vecSra16 is an arithmetic right shift by 16 on 64-bit lanes. The logical
// shift leaves the sign at bit 47, and xor-subtract of 2^47 extends it.
func vecSra16(x archsimd.Int64x4) archsimd.Int64x4 {
	bias := archsimd.BroadcastInt64x4(1 << 47)
	return x.AsUint64x4().ShiftAllRight(16).AsInt64x4().Xor(bias).Sub(bias)
}

// vecSra32 is an arithmetic right shift by 32 on 64-bit lanes.
func vecSra32(x archsimd.Int64x4) archsimd.Int64x4 {
	bias := archsimd.BroadcastInt64x4(1 << 31)
	return x.AsUint64x4().ShiftAllRight(32).AsInt64x4().Xor(bias).Sub(bias)
}

// vecProducts48 returns the eight Q48 products of one block: lanes for
// elements 0..3 and 4..7.
func vecProducts48(x, y archsimd.Int32x8, order archsimd.Uint32x8) (lo, hi archsimd.Int64x4) {
	x = x.Permute(order)
	y = y.Permute(order)
	xOdd := x.AsUint64x4().ShiftAllRight(32).AsInt32x8()
	yOdd := y.AsUint64x4().ShiftAllRight(32).AsInt32x8()
	lo = vecSra16(x.MulWidenEven(y))
	hi = vecSra16(xOdd.MulWidenEven(yOdd))
	return lo, hi
}

// vecAddSat64 adds with Q48 saturation and returns the saturated lane count.
// Overflow is read from the sign of (x^r)&(y^r); the saturated value is Max
// for a non-negative x and Min for a negative x.
func vecAddSat64(x, y archsimd.Int64x4) (archsimd.Int64x4, uint64) {
	zero := archsimd.BroadcastInt64x4(0)
	r := x.Add(y)
	ovf := zero.Greater(x.Xor(r).And(y.Xor(r)))
	sat := archsimd.BroadcastInt64x4(q48RawMin).IfElse(zero.Greater(x), archsimd.BroadcastInt64x4(q48RawMax))
	return sat.IfElse(ovf, r), uint64(bits.OnesCount8(ovf.ToBits()))
}

// dot16AVX2 keeps eight lane partials, which is exactly the canonical order
// of dot16Scalar: element i joins partial i mod 8. The tail and the tree
// reduction run in scalar code on the stored partials.
func dot16AVX2(a, b []Q16) (Q48, uint64) {
	const lanes = 8
	ra, rb := rawInt32(a), rawInt32(b)
	order := archsimd.LoadUint32x8Array(&q48SplitOrder)
	p0 := archsimd.BroadcastInt64x4(0)
	p1 := archsimd.BroadcastInt64x4(0)
	var events uint64
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		lo, hi := vecProducts48(archsimd.LoadInt32x8(ra[i:]), archsimd.LoadInt32x8(rb[i:]), order)
		var e0, e1 uint64
		p0, e0 = vecAddSat64(p0, lo)
		p1, e1 = vecAddSat64(p1, hi)
		events += e0 + e1
	}
	var partial [dot16Lanes]int64
	p0.Store(partial[:4])
	p1.Store(partial[4:])
	for j := i; j < len(a); j++ {
		p := (int64(a[j].raw) * int64(b[j].raw)) >> 16
		r, ovf := q48AddSat(partial[j%dot16Lanes], p)
		if ovf {
			events++
		}
		partial[j%dot16Lanes] = r
	}
	return dot16Reduce(partial, &events), events
}

// vecMul16Parts computes q*f/2^16 from two 32x32 partials. r is the result
// modulo 2^64. t is floor(q*f/2^32): the result fits when t lies in
// [-2^47, 2^47), because the remaining 16 low bits are non-negative.
// f holds the Q16 operands in the even 32-bit lanes.
func vecMul16Parts(q archsimd.Int64x4, f archsimd.Int32x8) (r, t archsimd.Int64x4) {
	lo := q.AsInt32x8()
	hi := q.AsUint64x4().ShiftAllRight(32).AsInt32x8()
	// The low word is unsigned; a signed multiply is off by 2^32*f when its
	// sign bit is set. The mask is -1 there, so f joins the high partial by Sub.
	carry := lo.ShiftAllRight(31)
	hiP := hi.MulWidenEven(f).Sub(carry.MulWidenEven(f))
	loP := lo.MulWidenEven(f)
	r = hiP.ShiftAllLeft(16).Add(vecSra16(loP))
	t = hiP.Add(vecSra32(loP))
	return r, t
}

// vecOutside47 reports lanes where t is outside [-2^47, 2^47).
func vecOutside47(t archsimd.Int64x4) archsimd.Mask64x4 {
	biased := t.Add(archsimd.BroadcastInt64x4(1 << 47)).AsUint64x4().ShiftAllRight(48).AsInt64x4()
	return biased.Greater(archsimd.BroadcastInt64x4(0))
}

// q48Mul16AVX2 stores the wrapped result when no lane saturates. A block
// with a saturating lane runs in scalar, which also counts it. The test is
// exact, so the scalar path runs only where saturation happens.
func q48Mul16AVX2(dst, q []Q48, f []Q16) uint64 {
	const lanes = 8
	rd, rq, rf := rawInt64Q48(dst), rawInt64Q48(q), rawInt32(f)
	order := archsimd.LoadUint32x8Array(&q48SplitOrder)
	var events uint64
	i := 0
	for ; i+lanes <= len(rq); i += lanes {
		x := archsimd.LoadInt32x8(rf[i:]).Permute(order)
		xOdd := x.AsUint64x4().ShiftAllRight(32).AsInt32x8()
		r0, t0 := vecMul16Parts(archsimd.LoadInt64x4(rq[i:]), x)
		r1, t1 := vecMul16Parts(archsimd.LoadInt64x4(rq[i+4:]), xOdd)
		if vecOutside47(t0).Or(vecOutside47(t1)).ToBits() != 0 {
			events += q48Mul16Scalar(dst[i:i+lanes], q[i:i+lanes], f[i:i+lanes])
			continue
		}
		r0.Store(rd[i:])
		r1.Store(rd[i+4:])
	}
	return events + q48Mul16Scalar(dst[i:], q[i:], f[i:])
}
