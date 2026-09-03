//go:build goexperiment.simd && go1.27 && arm64

package fixed

import "simd/archsimd"

// NEON candidates for the Q48 batch kernels. They register into the same
// kernel lists as the AVX2 kernels, so the parity tests and BenchmarkBatch
// cover them on an arm64 host. They become the arm64 kernels only after the
// measurement passes.

func init() {
	dot16Kernels = append(dot16Kernels, dot16Kernel{"neon", dot16NEON})
	q48Mul16Kernels = append(q48Mul16Kernels,
		q48Mul16Kernel{"neon", q48Mul16NEON},
		q48Mul16Kernel{"neongate", q48Mul16NEONGate})
	mulAcc48Kernels = append(mulAcc48Kernels, mulAcc48Kernel{"neon", mulAcc48NEON})
}

// vecLaneSum64 adds the two lanes of a 64-bit counter vector.
func vecLaneSum64(count archsimd.Int64x2) uint64 {
	return uint64(count.GetElem(0)) + uint64(count.GetElem(1))
}

// vecProducts48NEON returns the four Q48 products of x and y: lanes for
// elements 0..1 and 2..3. SMULL widens the low two lanes; HiToLo brings the
// other two down. SSHR is a native 64-bit arithmetic shift.
func vecProducts48NEON(x, y archsimd.Int32x4) (lo, hi archsimd.Int64x2) {
	lo = x.MulWidenLo(y).ShiftAllRight(16)
	hi = x.HiToLo().MulWidenLo(y.HiToLo()).ShiftAllRight(16)
	return lo, hi
}

// vecAddSat64NEON adds with saturation in one SQADD and returns the lanes
// that saturated as a vector of 0 and -1.
func vecAddSat64NEON(x, y archsimd.Int64x2) (archsimd.Int64x2, archsimd.Int64x2) {
	s := x.AddSaturated(y)
	return s, x.Add(y).NotEqual(s).ToInt64x2()
}

// dot16NEON keeps eight partials in four vectors: element i joins partial
// i mod 8, which is the canonical order of dot16Scalar.
func dot16NEON(a, b []Q16) (Q48, uint64) {
	const lanes = 8
	ra, rb := rawInt32(a), rawInt32(b)
	zero := archsimd.BroadcastInt64x2(0)
	p0, p1, p2, p3 := zero, zero, zero, zero
	count := zero
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		lo, hi := vecProducts48NEON(archsimd.LoadInt32x4(ra[i:]), archsimd.LoadInt32x4(rb[i:]))
		var c0, c1, c2, c3 archsimd.Int64x2
		p0, c0 = vecAddSat64NEON(p0, lo)
		p1, c1 = vecAddSat64NEON(p1, hi)
		lo, hi = vecProducts48NEON(archsimd.LoadInt32x4(ra[i+4:]), archsimd.LoadInt32x4(rb[i+4:]))
		p2, c2 = vecAddSat64NEON(p2, lo)
		p3, c3 = vecAddSat64NEON(p3, hi)
		count = count.Sub(c0.Add(c1).Add(c2).Add(c3))
	}
	var partial [dot16Lanes]int64
	p0.Store(partial[0:2])
	p1.Store(partial[2:4])
	p2.Store(partial[4:6])
	p3.Store(partial[6:8])
	events := vecLaneSum64(count)
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

// vecMul16PartsNEON is vecMul16Parts on two lanes. ConcatEven and ConcatOdd
// gather the low and high words of both Q48 lanes into the low two 32-bit
// lanes, where SMULL reads them. f holds the factors in its low two lanes.
// arm64 archsimd reinterprets lanes only through the unsigned types.
func vecMul16PartsNEON(q archsimd.Int64x2, f archsimd.Int32x4) (r, t archsimd.Int64x2) {
	w := q.ConvertToUint64().ReshapeToUint32s().BitsToInt32()
	lo := w.ConcatEven(w)
	hi := w.ConcatOdd(w)
	carry := lo.ShiftAllRight(31)
	hiP := hi.MulWidenLo(f).Sub(carry.MulWidenLo(f))
	loP := lo.MulWidenLo(f)
	r = hiP.ShiftAllLeft(16).Add(loP.ShiftAllRight(16))
	t = hiP.Add(loP.ShiftAllRight(32))
	return r, t
}

// vecOutside47NEON returns the lanes where t is outside [-2^47, 2^47) as a
// vector of 0 and -1.
func vecOutside47NEON(t archsimd.Int64x2) archsimd.Int64x2 {
	biased := t.Add(archsimd.BroadcastInt64x2(1 << 47)).ConvertToUint64().ShiftAllRight(48).BitsToInt64()
	return biased.Greater(archsimd.BroadcastInt64x2(0)).ToInt64x2()
}

// vecMul16SatNEON saturates r where t says the product does not fit and
// returns the saturated lanes as 0 and -1.
func vecMul16SatNEON(r, t archsimd.Int64x2) (archsimd.Int64x2, archsimd.Int64x2) {
	zero := archsimd.BroadcastInt64x2(0)
	ovf := vecOutside47NEON(t)
	sat := archsimd.BroadcastInt64x2(q48RawMin).IfElse(zero.Greater(t), archsimd.BroadcastInt64x2(q48RawMax))
	return sat.IfElse(ovf.NotEqual(zero), r), ovf
}

func q48Mul16NEON(dst, q []Q48, f []Q16) uint64 {
	const lanes = 4
	rd, rq, rf := rawInt64Q48(dst), rawInt64Q48(q), rawInt32(f)
	count := archsimd.BroadcastInt64x2(0)
	i := 0
	for ; i+lanes <= len(rq); i += lanes {
		x := archsimd.LoadInt32x4(rf[i:])
		r0, t0 := vecMul16PartsNEON(archsimd.LoadInt64x2(rq[i:]), x)
		r1, t1 := vecMul16PartsNEON(archsimd.LoadInt64x2(rq[i+2:]), x.HiToLo())
		r0, c0 := vecMul16SatNEON(r0, t0)
		r1, c1 := vecMul16SatNEON(r1, t1)
		r0.Store(rd[i:])
		r1.Store(rd[i+2:])
		count = count.Sub(c0.Add(c1))
	}
	return vecLaneSum64(count) + q48Mul16Scalar(dst[i:], q[i:], f[i:])
}

// q48Mul16NEONGate stores the wrapped result when no lane saturates. NEON
// has no lane mask to a register, so the block test folds the two flag
// vectors and reads both lanes.
func q48Mul16NEONGate(dst, q []Q48, f []Q16) uint64 {
	const lanes = 4
	rd, rq, rf := rawInt64Q48(dst), rawInt64Q48(q), rawInt32(f)
	var events uint64
	i := 0
	for ; i+lanes <= len(rq); i += lanes {
		x := archsimd.LoadInt32x4(rf[i:])
		r0, t0 := vecMul16PartsNEON(archsimd.LoadInt64x2(rq[i:]), x)
		r1, t1 := vecMul16PartsNEON(archsimd.LoadInt64x2(rq[i+2:]), x.HiToLo())
		flags := vecOutside47NEON(t0).Or(vecOutside47NEON(t1))
		if flags.GetElem(0)|flags.GetElem(1) != 0 {
			events += q48Mul16Scalar(dst[i:i+lanes], q[i:i+lanes], f[i:i+lanes])
			continue
		}
		r0.Store(rd[i:])
		r1.Store(rd[i+2:])
	}
	return events + q48Mul16Scalar(dst[i:], q[i:], f[i:])
}

// mulAcc48NEON is the per-element accumulation with SQADD. It has no gate:
// the saturating add is one instruction here.
func mulAcc48NEON(acc []Q48, a, b []Q16) uint64 {
	const lanes = 4
	racc, ra, rb := rawInt64Q48(acc), rawInt32(a), rawInt32(b)
	count := archsimd.BroadcastInt64x2(0)
	i := 0
	for ; i+lanes <= len(ra); i += lanes {
		lo, hi := vecProducts48NEON(archsimd.LoadInt32x4(ra[i:]), archsimd.LoadInt32x4(rb[i:]))
		r0, c0 := vecAddSat64NEON(archsimd.LoadInt64x2(racc[i:]), lo)
		r1, c1 := vecAddSat64NEON(archsimd.LoadInt64x2(racc[i+2:]), hi)
		r0.Store(racc[i:])
		r1.Store(racc[i+2:])
		count = count.Sub(c0.Add(c1))
	}
	return vecLaneSum64(count) + mulAcc48Scalar(acc[i:], a[i:], b[i:])
}
