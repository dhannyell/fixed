//go:build goexperiment.simd && go1.27 && amd64

package fixed

import (
	"math/bits"
	"simd/archsimd"
)

// LaneWidth is the number of values in an amd64 SIMD lane type.
const LaneWidth = 8

type lane16Data = archsimd.Int32x8
type mask16Data = archsimd.Mask32x8

type lane48Data struct {
	lo archsimd.Int64x4
	hi archsimd.Int64x4
}

func lanesAvailable() bool { return archsimd.X86.AVX2() }

func lanePath() string { return "avx2" }

func recordLaneSaturations(events uint64) {
	if events != 0 {
		saturationEvents.Add(events)
	}
}

func splatLane16(q Q16) Lane16 {
	return Lane16{archsimd.BroadcastInt32x8(q.raw)}
}

func loadLane16(p *[LaneWidth]Q16) Lane16 {
	return Lane16{archsimd.LoadInt32x8(rawInt32(p[:]))}
}

func storeLane16(a Lane16, p *[LaneWidth]Q16) {
	a.v.Store(rawInt32(p[:]))
}

func addLane16(a, b Lane16) Lane16 {
	r, events := vecAddSat(a.v, b.v)
	recordLaneSaturations(events)
	return Lane16{r}
}

func subLane16(a, b Lane16) Lane16 {
	r, events := vecSubSat(a.v, b.v)
	recordLaneSaturations(events)
	return Lane16{r}
}

func mulLane16(a, b Lane16) Lane16 {
	x, y := a.v, b.v
	// MulWidenEven reads the even lanes; shifting each 64-bit pair exposes
	// the odd lanes for the second pass.
	aOdd := x.AsUint64x4().ShiftAllRight(32).AsInt32x8()
	bOdd := y.AsUint64x4().ShiftAllRight(32).AsInt32x8()
	even := x.MulWidenEven(y).AsUint64x4().ShiftAllRight(16)
	odd := aOdd.MulWidenEven(bOdd).AsUint64x4().ShiftAllRight(16)
	r, events := vecNarrowPair(even, odd)
	recordLaneSaturations(events)
	return Lane16{r}
}

func minLane16(a, b Lane16) Lane16 {
	return Lane16{a.v.Min(b.v)}
}

func maxLane16(a, b Lane16) Lane16 {
	return Lane16{a.v.Max(b.v)}
}

func symClampLane16(a, limit Lane16) Lane16 {
	x, hi := a.v, limit.v
	neg, events := vecSubSat(archsimd.BroadcastInt32x8(0), hi)
	recordLaneSaturations(events)
	return Lane16{x.Min(hi).Max(neg)}
}

func greaterLane16(a, b Lane16) Mask16 {
	return Mask16{a.v.Greater(b.v)}
}

func equalsLane16(a, b Lane16) Mask16 {
	return Mask16{a.v.Equal(b.v)}
}

func orMask16(m, n Mask16) Mask16 {
	return Mask16{m.v.Or(n.v)}
}

func allZeroMask16(m Mask16) bool { return m.v.ToBits() == 0 }

func blendLane16(m Mask16, a, b Lane16) Lane16 {
	return Lane16{a.v.IfElse(m.v, b.v)}
}

func lane16ToLane48(a Lane16) Lane48 {
	x := a.v
	return Lane48{lane48Data{
		lo: x.GetLo().ExtendToInt64(),
		hi: x.GetHi().ExtendToInt64(),
	}}
}

func splatLane48(q Q48) Lane48 {
	v := archsimd.BroadcastInt64x4(q.raw)
	return Lane48{lane48Data{lo: v, hi: v}}
}

func loadLane48(p *[LaneWidth]Q48) Lane48 {
	raw := rawInt64Q48(p[:])
	return Lane48{lane48Data{
		lo: archsimd.LoadInt64x4(raw[:4]),
		hi: archsimd.LoadInt64x4(raw[4:]),
	}}
}

func storeLane48(a Lane48, p *[LaneWidth]Q48) {
	raw := rawInt64Q48(p[:])
	x := a.v
	x.lo.Store(raw[:4])
	x.hi.Store(raw[4:])
}

func addLane48(a, b Lane48) Lane48 {
	x, y := a.v, b.v
	lo, loEvents := vecAddSat64(x.lo, y.lo)
	hi, hiEvents := vecAddSat64(x.hi, y.hi)
	recordLaneSaturations(loEvents + hiEvents)
	return Lane48{lane48Data{lo: lo, hi: hi}}
}

// laneSubSat64 subtracts with Q48 saturation. Overflow is read from the sign
// of (x^y)&(x^r), and x chooses the saturation edge.
func laneSubSat64(x, y archsimd.Int64x4) (archsimd.Int64x4, uint64) {
	zero := archsimd.BroadcastInt64x4(0)
	r := x.Sub(y)
	ovf := zero.Greater(x.Xor(y).And(x.Xor(r)))
	sat := archsimd.BroadcastInt64x4(q48RawMin).
		IfElse(zero.Greater(x), archsimd.BroadcastInt64x4(q48RawMax))
	if !SaturationCountingEnabled {
		return sat.IfElse(ovf, r), 0
	}
	return sat.IfElse(ovf, r), uint64(bits.OnesCount8(ovf.ToBits()))
}

func subLane48(a, b Lane48) Lane48 {
	x, y := a.v, b.v
	lo, loEvents := laneSubSat64(x.lo, y.lo)
	hi, hiEvents := laneSubSat64(x.hi, y.hi)
	recordLaneSaturations(loEvents + hiEvents)
	return Lane48{lane48Data{lo: lo, hi: hi}}
}

func mulAdd16Lane48(a Lane48, b, c Lane16) Lane48 {
	acc := a.v
	order := archsimd.LoadUint32x8Array(&q48SplitOrder)
	productLo, productHi := vecProducts48(b.v, c.v, order)
	lo, loEvents := vecAddSat64(acc.lo, productLo)
	hi, hiEvents := vecAddSat64(acc.hi, productHi)
	recordLaneSaturations(loEvents + hiEvents)
	return Lane48{lane48Data{lo: lo, hi: hi}}
}

// laneNarrowQ48 packs the low words and checks that each discarded high word
// equals the candidate sign extension.
func laneNarrowQ48(lo, hi archsimd.Int64x4) (archsimd.Int32x8, uint64) {
	loBits, hiBits := lo.AsUint64x4(), hi.AsUint64x4()
	candidate := loBits.ShiftAllLeft(32).ShiftAllRight(32).
		Or(hiBits.ShiftAllLeft(32)).AsInt32x8()
	highWord := loBits.ShiftAllRight(32).
		Or(hiBits.ShiftAllRight(32).ShiftAllLeft(32)).AsInt32x8()
	ovf := highWord.NotEqual(candidate.ShiftAllRight(31))
	sat := highWord.ShiftAllRight(31).Xor(archsimd.BroadcastInt32x8(q16RawMax))
	if !SaturationCountingEnabled {
		return sat.IfElse(ovf, candidate), 0
	}
	return sat.IfElse(ovf, candidate), uint64(bits.OnesCount8(ovf.ToBits()))
}

func lane48ToLane16(a Lane48) Lane16 {
	x := a.v
	r, events := laneNarrowQ48(x.lo, x.hi)
	order := archsimd.LoadUint32x8Array(&q16NarrowOrder)
	recordLaneSaturations(events)
	return Lane16{r.Permute(order)}
}
