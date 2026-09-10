//go:build goexperiment.simd && go1.27 && arm64

package fixed

import "simd/archsimd"

// LaneWidth is the number of values in an arm64 SIMD lane type.
const LaneWidth = 4

type lane16Data = archsimd.Int32x4
type mask16Data = archsimd.Mask32x4
type shift16Data = archsimd.Int32x4

type lane48Data struct {
	lo archsimd.Int64x2
	hi archsimd.Int64x2
}

func lanesAvailable() bool { return true }

func lanePath() string { return "neon" }

func recordLaneSaturations(events uint64) {
	if events != 0 {
		saturationEvents.Add(events)
	}
}

func splatLane16(q Q16) Lane16 {
	return Lane16{archsimd.BroadcastInt32x4(q.raw)}
}

func loadLane16(p *[LaneWidth]Q16) Lane16 {
	return Lane16{archsimd.LoadInt32x4(rawInt32(p[:]))}
}

func storeLane16(a Lane16, p *[LaneWidth]Q16) {
	a.v.Store(rawInt32(p[:]))
}

func laneMaskCount32(m archsimd.Mask32x4) uint64 {
	if !SaturationCountingEnabled {
		return 0
	}
	zero := archsimd.BroadcastInt32x4(0)
	return vecLaneSum(zero.Sub(m.ToInt32x4()))
}

func addLane16(a, b Lane16) Lane16 {
	x, y := a.v, b.v
	r := x.AddSaturated(y)
	recordLaneSaturations(laneMaskCount32(x.Add(y).NotEqual(r)))
	return Lane16{r}
}

func subLane16(a, b Lane16) Lane16 {
	x, y := a.v, b.v
	r := x.SubSaturated(y)
	recordLaneSaturations(laneMaskCount32(x.Sub(y).NotEqual(r)))
	return Lane16{r}
}

func mulLane16(a, b Lane16) Lane16 {
	x, y := a.v, b.v
	lo := x.MulWidenLo(y).ShiftAllRight(16)
	hi := x.HiToLo().MulWidenLo(y.HiToLo()).ShiftAllRight(16)
	r, ovf := vecNarrowPairNEON(lo, hi)
	zero := archsimd.BroadcastInt32x4(0)
	recordLaneSaturations(vecLaneSum(zero.Sub(ovf)))
	return Lane16{r}
}

func mulRoundLane16(a, b Lane16) Lane16 {
	x, y := a.v, b.v
	bias := archsimd.BroadcastInt64x2(q16RawHalf)
	lo := x.MulWidenLo(y).Add(bias).ShiftAllRight(16)
	hi := x.HiToLo().MulWidenLo(y.HiToLo()).Add(bias).ShiftAllRight(16)
	r, ovf := vecNarrowPairNEON(lo, hi)
	zero := archsimd.BroadcastInt32x4(0)
	recordLaneSaturations(vecLaneSum(zero.Sub(ovf)))
	return Lane16{r}
}

// NEON has no dedicated per-lane right shift. VSSHL reads a signed amount from
// the low byte of each lane: a positive amount shifts left, a negative amount
// shifts right. So this path stores the negated amount. That puts the negation
// in the constructors, which run once, and leaves the shift as one instruction.
// A stored amount reaches -31 at most, which fits the low byte.

func splatShift16(n uint8) Shift16 {
	return Shift16{archsimd.BroadcastInt32x4(-int32(n))}
}

func loadShift16(p *[LaneWidth]uint8) Shift16 {
	var w [LaneWidth]int32
	for i := range LaneWidth {
		w[i] = -int32(p[i])
	}
	return Shift16{archsimd.LoadInt32x4Array(&w)}
}

// scaleDownLane16 is one VSSHL. An arithmetic shift right cannot leave the
// int32 range, so this path records no saturation event.
func scaleDownLane16(a Lane16, s Shift16) Lane16 {
	return Lane16{a.v.Shift(s.v)}
}

// scaleDownRoundLane16 adds the bit below the truncation point to the shifted
// value, which rounds half away from the floor.
//
// This path stores the amount negated, so adding one to it reads one place
// higher. An amount of zero turns into a left shift by one place, whose low bit
// is zero, so it contributes nothing.
func scaleDownRoundLane16(a Lane16, s Shift16) Lane16 {
	ones := archsimd.BroadcastInt32x4(1)
	half := a.v.Shift(s.v.Add(ones)).And(ones)
	return Lane16{a.v.Shift(s.v).Add(half)}
}

func minLane16(a, b Lane16) Lane16 {
	return Lane16{a.v.Min(b.v)}
}

func maxLane16(a, b Lane16) Lane16 {
	return Lane16{a.v.Max(b.v)}
}

func symClampLane16(a, limit Lane16) Lane16 {
	zero := archsimd.BroadcastInt32x4(0)
	x, hi := a.v, limit.v
	neg := zero.SubSaturated(hi)
	recordLaneSaturations(laneMaskCount32(zero.Sub(hi).NotEqual(neg)))
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

func allZeroMask16(m Mask16) bool {
	return m.v.ToInt32x4().ReduceMin() == 0
}

func blendLane16(m Mask16, a, b Lane16) Lane16 {
	return Lane16{a.v.IfElse(m.v, b.v)}
}

func lane16ToLane48(a Lane16) Lane48 {
	x := a.v
	return Lane48{lane48Data{
		lo: x.ExtendLo2ToInt64(),
		hi: x.HiToLo().ExtendLo2ToInt64(),
	}}
}

func splatLane48(q Q48) Lane48 {
	v := archsimd.BroadcastInt64x2(q.raw)
	return Lane48{lane48Data{lo: v, hi: v}}
}

func loadLane48(p *[LaneWidth]Q48) Lane48 {
	raw := rawInt64Q48(p[:])
	return Lane48{lane48Data{
		lo: archsimd.LoadInt64x2(raw[:2]),
		hi: archsimd.LoadInt64x2(raw[2:]),
	}}
}

func storeLane48(a Lane48, p *[LaneWidth]Q48) {
	raw := rawInt64Q48(p[:])
	x := a.v
	x.lo.Store(raw[:2])
	x.hi.Store(raw[2:])
}

func laneMaskCount64(m archsimd.Mask64x2) uint64 {
	if !SaturationCountingEnabled {
		return 0
	}
	v := m.ToInt64x2()
	return uint64(-(v.GetElem(0) + v.GetElem(1)))
}

func laneAddSat64(x, y archsimd.Int64x2) (archsimd.Int64x2, uint64) {
	r := x.AddSaturated(y)
	return r, laneMaskCount64(x.Add(y).NotEqual(r))
}

func laneSubSat64(x, y archsimd.Int64x2) (archsimd.Int64x2, uint64) {
	r := x.SubSaturated(y)
	return r, laneMaskCount64(x.Sub(y).NotEqual(r))
}

func addLane48(a, b Lane48) Lane48 {
	x, y := a.v, b.v
	lo, loEvents := laneAddSat64(x.lo, y.lo)
	hi, hiEvents := laneAddSat64(x.hi, y.hi)
	recordLaneSaturations(loEvents + hiEvents)
	return Lane48{lane48Data{lo: lo, hi: hi}}
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
	x, y := b.v, c.v
	productLo := x.MulWidenLo(y).ShiftAllRight(16)
	productHi := x.HiToLo().MulWidenLo(y.HiToLo()).ShiftAllRight(16)
	lo, loEvents := laneAddSat64(acc.lo, productLo)
	hi, hiEvents := laneAddSat64(acc.hi, productHi)
	recordLaneSaturations(loEvents + hiEvents)
	return Lane48{lane48Data{lo: lo, hi: hi}}
}

func mulAdd16RoundLane48(a Lane48, b, c Lane16) Lane48 {
	acc := a.v
	x, y := b.v, c.v
	bias := archsimd.BroadcastInt64x2(q16RawHalf)
	productLo := x.MulWidenLo(y).Add(bias).ShiftAllRight(16)
	productHi := x.HiToLo().MulWidenLo(y.HiToLo()).Add(bias).ShiftAllRight(16)
	lo, loEvents := laneAddSat64(acc.lo, productLo)
	hi, hiEvents := laneAddSat64(acc.hi, productHi)
	recordLaneSaturations(loEvents + hiEvents)
	return Lane48{lane48Data{lo: lo, hi: hi}}
}

func lane48ToLane16(a Lane48) Lane16 {
	x := a.v
	r, ovf := vecNarrowPairNEON(x.lo, x.hi)
	zero := archsimd.BroadcastInt32x4(0)
	recordLaneSaturations(vecLaneSum(zero.Sub(ovf)))
	return Lane16{r}
}
