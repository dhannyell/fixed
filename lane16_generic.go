//go:build !goexperiment.simd || !go1.27 || (!amd64 && !arm64)

package fixed

// LaneWidth is the number of values in a lane type on this build path.
const LaneWidth = 4

type lane16Data [LaneWidth]int32
type mask16Data [LaneWidth]int32
type lane48Data [LaneWidth]int64

func lanesAvailable() bool { return true }

func lanePath() string { return "generic" }

func splatLane16(q Q16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		r.v[i] = q.raw
	}
	return r
}

func loadLane16(p *[LaneWidth]Q16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		r.v[i] = p[i].raw
	}
	return r
}

func storeLane16(a Lane16, p *[LaneWidth]Q16) {
	for i := range LaneWidth {
		p[i] = Q16{raw: a.v[i]}
	}
}

func addLane16(a, b Lane16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		r.v[i] = Q16{raw: a.v[i]}.Add(Q16{raw: b.v[i]}).raw
	}
	return r
}

func subLane16(a, b Lane16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		r.v[i] = Q16{raw: a.v[i]}.Sub(Q16{raw: b.v[i]}).raw
	}
	return r
}

func mulLane16(a, b Lane16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		r.v[i] = Q16{raw: a.v[i]}.Mul(Q16{raw: b.v[i]}).raw
	}
	return r
}

func mulRoundLane16(a, b Lane16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		r.v[i] = Q16{raw: a.v[i]}.MulRound(Q16{raw: b.v[i]}).raw
	}
	return r
}

func minLane16(a, b Lane16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		r.v[i] = Q16{raw: a.v[i]}.Min(Q16{raw: b.v[i]}).raw
	}
	return r
}

func maxLane16(a, b Lane16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		r.v[i] = Q16{raw: a.v[i]}.Max(Q16{raw: b.v[i]}).raw
	}
	return r
}

func symClampLane16(a, limit Lane16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		v := Q16{raw: a.v[i]}
		hi := Q16{raw: limit.v[i]}
		r.v[i] = v.Min(hi).Max(hi.Neg()).raw
	}
	return r
}

func greaterLane16(a, b Lane16) Mask16 {
	var r Mask16
	for i := range LaneWidth {
		if a.v[i] > b.v[i] {
			r.v[i] = -1
		}
	}
	return r
}

func equalsLane16(a, b Lane16) Mask16 {
	var r Mask16
	for i := range LaneWidth {
		if a.v[i] == b.v[i] {
			r.v[i] = -1
		}
	}
	return r
}

func orMask16(m, n Mask16) Mask16 {
	var r Mask16
	for i := range LaneWidth {
		r.v[i] = m.v[i] | n.v[i]
	}
	return r
}

func allZeroMask16(m Mask16) bool {
	for i := range LaneWidth {
		if m.v[i] != 0 {
			return false
		}
	}
	return true
}

func blendLane16(m Mask16, a, b Lane16) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		if m.v[i] != 0 {
			r.v[i] = a.v[i]
		} else {
			r.v[i] = b.v[i]
		}
	}
	return r
}

func lane16ToLane48(a Lane16) Lane48 {
	var r Lane48
	for i := range LaneWidth {
		r.v[i] = int64(a.v[i])
	}
	return r
}

func splatLane48(q Q48) Lane48 {
	var r Lane48
	for i := range LaneWidth {
		r.v[i] = q.raw
	}
	return r
}

func loadLane48(p *[LaneWidth]Q48) Lane48 {
	var r Lane48
	for i := range LaneWidth {
		r.v[i] = p[i].raw
	}
	return r
}

func storeLane48(a Lane48, p *[LaneWidth]Q48) {
	for i := range LaneWidth {
		p[i] = Q48{raw: a.v[i]}
	}
}

func addLane48(a, b Lane48) Lane48 {
	var r Lane48
	for i := range LaneWidth {
		r.v[i] = Q48{raw: a.v[i]}.Add(Q48{raw: b.v[i]}).raw
	}
	return r
}

func subLane48(a, b Lane48) Lane48 {
	var r Lane48
	for i := range LaneWidth {
		r.v[i] = Q48{raw: a.v[i]}.Sub(Q48{raw: b.v[i]}).raw
	}
	return r
}

func mulAdd16Lane48(a Lane48, b, c Lane16) Lane48 {
	var r Lane48
	for i := range LaneWidth {
		r.v[i] = Q48{raw: a.v[i]}.
			MulAdd16(Q16{raw: b.v[i]}, Q16{raw: c.v[i]}).raw
	}
	return r
}

func mulAdd16RoundLane48(a Lane48, b, c Lane16) Lane48 {
	var r Lane48
	for i := range LaneWidth {
		product := (int64(b.v[i])*int64(c.v[i]) + q16RawHalf) >> 16
		r.v[i] = Q48{raw: a.v[i]}.Add(Q48{raw: product}).raw
	}
	return r
}

func lane48ToLane16(a Lane48) Lane16 {
	var r Lane16
	for i := range LaneWidth {
		r.v[i] = Q48{raw: a.v[i]}.ToQ16().raw
	}
	return r
}
