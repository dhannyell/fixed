package fixed_test

import (
	"math"
	"testing"

	"github.com/dhannyell/fixed"
)

// These values are part of the compatibility contract.
func TestSinCosTurnsGoldenBits(t *testing.T) {
	cases := []struct {
		name      string
		got, want fixed.Q32
	}{
		{"Sin(0)", fixed.SinTurns(fixed.Q32Zero()), fixed.Q32Zero()},
		{"Sin(1/4)", fixed.SinTurns(fixed.Q32FromRatio(1, 4)), fixed.Q32One()},
		{"Sin(1/2)", fixed.SinTurns(fixed.Q32Half()), fixed.Q32Zero()},
		{"Sin(3/4)", fixed.SinTurns(fixed.Q32FromRatio(3, 4)), fixed.Q32One().Neg()},
		{"Sin(1/8)", fixed.SinTurns(fixed.Q32FromRatio(1, 8)), fixed.Q32FromRaw(0xB504F334)},
		{"Cos(0)", fixed.CosTurns(fixed.Q32Zero()), fixed.Q32One()},
		{"Cos(1/4)", fixed.CosTurns(fixed.Q32FromRatio(1, 4)), fixed.Q32Zero()},
		{"Cos(1/2)", fixed.CosTurns(fixed.Q32Half()), fixed.Q32One().Neg()},
	}
	for _, c := range cases {
		if !c.got.Eq(c.want) {
			t.Errorf("%s = %d, want %d", c.name, c.got.Raw(), c.want.Raw())
		}
	}
	// The 1/8 pin must be sin(pi/4) rounded to the Q32.32 grid.
	if e := math.Abs(float64(0xB504F334)*0x1p-32 - math.Sin(math.Pi/4)); e > 0x1p-33 {
		t.Errorf("Sin(1/8) pin is %.3g away from the oracle", e)
	}
}

func TestSinCosTurnsContracts(t *testing.T) {
	angles := []fixed.Q32{
		fixed.Q32Zero(),
		fixed.Q32FromRatio(1, 3),
		fixed.Q32FromRatio(-2, 7),
		fixed.Q32MustParse("0.123"),
		fixed.Q32FromRaw(1),
		fixed.Q32FromInt(41).Add(fixed.Q32FromRatio(5, 9)),
	}
	fixed.ResetSaturationCount()
	for _, a := range angles {
		if got, want := fixed.SinTurns(a.Add(fixed.Q32One())), fixed.SinTurns(a); !got.Eq(want) {
			t.Errorf("SinTurns(%v + 1 turn) = %d, want %d", a, got.Raw(), want.Raw())
		}
		if got, want := fixed.SinTurns(a.Neg()), fixed.SinTurns(a).Neg(); !got.Eq(want) {
			t.Errorf("SinTurns(-%v) = %d, want %d: sine is odd", a, got.Raw(), want.Raw())
		}
		if got, want := fixed.CosTurns(a.Neg()), fixed.CosTurns(a); !got.Eq(want) {
			t.Errorf("CosTurns(-%v) = %d, want %d: cosine is even", a, got.Raw(), want.Raw())
		}
	}
	for _, a := range []fixed.Q32{fixed.Q32MinValue(), fixed.Q32MaxValue()} {
		if v := fixed.SinTurns(a).Abs(); fixed.Q32One().Less(v) {
			t.Errorf("SinTurns(%d) = out of [-One, One]", a.Raw())
		}
	}
	if c := fixed.SaturationCount(); c != 0 {
		t.Errorf("SaturationCount = %d, want 0: the kernel never saturates", c)
	}
}

// The float64 oracle noise is negligible against the 2⁻²⁰ error limit.
func TestSinCosTurnsMeetTheFloor(t *testing.T) {
	const floor = 1.0 / (1 << 20)
	worst := 0.0
	check := func(u uint32) {
		a := fixed.Q32FromRaw(int64(u))
		x := 2 * math.Pi * float64(u) * 0x1p-32
		if e := math.Abs(float64(fixed.SinTurns(a).Raw())*0x1p-32 - math.Sin(x)); e > worst {
			worst = e
		}
		if e := math.Abs(float64(fixed.CosTurns(a).Raw())*0x1p-32 - math.Cos(x)); e > worst {
			worst = e
		}
	}
	for i := range uint64(1 << 20) {
		check(uint32(i << 12))
	}
	for _, u := range []uint32{
		0, 1, 1<<30 - 1, 1 << 30, 1<<30 + 1, 1<<31 - 1, 1 << 31,
		1<<31 + 1, 3<<30 - 1, 3 << 30, 3<<30 + 1, 1<<32 - 1, 0x55555555,
	} {
		check(u)
	}
	t.Logf("max abs error %.3g (floor %.3g)", worst, floor)
	if worst > floor {
		t.Errorf("max abs error %.3g exceeds the 2^-20 floor", worst)
	}
}

// atan(1) is exactly 1/8 turn on the Q32.32 grid.
func TestAtan2TurnsGoldenBits(t *testing.T) {
	one := fixed.Q32One()
	cases := []struct {
		name      string
		got, want fixed.Q32
	}{
		{"(0,0)", fixed.Atan2Turns(fixed.Q32Zero(), fixed.Q32Zero()), fixed.Q32Zero()},
		{"+x axis", fixed.Atan2Turns(fixed.Q32Zero(), one), fixed.Q32Zero()},
		{"+y axis", fixed.Atan2Turns(one, fixed.Q32Zero()), fixed.Q32FromRatio(1, 4)},
		{"-x axis", fixed.Atan2Turns(fixed.Q32Zero(), one.Neg()), fixed.Q32Half()},
		{"-y axis", fixed.Atan2Turns(one.Neg(), fixed.Q32Zero()), fixed.Q32FromRatio(-1, 4)},
		{"diag 1/8", fixed.Atan2Turns(one, one), fixed.Q32FromRatio(1, 8)},
		{"diag 3/8", fixed.Atan2Turns(one, one.Neg()), fixed.Q32FromRatio(3, 8)},
		{"diag -3/8", fixed.Atan2Turns(one.Neg(), one.Neg()), fixed.Q32FromRatio(-3, 8)},
		{"diag -1/8", fixed.Atan2Turns(one.Neg(), one), fixed.Q32FromRatio(-1, 8)},
		{"scale free", fixed.Atan2Turns(fixed.Q32FromInt(7), fixed.Q32FromInt(7)), fixed.Q32FromRatio(1, 8)},
	}
	for _, c := range cases {
		if !c.got.Eq(c.want) {
			t.Errorf("Atan2Turns %s = %d, want %d", c.name, c.got.Raw(), c.want.Raw())
		}
	}
}

// Angle differences wrap at half a turn.
func TestAtan2TurnsMeetsTheFloor(t *testing.T) {
	const floor = 1.0 / (1 << 20)
	worst := 0.0
	check := func(xr, yr int64) {
		if xr == 0 && yr == 0 {
			return
		}
		got := float64(fixed.Atan2Turns(fixed.Q32FromRaw(yr), fixed.Q32FromRaw(xr)).Raw()) * 0x1p-32
		want := math.Atan2(float64(yr), float64(xr)) / (2 * math.Pi)
		d := got - want
		if d > 0.5 {
			d -= 1
		} else if d <= -0.5 {
			d += 1
		}
		if e := math.Abs(d); e > worst {
			worst = e
		}
	}
	ux, uy := uint64(0), uint64(0)
	for range 1 << 18 {
		ux += 0x9E3779B97F4A7C15
		uy += 0xC2B2AE3D27D4EB4F
		check(int64(ux), int64(uy))
	}
	extremes := []int64{1, -1, 1 << 32, fixed.Q32MaxValue().Raw(), fixed.Q32MinValue().Raw()}
	for _, xr := range extremes {
		for _, yr := range extremes {
			check(xr, yr)
		}
	}
	t.Logf("max abs error %.3g turn (floor %.3g)", worst, floor)
	if worst > floor {
		t.Errorf("max abs error %.3g exceeds the 2^-20 floor", worst)
	}
}

// The 2⁻¹⁸ budget includes both table interpolations.
func FuzzAtan2SinCosRoundTrip(f *testing.F) {
	for _, v := range fuzzSeeds() {
		f.Add(v)
	}
	f.Fuzz(func(t *testing.T, raw int64) {
		a := fixed.Q32FromRaw(raw)
		back := fixed.Atan2Turns(fixed.SinTurns(a), fixed.CosTurns(a))
		d := int64(int32(uint32(back.Raw()) - uint32(raw)))
		if d < -1<<14 || d > 1<<14 {
			t.Errorf("round trip at raw %d drifted %d raw units, budget 2⁻¹⁸ turn", raw, d)
		}
	})
}

// The 2⁻¹⁸ budget includes both squares and multiplication floors.
func FuzzSinCosPythagoras(f *testing.F) {
	for _, v := range fuzzSeeds() {
		f.Add(v)
	}
	f.Fuzz(func(t *testing.T, raw int64) {
		a := fixed.Q32FromRaw(raw)
		s, c := fixed.SinTurns(a), fixed.CosTurns(a)
		r := s.Mul(s).Add(c.Mul(c))
		if r.Sub(fixed.Q32One()).Abs().Raw() > 1<<14 {
			t.Errorf("sin²+cos² at raw %d = %d, more than 2⁻¹⁸ away from One", raw, r.Raw())
		}
	})
}
