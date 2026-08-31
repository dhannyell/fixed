package fixed_test

import (
	"testing"

	"github.com/dhannyell/fixed"
)

func vec2FromInts(x, y int) fixed.Vec2 {
	return fixed.Vec2{X: fixed.FromInt(x), Y: fixed.FromInt(y)}
}

func TestVec2ExactCases(t *testing.T) {
	v := vec2FromInts(3, 4)
	if got := v.LenSq(); !got.Eq(fixed.FromInt(25)) {
		t.Errorf("LenSq(3,4) = %d, want 25", got.Raw())
	}
	if got := v.Len(); !got.Eq(fixed.FromInt(5)) {
		t.Errorf("Len(3,4) = %d, want 5", got.Raw())
	}
	if got := vec2FromInts(1, 2).Dot(vec2FromInts(3, 4)); !got.Eq(fixed.FromInt(11)) {
		t.Errorf("Dot((1,2),(3,4)) = %d, want 11", got.Raw())
	}
	if got := vec2FromInts(-3, -4).Len(); !got.Eq(fixed.FromInt(5)) {
		t.Errorf("Len(-3,-4) = %d, want 5", got.Raw())
	}
	if got := vec2FromInts(6, 8).Div(fixed.FromInt(2)); got != vec2FromInts(3, 4) {
		t.Errorf("Div((6,8), 2) = %v, want (3,4)", got)
	}
	if got := vec2FromInts(1, 1).Distance(vec2FromInts(4, 5)); !got.Eq(fixed.FromInt(5)) {
		t.Errorf("Distance((1,1),(4,5)) = %d, want 5", got.Raw())
	}
	if got := vec2FromInts(0, 0).Lerp(vec2FromInts(10, 20), fixed.Half()); got != vec2FromInts(5, 10) {
		t.Errorf("Lerp to (10,20) at half = %v, want (5,10)", got)
	}
	if got := vec2FromInts(0, -7).Normalize(); got != vec2FromInts(0, -1) {
		t.Errorf("Normalize(0,-7) = %v, want (0,-1)", got)
	}
	if got := (fixed.Vec2{}).Normalize(); got != (fixed.Vec2{}) {
		t.Errorf("Normalize(zero) = %v, want the zero vector", got)
	}
}

func TestVec2NormPreservesScale(t *testing.T) {
	cases := []struct {
		name        string
		x           fixed.Q
		length      fixed.Q
		unit        fixed.Vec2
		saturations uint64
	}{
		{"smallest positive", fixed.FromRaw(1), fixed.FromRaw(1), fixed.Vec2{X: fixed.One()}, 0},
		{"large representable", fixed.FromInt(1 << 20), fixed.FromInt(1 << 20), fixed.Vec2{X: fixed.One()}, 0},
		{"minimum component", fixed.MinValue(), fixed.MaxValue(), fixed.Vec2{X: fixed.One().Neg()}, 1},
	}
	for _, c := range cases {
		v := fixed.Vec2{X: c.x}
		fixed.ResetSaturationCount()
		if got := v.Len(); !got.Eq(c.length) {
			t.Errorf("%s Len = %v, want %v", c.name, got, c.length)
		}
		if got := v.Normalize(); got != c.unit {
			t.Errorf("%s Normalize = %v, want %v", c.name, got, c.unit)
		}
		if got := fixed.SaturationCount(); got != c.saturations {
			t.Errorf("%s SaturationCount = %d, want %d", c.name, got, c.saturations)
		}
	}

	u := (fixed.Vec2{X: fixed.FromRaw(1), Y: fixed.FromRaw(1)}).Normalize()
	if u.X.Raw() == 0 || !u.X.Eq(u.Y) {
		t.Fatalf("Normalize of the smallest diagonal = %v, want equal nonzero components", u)
	}
	if d := u.LenSq().Sub(fixed.One()).Abs().Raw(); d > 4 {
		t.Errorf("normalized smallest diagonal length² = %d raw units from One", d)
	}
}

func TestVec2SaturationPropagates(t *testing.T) {
	fixed.ResetSaturationCount()
	v := fixed.Vec2{X: fixed.FromInt(1 << 20), Y: fixed.Zero()}
	if got := v.LenSq(); !got.Eq(fixed.MaxValue()) {
		t.Errorf("LenSq(2^20, 0) = %d, want MaxValue", got.Raw())
	}
	if got := fixed.SaturationCount(); got != 1 {
		t.Errorf("SaturationCount = %d, want 1", got)
	}
}

func TestVec2DivByZeroPanics(t *testing.T) {
	expectPanic(t, func() { vec2FromInts(1, 2).Div(fixed.Zero()) })
}
