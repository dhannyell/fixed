package fixed_test

import (
	"testing"

	"github.com/dhannyell/fixed"
)

func vec2FromInts(x, y int) fixed.Vec2 {
	return fixed.Vec2{X: fixed.FromInt(x), Y: fixed.FromInt(y)}
}

// TestVec2ExactCases pins vector results that the Q32.32 format
// represents without rounding, so every comparison is bit-exact.
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

// TestVec2SaturationPropagates shows the headroom rule on vectors: one
// oversized component saturates LenSq and counts one event.
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

// FuzzVec2DotVsBig checks Dot against the composed scalar oracle. The
// oracle mirrors the documented composition order: per-component Mul,
// then Add, each saturating on its own.
func FuzzVec2DotVsBig(f *testing.F) {
	for _, v := range fuzzSeeds() {
		f.Add(v, v, v, v)
		f.Add(v, int64(1), -v, int64(1)<<32)
	}
	f.Fuzz(func(t *testing.T, ax, ay, bx, by int64) {
		a := fixed.Vec2{X: fixed.FromRaw(ax), Y: fixed.FromRaw(ay)}
		b := fixed.Vec2{X: fixed.FromRaw(bx), Y: fixed.FromRaw(by)}
		want := oracleAdd(oracleMul(ax, bx), oracleMul(ay, by))
		if got := a.Dot(b).Raw(); got != want {
			t.Errorf("Dot((%d,%d),(%d,%d)) = %d, oracle says %d", ax, ay, bx, by, got, want)
		}
	})
}
