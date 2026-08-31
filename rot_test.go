package fixed_test

import (
	"testing"

	"github.com/dhannyell/fixed"
)

// TestRotQuarterTurnsExact checks the exact cases: on the quarter
// turns the sine and cosine are 0, One or -One, and Mul by these
// values loses no bits.
func TestRotQuarterTurnsExact(t *testing.T) {
	v := fixed.Vec2{X: fixed.FromInt(3), Y: fixed.FromInt(4)}
	cases := []struct {
		name string
		r    fixed.Rot
		want fixed.Vec2
	}{
		{"identity", fixed.RotIdentity(), v},
		{"quarter", fixed.RotFromTurns(fixed.FromRatio(1, 4)), fixed.Vec2{X: fixed.FromInt(-4), Y: fixed.FromInt(3)}},
		{"half", fixed.RotFromTurns(fixed.Half()), fixed.Vec2{X: fixed.FromInt(-3), Y: fixed.FromInt(-4)}},
		{"three quarters", fixed.RotFromTurns(fixed.FromRatio(3, 4)), fixed.Vec2{X: fixed.FromInt(4), Y: fixed.FromInt(-3)}},
	}
	for _, c := range cases {
		got := c.r.Apply(v)
		if !got.X.Eq(c.want.X) || !got.Y.Eq(c.want.Y) {
			t.Errorf("%s.Apply(3,4) = (%v, %v), want (%v, %v)", c.name, got.X, got.Y, c.want.X, c.want.Y)
		}
	}
}

// TestRotMulMatchesAngleSum compares composition against the direct
// evaluation at the summed angle. Budget per component: 2^-18, the
// kernel error doubled by the products plus the Mul floors.
func TestRotMulMatchesAngleSum(t *testing.T) {
	angles := []fixed.Q{
		fixed.Zero(),
		fixed.FromRatio(1, 3),
		fixed.FromRatio(-2, 7),
		fixed.MustParse("0.123"),
		fixed.FromRatio(5, 8),
	}
	const budget = 1 << 14
	for _, a := range angles {
		for _, b := range angles {
			got := fixed.RotFromTurns(a).Mul(fixed.RotFromTurns(b))
			want := fixed.RotFromTurns(a.Add(b))
			if d := got.Sin.Sub(want.Sin).Abs().Raw(); d > budget {
				t.Errorf("Mul(%v, %v).Sin drifted %d raw units", a, b, d)
			}
			if d := got.Cos.Sub(want.Cos).Abs().Raw(); d > budget {
				t.Errorf("Mul(%v, %v).Cos drifted %d raw units", a, b, d)
			}
		}
	}
}

// TestRotInvComposesToIdentity checks r.Mul(r.Inv()) against the
// identity within the same 2^-18 budget.
func TestRotInvComposesToIdentity(t *testing.T) {
	const budget = 1 << 14
	for _, a := range []fixed.Q{fixed.FromRatio(1, 3), fixed.MustParse("-0.4"), fixed.FromRatio(7, 9)} {
		r := fixed.RotFromTurns(a)
		got := r.Mul(r.Inv())
		if d := got.Sin.Abs().Raw(); d > budget {
			t.Errorf("Mul with Inv at %v: Sin = %d raw units from 0", a, d)
		}
		if d := got.Cos.Sub(fixed.One()).Abs().Raw(); d > budget {
			t.Errorf("Mul with Inv at %v: Cos = %d raw units from One", a, d)
		}
	}
}

// TestRotNormalize checks the drift repair after a long chain of Mul,
// and the zero-value escape hatch.
func TestRotNormalize(t *testing.T) {
	step := fixed.RotFromTurns(fixed.FromRatio(1, 100))
	r := fixed.RotIdentity()
	for range 100 {
		r = r.Mul(step)
	}
	r = r.Normalize()
	lenSq := r.Sin.Mul(r.Sin).Add(r.Cos.Mul(r.Cos))
	if d := lenSq.Sub(fixed.One()).Abs().Raw(); d > 4 {
		t.Errorf("length² after Normalize = %d raw units from One", d)
	}
	if got := (fixed.Rot{}).Normalize(); !got.Cos.Eq(fixed.One()) || !got.Sin.Eq(fixed.Zero()) {
		t.Errorf("Normalize of the zero Rot = (%v, %v), want the identity", got.Sin, got.Cos)
	}
}
