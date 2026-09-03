package fixed_test

import (
	"testing"

	"github.com/dhannyell/fixed"
)

func TestRotQuarterTurnsExact(t *testing.T) {
	v := fixed.Vec2{X: fixed.Q32FromInt(3), Y: fixed.Q32FromInt(4)}
	cases := []struct {
		name string
		r    fixed.Rot
		want fixed.Vec2
	}{
		{"identity", fixed.RotIdentity(), v},
		{"quarter", fixed.RotFromTurns(fixed.Q32FromRatio(1, 4)), fixed.Vec2{X: fixed.Q32FromInt(-4), Y: fixed.Q32FromInt(3)}},
		{"half", fixed.RotFromTurns(fixed.Q32Half()), fixed.Vec2{X: fixed.Q32FromInt(-3), Y: fixed.Q32FromInt(-4)}},
		{"three quarters", fixed.RotFromTurns(fixed.Q32FromRatio(3, 4)), fixed.Vec2{X: fixed.Q32FromInt(4), Y: fixed.Q32FromInt(-3)}},
	}
	for _, c := range cases {
		got := c.r.Apply(v)
		if !got.X.Eq(c.want.X) || !got.Y.Eq(c.want.Y) {
			t.Errorf("%s.Apply(3,4) = (%v, %v), want (%v, %v)", c.name, got.X, got.Y, c.want.X, c.want.Y)
		}
	}
}

func TestRotFromTurnsMatchesSeparateSineAndCosine(t *testing.T) {
	for _, raw := range []int64{0, 1, -1, 1 << 30, 1 << 31, 3 << 30, -1 << 32, 0x55555555} {
		r := fixed.RotFromTurns(fixed.Q32FromRaw(raw))
		if !r.Sin.Eq(fixed.SinTurns(fixed.Q32FromRaw(raw))) || !r.Cos.Eq(fixed.CosTurns(fixed.Q32FromRaw(raw))) {
			t.Errorf("RotFromTurns(%d) = (%d, %d), want (%d, %d)", raw, r.Sin.Raw(), r.Cos.Raw(), fixed.SinTurns(fixed.Q32FromRaw(raw)).Raw(), fixed.CosTurns(fixed.Q32FromRaw(raw)).Raw())
		}
	}
}

func TestRotMulMatchesAngleSum(t *testing.T) {
	angles := []fixed.Q32{
		fixed.Q32Zero(),
		fixed.Q32FromRatio(1, 3),
		fixed.Q32FromRatio(-2, 7),
		fixed.Q32MustParse("0.123"),
		fixed.Q32FromRatio(5, 8),
	}
	const budget = 1 << 14 // 2⁻¹⁸ per component.
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

func TestRotInvComposesToIdentity(t *testing.T) {
	const budget = 1 << 14 // 2⁻¹⁸ per component.
	for _, a := range []fixed.Q32{fixed.Q32FromRatio(1, 3), fixed.Q32MustParse("-0.4"), fixed.Q32FromRatio(7, 9)} {
		r := fixed.RotFromTurns(a)
		got := r.Mul(r.Inv())
		if d := got.Sin.Abs().Raw(); d > budget {
			t.Errorf("Mul with Inv at %v: Sin = %d raw units from 0", a, d)
		}
		if d := got.Cos.Sub(fixed.Q32One()).Abs().Raw(); d > budget {
			t.Errorf("Mul with Inv at %v: Cos = %d raw units from One", a, d)
		}
	}
}

func TestRotInvNormalized(t *testing.T) {
	cases := []fixed.Rot{
		{Sin: fixed.Q32FromInt(3), Cos: fixed.Q32FromInt(4)},
		fixed.RotFromTurns(fixed.Q32FromRatio(1, 7)).Mul(fixed.RotFromTurns(fixed.Q32FromRatio(2, 9))),
		{},
	}
	for _, r := range cases {
		got := r.InvNormalized()
		want := r.Normalize().Inv()
		if got != want {
			t.Fatalf("InvNormalized(%v) = %v, Normalize().Inv() = %v", r, got, want)
		}
		identity := r.Normalize().Mul(got)
		if d := identity.Sin.Abs().Raw(); d > 4 {
			t.Errorf("normalized inverse sine is %d raw units from zero", d)
		}
		if d := identity.Cos.Sub(fixed.Q32One()).Abs().Raw(); d > 4 {
			t.Errorf("normalized inverse cosine is %d raw units from one", d)
		}
	}
}

func TestRotNormalize(t *testing.T) {
	step := fixed.RotFromTurns(fixed.Q32FromRatio(1, 100))
	r := fixed.RotIdentity()
	for range 100 {
		r = r.Mul(step)
	}
	r = r.Normalize()
	lenSq := r.Sin.Mul(r.Sin).Add(r.Cos.Mul(r.Cos))
	if d := lenSq.Sub(fixed.Q32One()).Abs().Raw(); d > 4 {
		t.Errorf("length² after Normalize = %d raw units from One", d)
	}
	if got := (fixed.Rot{}).Normalize(); !got.Cos.Eq(fixed.Q32One()) || !got.Sin.Eq(fixed.Q32Zero()) {
		t.Errorf("Normalize of the zero Rot = (%v, %v), want the identity", got.Sin, got.Cos)
	}
	if got := (fixed.Rot{Sin: fixed.Q32FromRaw(1)}).Normalize(); !got.Sin.Eq(fixed.Q32One()) || !got.Cos.Eq(fixed.Q32Zero()) {
		t.Errorf("Normalize of the smallest nonzero Rot = (%v, %v), want (1, 0)", got.Sin, got.Cos)
	}
}
