package fixed_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/dhannyell/fixed"
)

func oracleQ16Sqrt(raw int32) int32 {
	z := new(big.Int).Lsh(big.NewInt(int64(raw)), 16)
	return int32(z.Sqrt(z).Int64())
}

func TestQ16MulAndDivProduceExactBits(t *testing.T) {
	if got := fixed.Q16One().Mul(fixed.Q16One()); !got.Eq(fixed.Q16One()) {
		t.Errorf("One.Mul(One) = %d, want One", got.Raw())
	}
	if got := fixed.Q16Half().Mul(fixed.Q16Half()).Raw(); got != 1<<14 {
		t.Errorf("Half.Mul(Half) = %d, want %d", got, int32(1)<<14)
	}
	if got := fixed.Q16FromInt(3).Div(fixed.Q16FromInt(2)).Raw(); got != 3<<15 {
		t.Errorf("3/2 = %d, want %d", got, int32(3)<<15)
	}
}

func TestQ16MulFloorsTinyNegativeProducts(t *testing.T) {
	// The arithmetic shift floors this product. It must not become zero.
	if got := fixed.Q16FromRaw(-1).Mul(fixed.Q16FromRaw(1)).Raw(); got != -1 {
		t.Errorf("FromRaw(-1).Mul(FromRaw(1)) = %d, want -1", got)
	}
	// -3·2⁻¹⁶ times one half floors to -2·2⁻¹⁶, not toward zero.
	if got := fixed.Q16FromRaw(-3).Mul(fixed.Q16Half()).Raw(); got != -2 {
		t.Errorf("FromRaw(-3).Mul(Half) = %d, want -2", got)
	}
}

func TestQ16DivTruncatesTowardZero(t *testing.T) {
	// Floor gives -21846. This result verifies truncation toward zero.
	if got := fixed.Q16FromRatio(-1, 3).Raw(); got != -21845 {
		t.Errorf("Q16FromRatio(-1, 3) = %d, want -21845", got)
	}
	if got := fixed.Q16FromInt(-1).Div(fixed.Q16FromInt(3)).Raw(); got != -21845 {
		t.Errorf("(-1).Div(3) = %d, want -21845", got)
	}
}

func TestQ16SaturationClampsAndCounts(t *testing.T) {
	cases := []struct {
		name string
		op   func() fixed.Q16
		want fixed.Q16
	}{
		{"MaxValue.Add(epsilon)", func() fixed.Q16 { return fixed.Q16MaxValue().Add(fixed.Q16FromRaw(1)) }, fixed.Q16MaxValue()},
		{"MinValue.Sub(epsilon)", func() fixed.Q16 { return fixed.Q16MinValue().Sub(fixed.Q16FromRaw(1)) }, fixed.Q16MinValue()},
		{"2^8 * 2^8", func() fixed.Q16 { return fixed.Q16FromInt(1 << 8).Mul(fixed.Q16FromInt(1 << 8)) }, fixed.Q16MaxValue()},
		{"MaxValue / epsilon", func() fixed.Q16 { return fixed.Q16MaxValue().Div(fixed.Q16FromRaw(1)) }, fixed.Q16MaxValue()},
		{"FromRatio overflow", func() fixed.Q16 { return fixed.Q16FromRatio(-1<<15, -1) }, fixed.Q16MaxValue()},
		{"FromInt overflow", func() fixed.Q16 { return fixed.Q16FromInt(1 << 15) }, fixed.Q16MaxValue()},
		{"FromInt underflow", func() fixed.Q16 { return fixed.Q16FromInt(-1<<15 - 1) }, fixed.Q16MinValue()},
		{"MustParse overflow", func() fixed.Q16 { return fixed.Q16MustParse("32768") }, fixed.Q16MaxValue()},
		{"MinValue.Neg()", func() fixed.Q16 { return fixed.Q16MinValue().Neg() }, fixed.Q16MaxValue()},
		{"MinValue.Abs()", func() fixed.Q16 { return fixed.Q16MinValue().Abs() }, fixed.Q16MaxValue()},
		{"MaxValue.Ceil()", func() fixed.Q16 { return fixed.Q16MaxValue().Ceil() }, fixed.Q16MaxValue()},
		{"MaxValue.Round()", func() fixed.Q16 { return fixed.Q16MaxValue().Round() }, fixed.Q16MaxValue()},
		{"Q32.ToQ16 overflow", func() fixed.Q16 { return fixed.Q32MaxValue().ToQ16() }, fixed.Q16MaxValue()},
		{"Q32.ToQ16 underflow", func() fixed.Q16 { return fixed.Q32MinValue().ToQ16() }, fixed.Q16MinValue()},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fixed.ResetSaturationCount()
			if got := c.op(); !got.Eq(c.want) {
				t.Errorf("result = %d, want %d", got.Raw(), c.want.Raw())
			}
			if got := fixed.SaturationCount(); got != 1 {
				t.Errorf("SaturationCount = %d, want 1", got)
			}
		})
	}
}

func TestQ16ExactBoundariesDoNotSaturate(t *testing.T) {
	fixed.ResetSaturationCount()
	if got := fixed.Q16MinValue().Sub(fixed.Q16MinValue()); !got.Eq(fixed.Q16Zero()) {
		t.Errorf("MinValue.Sub(MinValue) = %d, want Zero", got.Raw())
	}
	if got := fixed.Q16MinValue().Div(fixed.Q16One()); !got.Eq(fixed.Q16MinValue()) {
		t.Errorf("MinValue.Div(One) = %d, want MinValue", got.Raw())
	}
	if got := fixed.Q16FromInt(-1 << 15); !got.Eq(fixed.Q16MinValue()) {
		t.Errorf("Q16FromInt(-1<<15) = %d, want MinValue", got.Raw())
	}
	if got := fixed.Q16MustParse("-32768"); !got.Eq(fixed.Q16MinValue()) {
		t.Errorf("Q16MustParse(-32768) = %d, want MinValue", got.Raw())
	}
	if c := fixed.SaturationCount(); c != 0 {
		t.Errorf("SaturationCount = %d, want 0: exact boundary results are not events", c)
	}
}

func TestQ16DomainErrorsPanic(t *testing.T) {
	t.Run("DivByZero", func(t *testing.T) {
		expectPanic(t, func() { fixed.Q16One().Div(fixed.Q16Zero()) })
	})
	t.Run("FromRatioZeroDenominator", func(t *testing.T) {
		expectPanic(t, func() { fixed.Q16FromRatio(1, 0) })
	})
	t.Run("SqrtOfNegative", func(t *testing.T) {
		expectPanic(t, func() { fixed.Q16FromInt(-1).Sqrt() })
	})
}

func TestQ16SqrtFloorsTheRoot(t *testing.T) {
	if got := fixed.Q16FromInt(4).Sqrt(); !got.Eq(fixed.Q16FromInt(2)) {
		t.Errorf("Sqrt(4) = %d, want 2", got.Raw())
	}
	if got := fixed.Q16FromInt(2).Sqrt().Raw(); got != 92681 {
		t.Errorf("Sqrt(2) = %d, want 92681", got)
	}
	if got := fixed.Q16Zero().Sqrt(); !got.Eq(fixed.Q16Zero()) {
		t.Errorf("Sqrt(0) = %d, want 0", got.Raw())
	}
	if got, want := fixed.Q16MaxValue().Sqrt().Raw(), oracleQ16Sqrt(math.MaxInt32); got != want {
		t.Errorf("Sqrt(MaxValue) = %d, oracle says %d", got, want)
	}
}

func TestQ16IntegerConversions(t *testing.T) {
	cases := []struct {
		num, den                  int
		floor, ceil, round, trunc int
	}{
		{5, 2, 2, 3, 3, 2},
		{-5, 2, -3, -2, -3, -2},
		{2, 1, 2, 2, 2, 2},
		{-2, 1, -2, -2, -2, -2},
		{-1, 2, -1, 0, -1, 0},
	}
	for _, c := range cases {
		q := fixed.Q16FromRatio(c.num, c.den)
		if got := q.Floor().Int(); got != c.floor {
			t.Errorf("Floor(%d/%d) = %d, want %d", c.num, c.den, got, c.floor)
		}
		if got := q.Ceil().Int(); got != c.ceil {
			t.Errorf("Ceil(%d/%d) = %d, want %d", c.num, c.den, got, c.ceil)
		}
		if got := q.Round().Int(); got != c.round {
			t.Errorf("Round(%d/%d) = %d, want %d", c.num, c.den, got, c.round)
		}
		if got := q.Int(); got != c.trunc {
			t.Errorf("Int(%d/%d) = %d, want %d", c.num, c.den, got, c.trunc)
		}
	}
}

func TestQ16ComparisonsOrderValues(t *testing.T) {
	a, b := fixed.Q16FromInt(-2), fixed.Q16One()
	if a.Cmp(b) != -1 || b.Cmp(a) != 1 || a.Cmp(a) != 0 {
		t.Errorf("Cmp does not order %d and %d", a.Raw(), b.Raw())
	}
	if !a.Less(b) || !b.Greater(a) || a.Greater(a) {
		t.Errorf("Less and Greater disagree with the order of %d and %d", a.Raw(), b.Raw())
	}
	if !a.Min(b).Eq(a) || !a.Max(b).Eq(b) {
		t.Errorf("Min and Max disagree with the order of %d and %d", a.Raw(), b.Raw())
	}
	if got := fixed.Q16FromInt(9).Clamp(a, b); !got.Eq(b) {
		t.Errorf("Clamp(9) = %d, want the upper bound", got.Raw())
	}
}

func q16BoundaryRaws() []int32 {
	return []int32{
		0, 1, -1, 3, -3,
		1 << 15, -(1 << 15), 1 << 16, -(1 << 16),
		1 << 24, -(1 << 24), 0x5555, -0x5555,
		math.MinInt32, math.MinInt32 + 1, math.MaxInt32, math.MaxInt32 - 1,
	}
}

// TestQ16ConversionsWidenExactlyAndNarrowByFloor pins the conversion law:
// widening is exact, narrowing floors to the Q16.16 grid.
func TestQ16ConversionsWidenExactlyAndNarrowByFloor(t *testing.T) {
	fixed.ResetSaturationCount()
	for _, raw := range q16BoundaryRaws() {
		q := fixed.Q16FromRaw(raw)
		if got := q.ToQ32().ToQ16(); !got.Eq(q) {
			t.Errorf("round trip of raw %d = %d", raw, got.Raw())
		}
	}
	if c := fixed.SaturationCount(); c != 0 {
		t.Errorf("SaturationCount = %d, want 0: the round trip is exact", c)
	}
	if got := fixed.Q16One().ToQ32(); !got.Eq(fixed.Q32One()) {
		t.Errorf("Q16One widens to %d, want Q32One", got.Raw())
	}
	// The narrowing floors: -2⁻³² lands one full Q16 step below zero.
	if got := fixed.Q32FromRaw(-1).ToQ16().Raw(); got != -1 {
		t.Errorf("ToQ16(-2⁻³²) = %d, want -1", got)
	}
}

// TestQ16OpsMatchTheWidenedPath checks Add, Sub and Mul against the widened
// Q32 computation. The law does not hold for Div, which truncates toward
// zero while the narrowing conversion floors.
func TestQ16OpsMatchTheWidenedPath(t *testing.T) {
	raws := q16BoundaryRaws()
	for _, a := range raws {
		for _, b := range raws {
			qa, qb := fixed.Q16FromRaw(a), fixed.Q16FromRaw(b)
			wa, wb := qa.ToQ32(), qb.ToQ32()
			if got, want := qa.Add(qb), wa.Add(wb).ToQ16(); !got.Eq(want) {
				t.Fatalf("Add(%d, %d) = %d, widened path says %d", a, b, got.Raw(), want.Raw())
			}
			if got, want := qa.Sub(qb), wa.Sub(wb).ToQ16(); !got.Eq(want) {
				t.Fatalf("Sub(%d, %d) = %d, widened path says %d", a, b, got.Raw(), want.Raw())
			}
			if got, want := qa.Mul(qb), wa.Mul(wb).ToQ16(); !got.Eq(want) {
				t.Fatalf("Mul(%d, %d) = %d, widened path says %d", a, b, got.Raw(), want.Raw())
			}
		}
	}
}

func FuzzQ16MulVsWidened(f *testing.F) {
	for _, v := range q16BoundaryRaws() {
		f.Add(v, v)
		f.Add(v, int32(1))
	}
	f.Fuzz(func(t *testing.T, a, b int32) {
		qa, qb := fixed.Q16FromRaw(a), fixed.Q16FromRaw(b)
		if got, want := qa.Mul(qb), qa.ToQ32().Mul(qb.ToQ32()).ToQ16(); !got.Eq(want) {
			t.Errorf("Mul(%d, %d) = %d, widened path says %d", a, b, got.Raw(), want.Raw())
		}
		if x, y := qa.Mul(qb), qb.Mul(qa); !x.Eq(y) {
			t.Errorf("Mul(%d, %d) != Mul(%d, %d)", a, b, b, a)
		}
	})
}
