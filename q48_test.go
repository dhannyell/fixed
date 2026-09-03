package fixed_test

import (
	"math"
	"math/big"
	"strconv"
	"testing"

	"github.com/dhannyell/fixed"
)

// q48Int builds a wide integer without an int argument, so the tests also
// compile on 32-bit architectures.
func q48Int(i int64) fixed.Q48 { return fixed.Q48FromRaw(i << 16) }

func oracleQ48Sqrt(raw int64) int64 {
	z := new(big.Int).Lsh(big.NewInt(raw), 16)
	return z.Sqrt(z).Int64()
}

func TestQ48MulAndDivProduceExactBits(t *testing.T) {
	if got := fixed.Q48One().Mul(fixed.Q48One()); !got.Eq(fixed.Q48One()) {
		t.Errorf("One.Mul(One) = %d, want One", got.Raw())
	}
	if got := fixed.Q48Half().Mul(fixed.Q48Half()).Raw(); got != 1<<14 {
		t.Errorf("Half.Mul(Half) = %d, want %d", got, int64(1)<<14)
	}
	if got := fixed.Q48FromInt(3).Div(fixed.Q48FromInt(2)).Raw(); got != 3<<15 {
		t.Errorf("3/2 = %d, want %d", got, int64(3)<<15)
	}
	// The product needs the high half of the 128-bit multiply.
	wide := q48Int(1 << 40)
	if got := wide.Mul(fixed.Q48FromRatio(1, 1<<10)); !got.Eq(q48Int(1 << 30)) {
		t.Errorf("2^40 * 2^-10 = %d, want 2^30", got.Raw())
	}
}

func TestQ48MulFloorsTinyNegativeProducts(t *testing.T) {
	// The arithmetic shift floors this product. It must not become zero.
	if got := fixed.Q48FromRaw(-1).Mul(fixed.Q48FromRaw(1)).Raw(); got != -1 {
		t.Errorf("FromRaw(-1).Mul(FromRaw(1)) = %d, want -1", got)
	}
	if got := fixed.Q48FromRaw(-3).Mul(fixed.Q48Half()).Raw(); got != -2 {
		t.Errorf("FromRaw(-3).Mul(Half) = %d, want -2", got)
	}
}

func TestQ48DivTruncatesTowardZero(t *testing.T) {
	// Floor gives -21846. This result verifies truncation toward zero.
	if got := fixed.Q48FromRatio(-1, 3).Raw(); got != -21845 {
		t.Errorf("Q48FromRatio(-1, 3) = %d, want -21845", got)
	}
	if got := fixed.Q48FromInt(-1).Div(fixed.Q48FromInt(3)).Raw(); got != -21845 {
		t.Errorf("(-1).Div(3) = %d, want -21845", got)
	}
}

func TestQ48SaturationClampsAndCounts(t *testing.T) {
	cases := []struct {
		name string
		op   func() fixed.Q48
		want fixed.Q48
	}{
		{"MaxValue.Add(epsilon)", func() fixed.Q48 { return fixed.Q48MaxValue().Add(fixed.Q48FromRaw(1)) }, fixed.Q48MaxValue()},
		{"MinValue.Sub(epsilon)", func() fixed.Q48 { return fixed.Q48MinValue().Sub(fixed.Q48FromRaw(1)) }, fixed.Q48MinValue()},
		{"2^24 * 2^24", func() fixed.Q48 { return fixed.Q48FromInt(1 << 24).Mul(fixed.Q48FromInt(1 << 24)) }, fixed.Q48MaxValue()},
		{"2^24 * -2^24", func() fixed.Q48 { return fixed.Q48FromInt(1 << 24).Mul(fixed.Q48FromInt(-1 << 24)) }, fixed.Q48MinValue()},
		{"MaxValue / epsilon", func() fixed.Q48 { return fixed.Q48MaxValue().Div(fixed.Q48FromRaw(1)) }, fixed.Q48MaxValue()},
		{"MustParse overflow", func() fixed.Q48 { return fixed.Q48MustParse("140737488355328") }, fixed.Q48MaxValue()},
		{"MinValue.Neg()", func() fixed.Q48 { return fixed.Q48MinValue().Neg() }, fixed.Q48MaxValue()},
		{"MinValue.Abs()", func() fixed.Q48 { return fixed.Q48MinValue().Abs() }, fixed.Q48MaxValue()},
		{"MaxValue.Ceil()", func() fixed.Q48 { return fixed.Q48MaxValue().Ceil() }, fixed.Q48MaxValue()},
		{"MaxValue.Round()", func() fixed.Q48 { return fixed.Q48MaxValue().Round() }, fixed.Q48MaxValue()},
		{"MulAdd16 overflow", func() fixed.Q48 {
			return fixed.Q48MaxValue().MulAdd16(fixed.Q16One(), fixed.Q16One())
		}, fixed.Q48MaxValue()},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fixed.ResetSaturationCount()
			if got := c.op(); !got.Eq(c.want) {
				t.Errorf("got %d, want %d", got.Raw(), c.want.Raw())
			}
			if n := fixed.SaturationCount(); n != 1 {
				t.Errorf("SaturationCount = %d, want 1", n)
			}
		})
	}
}

// TestQ48FromIntSaturatesAtTheIntegerBounds needs int arguments wider than
// 32 bits, so it runs only on 64-bit architectures.
func TestQ48FromIntSaturatesAtTheIntegerBounds(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("wide int inputs require a 64-bit architecture")
	}
	// A variable keeps the constants out of the 32-bit compile.
	bound := int64(1) << 47
	cases := []struct {
		name string
		op   func() fixed.Q48
		want fixed.Q48
	}{
		{"FromRatio overflow", func() fixed.Q48 { return fixed.Q48FromRatio(int(-bound), -1) }, fixed.Q48MaxValue()},
		{"FromInt overflow", func() fixed.Q48 { return fixed.Q48FromInt(int(bound)) }, fixed.Q48MaxValue()},
		{"FromInt underflow", func() fixed.Q48 { return fixed.Q48FromInt(int(-bound - 1)) }, fixed.Q48MinValue()},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fixed.ResetSaturationCount()
			if got := c.op(); !got.Eq(c.want) {
				t.Errorf("got %d, want %d", got.Raw(), c.want.Raw())
			}
			if n := fixed.SaturationCount(); n != 1 {
				t.Errorf("SaturationCount = %d, want 1", n)
			}
		})
	}
	fixed.ResetSaturationCount()
	if got := fixed.Q48FromInt(int(-bound)); !got.Eq(fixed.Q48MinValue()) {
		t.Errorf("Q48FromInt(-1<<47) = %d, want MinValue", got.Raw())
	}
	if c := fixed.SaturationCount(); c != 0 {
		t.Errorf("SaturationCount = %d, want 0: the exact bound is not an event", c)
	}
}

func TestQ48ExactBoundariesDoNotSaturate(t *testing.T) {
	fixed.ResetSaturationCount()
	if got := fixed.Q48MinValue().Sub(fixed.Q48MinValue()); !got.Eq(fixed.Q48Zero()) {
		t.Errorf("MinValue.Sub(MinValue) = %d, want Zero", got.Raw())
	}
	if got := fixed.Q48MinValue().Div(fixed.Q48One()); !got.Eq(fixed.Q48MinValue()) {
		t.Errorf("MinValue.Div(One) = %d, want MinValue", got.Raw())
	}
	if got := fixed.Q48MustParse("-140737488355328"); !got.Eq(fixed.Q48MinValue()) {
		t.Errorf("Q48MustParse(-2^47) = %d, want MinValue", got.Raw())
	}
	if c := fixed.SaturationCount(); c != 0 {
		t.Errorf("SaturationCount = %d, want 0: exact boundary results are not events", c)
	}
}

func TestQ48DomainErrorsPanic(t *testing.T) {
	t.Run("DivByZero", func(t *testing.T) {
		expectPanic(t, func() { fixed.Q48One().Div(fixed.Q48Zero()) })
	})
	t.Run("FromRatioZeroDenominator", func(t *testing.T) {
		expectPanic(t, func() { fixed.Q48FromRatio(1, 0) })
	})
	t.Run("SqrtOfNegative", func(t *testing.T) {
		expectPanic(t, func() { fixed.Q48FromInt(-1).Sqrt() })
	})
}

func TestQ48SqrtFloorsTheRoot(t *testing.T) {
	if got := fixed.Q48FromInt(4).Sqrt(); !got.Eq(fixed.Q48FromInt(2)) {
		t.Errorf("Sqrt(4) = %d, want 2", got.Raw())
	}
	if got := fixed.Q48FromInt(2).Sqrt().Raw(); got != 92681 {
		t.Errorf("Sqrt(2) = %d, want 92681", got)
	}
	// The radicand of MaxValue exceeds 64 bits.
	if got, want := fixed.Q48MaxValue().Sqrt().Raw(), oracleQ48Sqrt(math.MaxInt64); got != want {
		t.Errorf("Sqrt(MaxValue) = %d, oracle says %d", got, want)
	}
}

func TestQ48IntegerConversions(t *testing.T) {
	cases := []struct {
		num, den                  int
		floor, ceil, round, trunc int
	}{
		{5, 2, 2, 3, 3, 2},
		{-5, 2, -3, -2, -3, -2},
		{-1, 2, -1, 0, -1, 0},
	}
	for _, c := range cases {
		q := fixed.Q48FromRatio(c.num, c.den)
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

func TestQ48ComparisonsOrderValues(t *testing.T) {
	a, b := fixed.Q48FromInt(-2), fixed.Q48One()
	if a.Cmp(b) != -1 || b.Cmp(a) != 1 || a.Cmp(a) != 0 {
		t.Errorf("Cmp does not order %d and %d", a.Raw(), b.Raw())
	}
	if !a.Less(b) || !b.Greater(a) || a.Greater(a) {
		t.Errorf("Less and Greater disagree with the order of %d and %d", a.Raw(), b.Raw())
	}
	if !a.Min(b).Eq(a) || !a.Max(b).Eq(b) {
		t.Errorf("Min and Max disagree with the order of %d and %d", a.Raw(), b.Raw())
	}
	if got := fixed.Q48FromInt(9).Clamp(a, b); !got.Eq(b) {
		t.Errorf("Clamp(9) = %d, want the upper bound", got.Raw())
	}
}

// TestQ48ConversionsFollowTheGridLaw pins the conversion law: widening is
// exact, narrowing floors and saturates. Q16 and Q48 share one grid, so the
// Q16 round trip never moves a value.
func TestQ48ConversionsFollowTheGridLaw(t *testing.T) {
	fixed.ResetSaturationCount()
	for _, raw := range q16BoundaryRaws() {
		q := fixed.Q16FromRaw(raw)
		if got := q.ToQ48().ToQ16(); !got.Eq(q) {
			t.Errorf("Q16 round trip of raw %d = %d", raw, got.Raw())
		}
		if got, want := q.ToQ48().ToQ32(), q.ToQ32(); !got.Eq(want) {
			t.Errorf("Q16→Q48→Q32 of raw %d = %d, want %d", raw, got.Raw(), want.Raw())
		}
	}
	for _, q := range []fixed.Q32{fixed.Q32Zero(), fixed.Q32FromInt(-1), fixed.Q32MaxValue().Floor(), fixed.Q32MinValue()} {
		if got := q.ToQ48().ToQ32(); !got.Eq(q) {
			t.Errorf("Q32 round trip of %d = %d", q.Raw(), got.Raw())
		}
	}
	if c := fixed.SaturationCount(); c != 0 {
		t.Errorf("SaturationCount = %d, want 0: the round trips are exact", c)
	}

	// Narrowing Q32 floors: -2⁻³² lands one full Q48 step below zero.
	if got := fixed.Q32FromRaw(-1).ToQ48().Raw(); got != -1 {
		t.Errorf("ToQ48(-2⁻³²) = %d, want -1", got)
	}
	// Narrowing Q48 saturates one step past the Q16 and Q32 ranges.
	if got := fixed.Q48FromInt(1 << 15).ToQ16(); !got.Eq(fixed.Q16MaxValue()) {
		t.Errorf("ToQ16(2^15) = %d, want Q16MaxValue", got.Raw())
	}
	if got := q48Int(1 << 31).ToQ32(); !got.Eq(fixed.Q32MaxValue()) {
		t.Errorf("ToQ32(2^31) = %d, want Q32MaxValue", got.Raw())
	}
	if got := q48Int(-1<<31 - 1).ToQ32(); !got.Eq(fixed.Q32MinValue()) {
		t.Errorf("ToQ32(-2^31-1) = %d, want Q32MinValue", got.Raw())
	}
}

// TestQ48MulAdd16MatchesTheWidenedPath checks the accumulator against the
// widened Q48 multiply and against the Q16 multiply after narrowing.
func TestQ48MulAdd16MatchesTheWidenedPath(t *testing.T) {
	raws := q16BoundaryRaws()
	for _, a := range raws {
		for _, b := range raws {
			qa, qb := fixed.Q16FromRaw(a), fixed.Q16FromRaw(b)
			got := fixed.Q48Zero().MulAdd16(qa, qb)
			if want := qa.ToQ48().Mul(qb.ToQ48()); !got.Eq(want) {
				t.Fatalf("MulAdd16(%d, %d) = %d, widened Mul says %d", a, b, got.Raw(), want.Raw())
			}
			if want := qa.Mul(qb); !got.ToQ16().Eq(want) {
				t.Fatalf("MulAdd16(%d, %d).ToQ16() = %d, Q16 Mul says %d", a, b, got.ToQ16().Raw(), want.Raw())
			}
		}
	}
}

// TestQ48AccumulatesBeyondTheQ16Range is the reason the format exists: a sum
// of Q16 products that saturates in Q16 stays exact in Q48.
func TestQ48AccumulatesBeyondTheQ16Range(t *testing.T) {
	fixed.ResetSaturationCount()
	x := fixed.Q16FromInt(1 << 10)
	var acc fixed.Q48
	for range 1 << 12 {
		acc = acc.MulAdd16(x, x)
	}
	if want := q48Int(1 << 32); !acc.Eq(want) {
		t.Errorf("sum = %s, want %s", acc, want)
	}
	if c := fixed.SaturationCount(); c != 0 {
		t.Errorf("SaturationCount = %d, want 0", c)
	}
}

func FuzzQ48MulAdd16VsWidened(f *testing.F) {
	for _, v := range q16BoundaryRaws() {
		f.Add(v, v)
		f.Add(v, int32(1))
	}
	f.Fuzz(func(t *testing.T, a, b int32) {
		qa, qb := fixed.Q16FromRaw(a), fixed.Q16FromRaw(b)
		got := fixed.Q48Zero().MulAdd16(qa, qb)
		if want := qa.ToQ48().Mul(qb.ToQ48()); !got.Eq(want) {
			t.Errorf("MulAdd16(%d, %d) = %d, widened Mul says %d", a, b, got.Raw(), want.Raw())
		}
		if want := qa.Mul(qb); !got.ToQ16().Eq(want) {
			t.Errorf("MulAdd16(%d, %d).ToQ16() = %d, Q16 Mul says %d", a, b, got.ToQ16().Raw(), want.Raw())
		}
	})
}

func TestQ48StringFormatsExactDecimals(t *testing.T) {
	cases := []struct {
		q    fixed.Q48
		want string
	}{
		{fixed.Q48FromInt(3), "3"},
		{fixed.Q48FromRatio(7, 2), "3.5"},
		{fixed.Q48FromRaw(1), "0.0000152587890625"},
		{fixed.Q48MinValue(), "-140737488355328"},
		{fixed.Q48MaxValue(), "140737488355327.9999847412109375"},
	}
	for _, c := range cases {
		if got := c.q.String(); got != c.want {
			t.Errorf("String(%d) = %q, want %q", c.q.Raw(), got, c.want)
		}
	}
}

func TestQ48MustParseReadsDecimalLiterals(t *testing.T) {
	cases := []struct {
		in  string
		raw int64
	}{
		{"1.5", 98304},
		{"0.00000762939453125", 1}, // An exact half rounds away from zero.
		{"-0.00000762939453125", -1},
		{"0.00000762939453124", 0},
		{"140737488355327.9999847412109375", math.MaxInt64},
		{"-140737488355328", math.MinInt64},
	}
	for _, c := range cases {
		if got := fixed.Q48MustParse(c.in).Raw(); got != c.raw {
			t.Errorf("Q48MustParse(%q) = %d, want %d", c.in, got, c.raw)
		}
	}
	t.Run("malformed/6.", func(t *testing.T) {
		expectPanic(t, func() { fixed.Q48MustParse("6.") })
	})
}

func FuzzQ48TextRoundTrip(f *testing.F) {
	for _, raw := range fuzzSeeds() {
		f.Add(raw)
	}
	f.Fuzz(func(t *testing.T, raw int64) {
		q := fixed.Q48FromRaw(raw)
		if got := fixed.Q48MustParse(q.String()); !got.Eq(q) {
			t.Errorf("Q48MustParse(String(%d)) = %d", raw, got.Raw())
		}
	})
}
