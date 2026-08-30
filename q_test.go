package fixed_test

import (
	"go/parser"
	"go/token"
	"math"
	"math/big"
	"os"
	"slices"
	"strings"
	"testing"

	fixed "github.com/dhannyell/fixed"
)

// Oracle helpers. Table tests and fuzz targets share them so both
// verify the same contract.

var (
	bigMin   = big.NewInt(math.MinInt64)
	bigMax   = big.NewInt(math.MaxInt64)
	bigScale = new(big.Int).Lsh(big.NewInt(1), 32)
)

func clampRaw(z *big.Int) int64 {
	if z.Cmp(bigMin) < 0 {
		return math.MinInt64
	}
	if z.Cmp(bigMax) > 0 {
		return math.MaxInt64
	}
	return z.Int64()
}

func oracleAdd(a, b int64) int64 {
	return clampRaw(new(big.Int).Add(big.NewInt(a), big.NewInt(b)))
}

func oracleSub(a, b int64) int64 {
	return clampRaw(new(big.Int).Sub(big.NewInt(a), big.NewInt(b)))
}

func oracleMul(a, b int64) int64 {
	z := new(big.Int).Mul(big.NewInt(a), big.NewInt(b))
	// Euclidean Div with a positive divisor floors, matching the shift.
	z.Div(z, bigScale)
	return clampRaw(z)
}

func oracleDiv(a, b int64) int64 {
	z := new(big.Int).Lsh(big.NewInt(a), 32)
	// Quo truncates toward zero, matching the magnitude division.
	z.Quo(z, big.NewInt(b))
	return clampRaw(z)
}

func oracleSqrt(a int64) int64 {
	z := new(big.Int).Lsh(big.NewInt(a), 32)
	return z.Sqrt(z).Int64()
}

func TestMulAndDivProduceExactBits(t *testing.T) {
	if got := fixed.One().Mul(fixed.One()); !got.Eq(fixed.One()) {
		t.Errorf("One.Mul(One) = %d, want One", got.Raw())
	}
	if got := fixed.Half().Mul(fixed.Half()).Raw(); got != 1<<30 {
		t.Errorf("Half.Mul(Half) = %d, want %d", got, int64(1)<<30)
	}
	if got := fixed.FromInt(3).Div(fixed.FromInt(2)).Raw(); got != 3<<31 {
		t.Errorf("3/2 = %d, want %d", got, int64(3)<<31)
	}
}

func TestMulFloorsTinyNegativeProducts(t *testing.T) {
	// The 128-bit arithmetic shift floors; the product must stay -2⁻⁶⁴,
	// not collapse to zero.
	if got := fixed.FromRaw(-1).Mul(fixed.FromRaw(1)).Raw(); got != -1 {
		t.Errorf("FromRaw(-1).Mul(FromRaw(1)) = %d, want -1", got)
	}
}

func TestDivTruncatesTowardZero(t *testing.T) {
	// Floor would give -1431655766; the vector pins the documented
	// Mul/Div asymmetry.
	if got := fixed.FromRatio(-1, 3).Raw(); got != -1431655765 {
		t.Errorf("FromRatio(-1, 3) = %d, want -1431655765", got)
	}
}

func TestSaturationClampsAndCounts(t *testing.T) {
	fixed.ResetSaturationCount()
	steps := []struct {
		name string
		op   func() fixed.Q
		want fixed.Q
	}{
		{"MaxValue.Add(One)", func() fixed.Q { return fixed.MaxValue().Add(fixed.One()) }, fixed.MaxValue()},
		{"MinValue.Sub(One)", func() fixed.Q { return fixed.MinValue().Sub(fixed.One()) }, fixed.MinValue()},
		{"2^20 * 2^20", func() fixed.Q { return fixed.FromInt(1 << 20).Mul(fixed.FromInt(1 << 20)) }, fixed.MaxValue()},
		{"MinValue.Neg()", func() fixed.Q { return fixed.MinValue().Neg() }, fixed.MaxValue()},
		{"MinValue.Abs()", func() fixed.Q { return fixed.MinValue().Abs() }, fixed.MaxValue()},
	}
	for i, s := range steps {
		if got := s.op(); !got.Eq(s.want) {
			t.Errorf("%s = %d, want %d", s.name, got.Raw(), s.want.Raw())
		}
		if c := fixed.SaturationCount(); c != uint64(i+1) {
			t.Errorf("after %s: SaturationCount = %d, want %d", s.name, c, i+1)
		}
	}
}

func TestExactBoundariesDoNotSaturate(t *testing.T) {
	fixed.ResetSaturationCount()
	if got := fixed.MinValue().Sub(fixed.MinValue()); !got.Eq(fixed.Zero()) {
		t.Errorf("MinValue.Sub(MinValue) = %d, want Zero", got.Raw())
	}
	if got := fixed.MinValue().Div(fixed.One()); !got.Eq(fixed.MinValue()) {
		t.Errorf("MinValue.Div(One) = %d, want MinValue", got.Raw())
	}
	if got := fixed.FromInt(-1 << 31); !got.Eq(fixed.MinValue()) {
		t.Errorf("FromInt(-1<<31) = %d, want MinValue", got.Raw())
	}
	if c := fixed.SaturationCount(); c != 0 {
		t.Errorf("SaturationCount = %d, want 0: exact boundary results are not events", c)
	}
}

func expectPanic(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Error("expected a panic")
		}
	}()
	f()
}

func TestDomainErrorsPanic(t *testing.T) {
	t.Run("DivByZero", func(t *testing.T) {
		expectPanic(t, func() { fixed.One().Div(fixed.Zero()) })
	})
	t.Run("FromRatioZeroDenominator", func(t *testing.T) {
		expectPanic(t, func() { fixed.FromRatio(1, 0) })
	})
	t.Run("SqrtOfNegative", func(t *testing.T) {
		expectPanic(t, func() { fixed.FromInt(-1).Sqrt() })
	})
}

func TestSqrtFloorsTheRoot(t *testing.T) {
	if got := fixed.FromInt(4).Sqrt(); !got.Eq(fixed.FromInt(2)) {
		t.Errorf("Sqrt(4) = %d, want 2", got.Raw())
	}
	if got := fixed.FromInt(2).Sqrt().Raw(); got != 0x16A09E667 {
		t.Errorf("Sqrt(2) = %#x, want 0x16A09E667", got)
	}
	if got := fixed.Zero().Sqrt(); !got.Eq(fixed.Zero()) {
		t.Errorf("Sqrt(0) = %d, want 0", got.Raw())
	}
	if got, want := fixed.MaxValue().Sqrt().Raw(), oracleSqrt(math.MaxInt64); got != want {
		t.Errorf("Sqrt(MaxValue) = %d, oracle says %d", got, want)
	}
}

func TestIntegerConversions(t *testing.T) {
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
		q := fixed.FromRatio(c.num, c.den)
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

func TestStringFormatsExactDecimals(t *testing.T) {
	cases := []struct {
		q    fixed.Q
		want string
	}{
		{fixed.FromInt(3), "3"},
		{fixed.FromRatio(7, 2), "3.5"},
		{fixed.FromRatio(-1, 4), "-0.25"},
		// The expansion of every Q32.32 terminates; String shows it whole.
		{fixed.FromRaw(1), "0.00000000023283064365386962890625"},
		{fixed.MinValue(), "-2147483648"},
		{fixed.MaxValue(), "2147483647.99999999976716935634613037109375"},
	}
	for _, c := range cases {
		if got := c.q.String(); got != c.want {
			t.Errorf("String(%d) = %q, want %q", c.q.Raw(), got, c.want)
		}
	}
}

func TestMustParseReadsDecimalLiterals(t *testing.T) {
	cases := []struct {
		in  string
		raw int64
	}{
		{"6.25", 26843545600},
		{"-0.001", -4294967}, // nearest: 0.001·2³² = 4294967.296
		{"0.1", 429496730},   // nearest rounds up: 0.1·2³² = 429496729.6
		{"2147483648", math.MaxInt64},
		{"-2147483648", math.MinInt64},
	}
	for _, c := range cases {
		if got := fixed.MustParse(c.in).Raw(); got != c.raw {
			t.Errorf("MustParse(%q) = %d, want %d", c.in, got, c.raw)
		}
	}
	for _, bad := range []string{"", "6.", ".5", "+1", "1e3", "--1", "0.12345678901234567890"} {
		t.Run("malformed "+bad, func(t *testing.T) {
			expectPanic(t, func() { fixed.MustParse(bad) })
		})
	}
}

// boundaryRaws returns the deduplicated boundary set of the plan:
// zeros, units, named values, power-of-two neighborhoods and the int64
// extremes.
func boundaryRaws() []int64 {
	const half, one = int64(1) << 31, int64(1) << 32
	set := map[int64]struct{}{}
	add := func(v int64) { set[v] = struct{}{} }
	for _, v := range []int64{
		0, 1, -1, 2, -2,
		half, -half, one, -one, one + 1, one - 1, -one + 1, -one - 1,
		math.MinInt64, math.MinInt64 + 1, math.MaxInt64, math.MaxInt64 - 1,
		0x55555555, -0x55555555,
	} {
		add(v)
	}
	for _, k := range []uint{16, 31, 32, 33, 47, 48, 62} {
		p := int64(1) << k
		for _, d := range []int64{-1, 0, 1} {
			add(p + d)
			add(-p + d)
		}
	}
	out := make([]int64, 0, len(set))
	for v := range set {
		out = append(out, v)
	}
	// A sorted set keeps failure output reproducible.
	slices.Sort(out)
	return out
}

func TestBoundaryCrossProductVsBig(t *testing.T) {
	raws := boundaryRaws()
	for _, a := range raws {
		for _, b := range raws {
			qa, qb := fixed.FromRaw(a), fixed.FromRaw(b)
			if got, want := qa.Add(qb).Raw(), oracleAdd(a, b); got != want {
				t.Fatalf("Add(%d, %d) = %d, oracle says %d", a, b, got, want)
			}
			if got, want := qa.Sub(qb).Raw(), oracleSub(a, b); got != want {
				t.Fatalf("Sub(%d, %d) = %d, oracle says %d", a, b, got, want)
			}
			if got, want := qa.Mul(qb).Raw(), oracleMul(a, b); got != want {
				t.Fatalf("Mul(%d, %d) = %d, oracle says %d", a, b, got, want)
			}
			if b != 0 {
				if got, want := qa.Div(qb).Raw(), oracleDiv(a, b); got != want {
					t.Fatalf("Div(%d, %d) = %d, oracle says %d", a, b, got, want)
				}
			}
		}
	}
}

// TestArrowRule is the machine guard of the import rule: the package
// compiles from the stdlib allowlist alone. Test files stay free.
func TestArrowRule(t *testing.T) {
	allow := map[string]bool{`"math/bits"`: true, `"sync/atomic"`: true}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), name, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imp := range file.Imports {
			if !allow[imp.Path.Value] {
				t.Errorf("%s imports %s: outside the allowlist", name, imp.Path.Value)
			}
		}
	}
}

// Differential fuzz. The oracle is the same math/big helper set the
// table tests use.

func fuzzSeeds() []int64 {
	return []int64{
		0, 1, -1,
		int64(1) << 31, -(int64(1) << 31),
		int64(1) << 32, -(int64(1) << 32),
		math.MinInt64, math.MaxInt64,
		0x55555555, -0x55555555,
		int64(1) << 48, -(int64(1) << 48),
	}
}

func FuzzAddSubVsBig(f *testing.F) {
	for _, v := range fuzzSeeds() {
		f.Add(v, v)
		f.Add(v, int64(1))
	}
	f.Fuzz(func(t *testing.T, a, b int64) {
		qa, qb := fixed.FromRaw(a), fixed.FromRaw(b)
		if got, want := qa.Add(qb).Raw(), oracleAdd(a, b); got != want {
			t.Errorf("Add(%d, %d) = %d, oracle says %d", a, b, got, want)
		}
		if got, want := qa.Sub(qb).Raw(), oracleSub(a, b); got != want {
			t.Errorf("Sub(%d, %d) = %d, oracle says %d", a, b, got, want)
		}
		// Commutativity holds bit-exactly; associativity does not, by design.
		if x, y := qa.Add(qb), qb.Add(qa); !x.Eq(y) {
			t.Errorf("Add(%d, %d) != Add(%d, %d)", a, b, b, a)
		}
	})
}

func FuzzMulVsBig(f *testing.F) {
	for _, v := range fuzzSeeds() {
		f.Add(v, v)
		f.Add(v, int64(1))
	}
	f.Fuzz(func(t *testing.T, a, b int64) {
		qa, qb := fixed.FromRaw(a), fixed.FromRaw(b)
		if got, want := qa.Mul(qb).Raw(), oracleMul(a, b); got != want {
			t.Errorf("Mul(%d, %d) = %d, oracle says %d", a, b, got, want)
		}
		if x, y := qa.Mul(qb), qb.Mul(qa); !x.Eq(y) {
			t.Errorf("Mul(%d, %d) != Mul(%d, %d)", a, b, b, a)
		}
	})
}

func FuzzDivVsBig(f *testing.F) {
	for _, v := range fuzzSeeds() {
		f.Add(v, v)
		f.Add(v, int64(1))
	}
	f.Fuzz(func(t *testing.T, a, b int64) {
		if b == 0 {
			t.Skip()
		}
		if got, want := fixed.FromRaw(a).Div(fixed.FromRaw(b)).Raw(), oracleDiv(a, b); got != want {
			t.Errorf("Div(%d, %d) = %d, oracle says %d", a, b, got, want)
		}
	})
}

func FuzzSqrtVsBig(f *testing.F) {
	for _, v := range fuzzSeeds() {
		f.Add(v)
	}
	f.Fuzz(func(t *testing.T, a int64) {
		if a < 0 {
			t.Skip()
		}
		if got, want := fixed.FromRaw(a).Sqrt().Raw(), oracleSqrt(a); got != want {
			t.Errorf("Sqrt(%d) = %d, oracle says %d", a, got, want)
		}
	})
}
