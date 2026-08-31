package fixed_test

import (
	"math"
	"testing"

	"github.com/dhannyell/fixed"
)

func TestStringFormatsExactDecimals(t *testing.T) {
	cases := []struct {
		q    fixed.Q32
		want string
	}{
		{fixed.FromInt(3), "3"},
		{fixed.FromRatio(7, 2), "3.5"},
		{fixed.FromRatio(-1, 4), "-0.25"},
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
		{"-0.001", -4294967},                       // 0.001·2³² = 4294967.296.
		{"0.1", 429496730},                         // 0.1·2³² = 429496729.6.
		{"0.000000000116415321826934814453125", 1}, // An exact half rounds away from zero.
		{"-0.000000000116415321826934814453125", -1},
		{"0.000000000116415321826934814453124", 0},
		{"2147483648", math.MaxInt64},
		{"-2147483648", math.MinInt64},
	}
	for _, c := range cases {
		if got := fixed.MustParse(c.in).Raw(); got != c.raw {
			t.Errorf("MustParse(%q) = %d, want %d", c.in, got, c.raw)
		}
	}
	for _, bad := range []string{"", "6.", ".5", "+1", "1e3", "--1"} {
		t.Run("malformed/"+bad, func(t *testing.T) {
			expectPanic(t, func() { fixed.MustParse(bad) })
		})
	}
}

func FuzzTextRoundTrip(f *testing.F) {
	for _, raw := range fuzzSeeds() {
		f.Add(raw)
	}
	f.Fuzz(func(t *testing.T, raw int64) {
		q := fixed.FromRaw(raw)
		if got := fixed.MustParse(q.String()); !got.Eq(q) {
			t.Errorf("MustParse(String(%d)) = %d", raw, got.Raw())
		}
	})
}
