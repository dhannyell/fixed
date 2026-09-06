package fixed

import "testing"

func referenceUnitPair(x, y int64) (Q32, Q32) {
	mx, my := magnitude(x), magnitude(y)
	scale := max(mx, my)
	if scale == 0 {
		return Q32{}, Q32{}
	}
	sx := divMag(mx, scale, x < 0)
	sy := divMag(my, scale, y < 0)
	n := sx.Mul(sx).Add(sy.Mul(sy)).Sqrt()
	return sx.Div(n), sy.Div(n)
}

func TestUnitPairPreservesReferenceBits(t *testing.T) {
	cases := [][2]int64{
		{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1},
		{1, 1}, {-1, 1}, {1, -1}, {-1, -1},
		{1, 2}, {-2, 1}, {1 << 32, 1}, {1, 1 << 32},
		{q32RawMin, q32RawMax}, {q32RawMax, q32RawMin},
		{q32RawMin, 0}, {0, q32RawMin}, {q32RawMax, q32RawMax},
		{q32RawMax, 0}, {0, q32RawMax}, {0, -q32RawMax}, {-q32RawMax, 0},
		{0x55555555, -0x12345678}, {-0x7fffffffffffffff, 0x102030405060708},
	}
	for _, c := range cases {
		before := SaturationCount()
		gotX, gotY := unitPair(c[0], c[1])
		gotEvents := SaturationCount() - before
		before = SaturationCount()
		wantX, wantY := referenceUnitPair(c[0], c[1])
		wantEvents := SaturationCount() - before
		if gotX != wantX || gotY != wantY {
			t.Errorf("unitPair(%d, %d) = (%d, %d), want (%d, %d)", c[0], c[1], gotX.raw, gotY.raw, wantX.raw, wantY.raw)
		}
		if gotEvents != wantEvents {
			t.Errorf("unitPair(%d, %d) recorded %d events, want %d", c[0], c[1], gotEvents, wantEvents)
		}
	}
}
