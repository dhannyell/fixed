package fixed_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/dhannyell/fixed"
)

const (
	laneRandomVectors = 10_000
	laneStride        = uint64(6364136223846793005)
)

func TestLanePath(t *testing.T) {
	path := fixed.LanePath()
	t.Logf("path: %s width %d", path, fixed.LaneWidth)

	wantWidth := 4
	switch path {
	case "avx2":
		wantWidth = 8
	case "neon", "generic":
	default:
		t.Fatalf("LanePath() = %q, want avx2, neon, or generic", path)
	}
	if fixed.LaneWidth != wantWidth {
		t.Errorf("LaneWidth = %d on %s, want %d", fixed.LaneWidth, path, wantWidth)
	}
	if path != "avx2" && !fixed.LanesAvailable() {
		t.Errorf("LanesAvailable() = false on %s", path)
	}
	if want := os.Getenv("FIXED_BATCH_PATH"); want != "" && path != want {
		t.Errorf("LanePath() = %q, FIXED_BATCH_PATH demands %q", path, want)
	}
}

func TestLaneMethodsMatchScalar(t *testing.T) {
	if !fixed.LanesAvailable() {
		t.Skip("lane operations require AVX2 in this build")
	}

	rng := rand.New(rand.NewSource(0x16_48_cafe))
	for vector := range laneRandomVectors {
		a, b, c, q, r := laneTestInputs(vector, rng)
		exerciseLane16Vector(t, a, b, c)
		exerciseLane48Vector(t, q, r, b, c)
	}
}

func laneTestInputs(vector int, rng *rand.Rand) (
	a, b, c [fixed.LaneWidth]fixed.Q16,
	q, r [fixed.LaneWidth]fixed.Q48,
) {
	q16Edges := [...]int32{
		math.MinInt32, math.MinInt32 + 1, -1 << 16, -1, 0,
		1, 1 << 16, math.MaxInt32 - 1, math.MaxInt32,
	}
	q48Edges := [...]int64{
		math.MinInt64, math.MinInt64 + 1, -1 << 32, -1, 0,
		1, 1 << 32, math.MaxInt64 - 1, math.MaxInt64,
	}

	for lane := range fixed.LaneWidth {
		aRaw := int32(rng.Uint32())
		bRaw := int32(rng.Uint32())
		cRaw := int32(rng.Uint32())
		qRaw := int64(rng.Uint64())
		rRaw := int64(rng.Uint64())
		if vector < len(q16Edges)*len(q16Edges) {
			index := vector*fixed.LaneWidth + lane
			aRaw = q16Edges[index%len(q16Edges)]
			bRaw = q16Edges[index/len(q16Edges)%len(q16Edges)]
			cRaw = q16Edges[(index*5+3)%len(q16Edges)]
		}
		if vector < len(q48Edges)*len(q48Edges) {
			index := vector*fixed.LaneWidth + lane
			qRaw = q48Edges[index%len(q48Edges)]
			rRaw = q48Edges[index/len(q48Edges)%len(q48Edges)]
		}
		a[lane] = fixed.Q16FromRaw(aRaw)
		b[lane] = fixed.Q16FromRaw(bRaw)
		c[lane] = fixed.Q16FromRaw(cRaw)
		q[lane] = fixed.Q48FromRaw(qRaw)
		r[lane] = fixed.Q48FromRaw(rRaw)
	}

	// These lanes force the Q48 accumulator and Q16 product boundaries.
	q[0] = fixed.Q48MaxValue()
	b[0], c[0] = fixed.Q16One(), fixed.Q16One()
	if fixed.LaneWidth > 1 {
		q[1] = fixed.Q48MinValue()
		b[1], c[1] = fixed.Q16One(), fixed.Q16FromInt(-1)
	}
	return a, b, c, q, r
}

func exerciseLane16Vector(
	t *testing.T,
	a, b, c [fixed.LaneWidth]fixed.Q16,
) {
	t.Helper()
	la, lb, lc := fixed.LoadLane16(&a), fixed.LoadLane16(&b), fixed.LoadLane16(&c)

	checkLane16(t, "load/store", func() fixed.Lane16 { return la }, func(i int) fixed.Q16 { return a[i] })
	checkLane16(t, "splat", func() fixed.Lane16 { return fixed.SplatLane16(a[0]) }, func(int) fixed.Q16 { return a[0] })
	checkLane16(t, "add", func() fixed.Lane16 { return la.Add(lb) }, func(i int) fixed.Q16 { return a[i].Add(b[i]) })
	checkLane16(t, "sub", func() fixed.Lane16 { return la.Sub(lb) }, func(i int) fixed.Q16 { return a[i].Sub(b[i]) })
	checkLane16(t, "mul", func() fixed.Lane16 { return la.Mul(lb) }, func(i int) fixed.Q16 { return a[i].Mul(b[i]) })
	checkLane16(t, "mulround", func() fixed.Lane16 { return la.MulRound(lb) }, func(i int) fixed.Q16 {
		return a[i].MulRound(b[i])
	})
	checkLane16(t, "muladd", func() fixed.Lane16 { return la.MulAdd(lb, lc) }, func(i int) fixed.Q16 {
		return a[i].Add(b[i].Mul(c[i]))
	})
	checkLane16(t, "mulsub", func() fixed.Lane16 { return la.MulSub(lb, lc) }, func(i int) fixed.Q16 {
		return a[i].Sub(b[i].Mul(c[i]))
	})
	checkLane16(t, "min", func() fixed.Lane16 { return la.Min(lb) }, func(i int) fixed.Q16 { return a[i].Min(b[i]) })
	checkLane16(t, "max", func() fixed.Lane16 { return la.Max(lb) }, func(i int) fixed.Q16 { return a[i].Max(b[i]) })
	checkLane16(t, "symclamp", func() fixed.Lane16 { return la.SymClamp(lb) }, func(i int) fixed.Q16 {
		return a[i].Min(b[i]).Max(b[i].Neg())
	})
	checkLane16(t, "greater/blend", func() fixed.Lane16 {
		return fixed.BlendLane16(la.Greater(lb), la, lb)
	}, func(i int) fixed.Q16 {
		if a[i].Greater(b[i]) {
			return a[i]
		}
		return b[i]
	})
	checkLane16(t, "equals/blend", func() fixed.Lane16 {
		return fixed.BlendLane16(la.Equals(lb), la, lc)
	}, func(i int) fixed.Q16 {
		if a[i].Eq(b[i]) {
			return a[i]
		}
		return c[i]
	})
	checkLane16(t, "mask-or", func() fixed.Lane16 {
		mask := la.Greater(lb).Or(la.Equals(lb))
		return fixed.BlendLane16(mask, la, lb)
	}, func(i int) fixed.Q16 {
		if a[i].Greater(b[i]) || a[i].Eq(b[i]) {
			return a[i]
		}
		return b[i]
	})
	checkLane48(t, "to-lane48", func() fixed.Lane48 { return la.ToLane48() }, func(i int) fixed.Q48 {
		return a[i].ToQ48()
	})

	fixed.ResetSaturationCount()
	if !la.Greater(la).AllZero() {
		t.Fatal("Greater self mask is not all zero")
	}
	if la.Equals(la).AllZero() {
		t.Fatal("Equals self mask is all zero")
	}
	if got := fixed.SaturationCount(); got != 0 {
		t.Fatalf("mask operations recorded %d saturation events", got)
	}
}

func exerciseLane48Vector(
	t *testing.T,
	a, b [fixed.LaneWidth]fixed.Q48,
	x, y [fixed.LaneWidth]fixed.Q16,
) {
	t.Helper()
	la, lb := fixed.LoadLane48(&a), fixed.LoadLane48(&b)
	lx, ly := fixed.LoadLane16(&x), fixed.LoadLane16(&y)

	checkLane48(t, "load/store", func() fixed.Lane48 { return la }, func(i int) fixed.Q48 { return a[i] })
	checkLane48(t, "splat", func() fixed.Lane48 { return fixed.SplatLane48(a[0]) }, func(int) fixed.Q48 { return a[0] })
	checkLane48(t, "add", func() fixed.Lane48 { return la.Add(lb) }, func(i int) fixed.Q48 { return a[i].Add(b[i]) })
	checkLane48(t, "sub", func() fixed.Lane48 { return la.Sub(lb) }, func(i int) fixed.Q48 { return a[i].Sub(b[i]) })
	checkLane48(t, "muladd16", func() fixed.Lane48 { return la.MulAdd16(lx, ly) }, func(i int) fixed.Q48 {
		return a[i].MulAdd16(x[i], y[i])
	})
	checkLane48(t, "muladd16round", func() fixed.Lane48 { return la.MulAdd16Round(lx, ly) }, func(i int) fixed.Q48 {
		product := (int64(x[i].Raw())*int64(y[i].Raw()) + 1<<15) >> 16
		return a[i].Add(fixed.Q48FromRaw(product))
	})
	checkLane16(t, "to-lane16", func() fixed.Lane16 { return la.ToLane16() }, func(i int) fixed.Q16 {
		return a[i].ToQ16()
	})

}

func checkLane16(
	t *testing.T,
	name string,
	laneOp func() fixed.Lane16,
	scalarOp func(int) fixed.Q16,
) {
	t.Helper()
	fixed.ResetSaturationCount()
	gotLane := laneOp()
	gotEvents := fixed.SaturationCount()

	fixed.ResetSaturationCount()
	var want [fixed.LaneWidth]fixed.Q16
	for lane := range fixed.LaneWidth {
		want[lane] = scalarOp(lane)
	}
	wantEvents := fixed.SaturationCount()

	var got [fixed.LaneWidth]fixed.Q16
	gotLane.Store(&got)
	for lane := range fixed.LaneWidth {
		if !got[lane].Eq(want[lane]) {
			t.Fatalf("%s lane %d = %d, scalar says %d", name, lane, got[lane].Raw(), want[lane].Raw())
		}
	}
	if fixed.SaturationCountingEnabled && gotEvents != wantEvents {
		t.Fatalf("%s recorded %d saturation events, scalar recorded %d", name, gotEvents, wantEvents)
	}
}

func checkLane48(
	t *testing.T,
	name string,
	laneOp func() fixed.Lane48,
	scalarOp func(int) fixed.Q48,
) {
	t.Helper()
	fixed.ResetSaturationCount()
	gotLane := laneOp()
	gotEvents := fixed.SaturationCount()

	fixed.ResetSaturationCount()
	var want [fixed.LaneWidth]fixed.Q48
	for lane := range fixed.LaneWidth {
		want[lane] = scalarOp(lane)
	}
	wantEvents := fixed.SaturationCount()

	var got [fixed.LaneWidth]fixed.Q48
	gotLane.Store(&got)
	for lane := range fixed.LaneWidth {
		if !got[lane].Eq(want[lane]) {
			t.Fatalf("%s lane %d = %d, scalar says %d", name, lane, got[lane].Raw(), want[lane].Raw())
		}
	}
	if fixed.SaturationCountingEnabled && gotEvents != wantEvents {
		t.Fatalf("%s recorded %d saturation events, scalar recorded %d", name, gotEvents, wantEvents)
	}
}

func TestLaneScaleDownMatchesMulByPowerOfTwo(t *testing.T) {
	if !fixed.LanesAvailable() {
		t.Skip("lane operations require AVX2 in this build")
	}

	// The values cover both signs, the Q16 extremes, and the small magnitudes
	// where a truncating shift could part company with a rounding multiply.
	raws := []int32{0, 1, -1, 2, -2, 3, -3, 2457, -2457, 65535, -65535,
		1 << 20, -(1 << 20), fixed.Q16MaxValue().Raw(), fixed.Q16MinValue().Raw()}

	var values [fixed.LaneWidth]fixed.Q16
	var amounts [fixed.LaneWidth]uint8

	for base := range raws {
		for lane := range fixed.LaneWidth {
			values[lane] = fixed.Q16FromRaw(raws[(base+lane)%len(raws)])
			// Every lane takes its own amount, so a uniform shift would fail
			// this test. The amounts stay within the grid, at 16 and below.
			amounts[lane] = uint8((base + lane*3) % 17)
		}
		a := fixed.LoadLane16(&values)
		s := fixed.LoadShift16(&amounts)

		checkLane16(t, fmt.Sprintf("ScaleDown from raws[%d]", base),
			func() fixed.Lane16 { return a.ScaleDown(s) },
			func(lane int) fixed.Q16 {
				return values[lane].Mul(fixed.Q16FromRaw(int32(1) << 16 >> amounts[lane]))
			})

		// The rounded form answers to MulRound, tie rule included.
		checkLane16(t, fmt.Sprintf("ScaleDownRound from raws[%d]", base),
			func() fixed.Lane16 { return a.ScaleDownRound(s) },
			func(lane int) fixed.Q16 {
				return values[lane].MulRound(fixed.Q16FromRaw(int32(1) << 16 >> amounts[lane]))
			})
	}
}

func TestLaneScaleDownRoundBreaksTiesUpward(t *testing.T) {
	if !fixed.LanesAvailable() {
		t.Skip("lane operations require AVX2 in this build")
	}

	// A tie is a value whose discarded part is exactly one half. Both signs
	// must move toward positive infinity, which is where the two forms part:
	// ScaleDown takes the floor of the same input.
	cases := []struct {
		raw              int32
		amount           uint8
		wantRound, wantD int32
	}{
		{1, 1, 1, 0},
		{-1, 1, 0, -1},
		{3, 1, 2, 1},
		{-3, 1, -1, -2},
		{5, 2, 1, 1},
		{-5, 2, -1, -2},
		{-6, 2, -1, -2},
		{1 << 15, 16, 1, 0},
		{0, 5, 0, 0},
		{-2457, 0, -2457, -2457},
	}

	for _, c := range cases {
		var round, down [fixed.LaneWidth]fixed.Q16
		a := fixed.SplatLane16(fixed.Q16FromRaw(c.raw))
		s := fixed.SplatShift16(c.amount)
		a.ScaleDownRound(s).Store(&round)
		a.ScaleDown(s).Store(&down)
		// Every lane carries the same value and amount, so a lane that
		// disagrees means the shift read the wrong count for it.
		for lane := range fixed.LaneWidth {
			if round[lane].Raw() != c.wantRound || down[lane].Raw() != c.wantD {
				t.Errorf("raw %d by %d lane %d: round %d down %d, want round %d down %d",
					c.raw, c.amount, lane, round[lane].Raw(), down[lane].Raw(), c.wantRound, c.wantD)
			}
		}
	}
}

func TestLaneScaleDownPastTheQ16Grid(t *testing.T) {
	if !fixed.LanesAvailable() {
		t.Skip("lane operations require AVX2 in this build")
	}

	// Past 16 the descale leaves the Q16 grid, so ScaleDown stops tracking Mul:
	// it keeps shifting where Mul would give zero. An amount above 31 acts
	// like 31, which is what the constructors store.
	cases := []struct {
		raw             int32
		amount          uint8
		want, wantRound int32
	}{
		{1 << 20, 20, 1, 1},
		{1 << 20, 31, 0, 0},
		{1 << 20, 200, 0, 0},
		{-1, 17, -1, 0},
		{-1, 200, -1, 0},
		{fixed.Q16MinValue().Raw(), 31, -1, -1},
	}

	for _, c := range cases {
		var got, round [fixed.LaneWidth]fixed.Q16
		a := fixed.SplatLane16(fixed.Q16FromRaw(c.raw))
		s := fixed.SplatShift16(c.amount)
		a.ScaleDown(s).Store(&got)
		a.ScaleDownRound(s).Store(&round)
		for lane := range fixed.LaneWidth {
			if got[lane].Raw() != c.want || round[lane].Raw() != c.wantRound {
				t.Errorf("raw %d by %d lane %d: down %d round %d, want down %d round %d",
					c.raw, c.amount, lane, got[lane].Raw(), round[lane].Raw(), c.want, c.wantRound)
			}
		}
	}

	// LoadShift16 clamps on its own path, and the clamp is what stops a large
	// amount from becoming a left shift on NEON. Alternate 31 with an amount
	// past it: both stand for the same shift, so every lane must agree.
	var amounts [fixed.LaneWidth]uint8
	for lane := range fixed.LaneWidth {
		amounts[lane] = uint8(31 + lane%2*200)
	}
	var got [fixed.LaneWidth]fixed.Q16
	fixed.SplatLane16(fixed.Q16FromRaw(-1)).
		ScaleDownRound(fixed.LoadShift16(&amounts)).Store(&got)
	for lane := range fixed.LaneWidth {
		if got[lane].Raw() != 0 {
			t.Errorf("LoadShift16 lane %d by %d = %d, want 0", lane, amounts[lane], got[lane].Raw())
		}
	}
}
