package fixed_test

import (
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
		b[1], c[1] = fixed.Q16One(), fixed.Q16One()
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
