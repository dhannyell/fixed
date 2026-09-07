package fixed_test

import (
	"math"
	"testing"

	"github.com/dhannyell/fixed"
)

func FuzzLane16VsScalar(f *testing.F) {
	f.Add(int32(1), int32(-1), int32(0))
	f.Add(int32(math.MaxInt32), int32(1), int32(math.MaxInt32))
	f.Add(int32(math.MinInt32), int32(-1), int32(math.MinInt32))

	f.Fuzz(func(t *testing.T, seedA, seedB, seedC int32) {
		if !fixed.LanesAvailable() {
			t.Skip("lane operations require AVX2 in this build")
		}
		var a, b, c [fixed.LaneWidth]fixed.Q16
		ua, ub, uc := uint64(uint32(seedA)), uint64(uint32(seedB)), uint64(uint32(seedC))
		for lane := range fixed.LaneWidth {
			a[lane] = fixed.Q16FromRaw(int32(ua))
			b[lane] = fixed.Q16FromRaw(int32(ub))
			c[lane] = fixed.Q16FromRaw(int32(uc))
			ua = ua*laneStride + 1
			ub = ub*laneStride - 1
			uc = uc*laneStride + uint64(lane+3)
		}
		exerciseLane16Vector(t, a, b, c)
	})
}

func FuzzLane48VsScalar(f *testing.F) {
	f.Add(int64(1), int64(-1), int32(1), int32(-1))
	f.Add(int64(math.MaxInt64), int64(1), int32(1<<16), int32(1<<16))
	f.Add(int64(math.MinInt64), int64(-1), int32(math.MinInt32), int32(math.MaxInt32))

	f.Fuzz(func(t *testing.T, seedA, seedB int64, seedX, seedY int32) {
		if !fixed.LanesAvailable() {
			t.Skip("lane operations require AVX2 in this build")
		}
		var a, b [fixed.LaneWidth]fixed.Q48
		var x, y [fixed.LaneWidth]fixed.Q16
		ua, ub := uint64(seedA), uint64(seedB)
		ux, uy := uint64(uint32(seedX)), uint64(uint32(seedY))
		for lane := range fixed.LaneWidth {
			a[lane] = fixed.Q48FromRaw(int64(ua))
			b[lane] = fixed.Q48FromRaw(int64(ub))
			x[lane] = fixed.Q16FromRaw(int32(ux))
			y[lane] = fixed.Q16FromRaw(int32(uy))
			ua = ua*laneStride + 1
			ub = ub*laneStride - 1
			ux = ux*laneStride + uint64(lane+3)
			uy = uy*laneStride - uint64(lane+5)
		}
		exerciseLane48Vector(t, a, b, x, y)
	})
}
