package fixed_test

import (
	"math"
	"testing"

	"github.com/dhannyell/fixed"
)

const workloadSize = 256

type workloadPoint struct{ x, y float64 }

var workloadFixedPoints [workloadSize]fixed.Vec2
var workloadFloatPoints [workloadSize]workloadPoint

func transformFixed(dst, src *[workloadSize]fixed.Vec2, r fixed.Rot, offset fixed.Vec2) {
	for i, p := range src {
		dst[i] = r.Apply(p).Add(offset)
	}
}

func transformFloat(dst, src *[workloadSize]workloadPoint, sin, cos float64, offset workloadPoint) {
	for i, p := range src {
		dst[i] = workloadPoint{cos*p.x - sin*p.y + offset.x, sin*p.x + cos*p.y + offset.y}
	}
}

func transformInputs() ([workloadSize]fixed.Vec2, [workloadSize]workloadPoint, fixed.Rot) {
	var q [workloadSize]fixed.Vec2
	var f [workloadSize]workloadPoint
	for i := range q {
		x, y := int64(i-128)<<28, int64((i*73)%256-128)<<28
		q[i] = fixed.Vec2{X: fixed.Q32FromRaw(x), Y: fixed.Q32FromRaw(y)}
		f[i] = workloadPoint{float64(x) * 0x1p-32, float64(y) * 0x1p-32}
	}
	return q, f, fixed.RotFromTurns(fixed.Q32FromRatio(1, 12))
}

func dotInputs() ([workloadSize]fixed.Q16, [workloadSize]fixed.Q16, [workloadSize]float32, [workloadSize]float32) {
	var a, b [workloadSize]fixed.Q16
	var fa, fb [workloadSize]float32
	for i := range a {
		x, y := int32(i-128)*257, int32((i*73)%256-128)*257
		a[i], b[i] = fixed.Q16FromRaw(x), fixed.Q16FromRaw(y)
		fa[i], fb[i] = float32(x)*0x1p-16, float32(y)*0x1p-16
	}
	return a, b, fa, fb
}

func dotFixed(a, b *[workloadSize]fixed.Q16) fixed.Q48 {
	acc := fixed.Q48Zero()
	for i := range a {
		acc = acc.MulAdd16(a[i], b[i])
	}
	return acc
}

func dotFloat(a, b *[workloadSize]float32) float64 {
	var acc float64
	for i := range a {
		acc += float64(a[i]) * float64(b[i])
	}
	return acc
}

// Each iteration transforms a complete block, including loads and stores.
// Rotation construction and input preparation are outside the timed region.
func BenchmarkWorkloadTransform256(b *testing.B) {
	q, f, r := transformInputs()
	offset := fixed.Vec2{X: fixed.Q32FromInt(3), Y: fixed.Q32FromInt(-2)}
	b.Run("fixed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			transformFixed(&workloadFixedPoints, &q, r, offset)
		}
	})
	b.Run("float64", func(b *testing.B) {
		sin, cos := float64(r.Sin.Raw())*0x1p-32, float64(r.Cos.Raw())*0x1p-32
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			transformFloat(&workloadFloatPoints, &f, sin, cos, workloadPoint{3, -2})
		}
	})
}

// The single accumulator is part of this reduction workload. One iteration
// reduces all 256 pairs; it is not a throughput measurement of MulAdd16 alone.
func BenchmarkWorkloadDot256(b *testing.B) {
	a, c, fa, fc := dotInputs()
	b.Run("fixed", func(b *testing.B) {
		var result fixed.Q48
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			result = dotFixed(&a, &c)
		}
		b.StopTimer()
		benchSinkQ48 = result.Raw()
	})
	b.Run("float64", func(b *testing.B) {
		var result float64
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			result = dotFloat(&fa, &fc)
		}
		b.StopTimer()
		benchSinkFloat64 = result
	})
}

func TestWorkloadKernels(t *testing.T) {
	q, f, r := transformInputs()
	var got [workloadSize]fixed.Vec2
	var want [workloadSize]workloadPoint
	transformFixed(&got, &q, r, fixed.Vec2{X: fixed.Q32FromInt(3), Y: fixed.Q32FromInt(-2)})
	transformFloat(&want, &f, float64(r.Sin.Raw())*0x1p-32, float64(r.Cos.Raw())*0x1p-32, workloadPoint{3, -2})
	for i := range got {
		// Two floored products contribute at most two raw units per component.
		if math.Abs(float64(got[i].X.Raw())*0x1p-32-want[i].x) > 2*0x1p-32 ||
			math.Abs(float64(got[i].Y.Raw())*0x1p-32-want[i].y) > 2*0x1p-32 {
			t.Fatalf("transform point %d exceeds rounding bound", i)
		}
	}
	a, b, fa, fb := dotInputs()
	// MulAdd16 floors each product to the Q48 grid before accumulation.
	if math.Abs(float64(dotFixed(&a, &b).Raw())*0x1p-16-dotFloat(&fa, &fb)) > workloadSize*0x1p-16 {
		t.Fatal("dot product exceeds accumulated rounding bound")
	}
}
