package fixed_test

import (
	"math"
	"testing"

	"github.com/dhannyell/fixed"
)

var benchSink int64

// BenchmarkSinCosTurns uses a non-sequential step to exercise table access.
func BenchmarkSinCosTurns(b *testing.B) {
	var acc int64
	u := int64(0)
	for range b.N {
		a := fixed.FromRaw(u)
		acc += fixed.SinTurns(a).Raw() + fixed.CosTurns(a).Raw()
		u += 2654435761
	}
	benchSink = acc
}

// Throughput benchmarks accumulate into a sink, so iterations overlap in
// the pipeline. Latency benchmarks feed each result into the next call and
// measure the dependent cost one caller pays. Do not compare the two.

func BenchmarkScalarAddThroughput(b *testing.B) {
	step := fixed.FromRaw(1)
	var acc int64
	u := int64(1)
	for range b.N {
		acc += fixed.FromRaw(u).Add(step).Raw()
		u += 2654435761
	}
	benchSink = acc
}

func BenchmarkScalarAddLatency(b *testing.B) {
	// The step is one raw unit, so x stays far from saturation.
	step := fixed.FromRaw(1)
	x := fixed.Zero()
	for range b.N {
		x = x.Add(step)
	}
	benchSink = x.Raw()
}

func BenchmarkScalarMulThroughput(b *testing.B) {
	factor := fixed.FromRatio(255, 256)
	var acc int64
	u := int64(1)
	for range b.N {
		acc += fixed.FromRaw(u).Mul(factor).Raw()
		u += 2654435761
	}
	benchSink = acc
}

func BenchmarkScalarMulLatency(b *testing.B) {
	// x converges to a fixed point near 256 and stays in domain.
	factor := fixed.FromRatio(255, 256)
	one := fixed.One()
	x := fixed.FromInt(3)
	for range b.N {
		x = x.Mul(factor).Add(one)
	}
	benchSink = x.Raw()
}

func BenchmarkScalarDivThroughput(b *testing.B) {
	divisor := fixed.FromInt(3)
	var acc int64
	u := int64(1)
	for range b.N {
		acc += fixed.FromRaw(u).Div(divisor).Raw()
		u += 2654435761
	}
	benchSink = acc
}

func BenchmarkScalarDivLatency(b *testing.B) {
	// The chain holds x at exactly 3: 3/1.5 + 1 = 3 in Q32.32.
	divisor := fixed.FromRatio(3, 2)
	one := fixed.One()
	x := fixed.FromInt(3)
	for range b.N {
		x = x.Div(divisor).Add(one)
	}
	benchSink = x.Raw()
}

func BenchmarkScalarSqrtThroughput(b *testing.B) {
	var acc int64
	u := int64(1)
	for range b.N {
		acc += fixed.FromRaw(u & math.MaxInt64).Sqrt().Raw()
		u += 2654435761
	}
	benchSink = acc
}

func BenchmarkScalarSqrtLatency(b *testing.B) {
	// x stays near one million; the low bits vary so the operand is not
	// constant between iterations.
	scale := fixed.FromInt(1000)
	x := fixed.FromInt(1_000_000)
	u := int64(0)
	for range b.N {
		x = x.Sqrt().Mul(scale).Add(fixed.FromRaw(u & 0xFFFF))
		u += 2654435761
	}
	benchSink = x.Raw()
}

func BenchmarkVec2LenThroughput(b *testing.B) {
	var acc int64
	u := int64(1)
	for range b.N {
		v := fixed.Vec2{X: fixed.FromRaw(u), Y: fixed.FromRaw(u * 31)}
		acc += v.Len().Raw()
		u += 2654435761
	}
	benchSink = acc
}

func BenchmarkVec2LenLatency(b *testing.B) {
	// A 3-4-5 triangle: Len rebuilds the same vector, so the chain is
	// dependent while the magnitude stays put.
	cx := fixed.FromRatio(3, 5)
	cy := fixed.FromRatio(4, 5)
	v := fixed.Vec2{X: fixed.FromInt(300), Y: fixed.FromInt(400)}
	for range b.N {
		l := v.Len()
		v = fixed.Vec2{X: l.Mul(cx), Y: l.Mul(cy)}
	}
	benchSink = v.X.Raw()
}
