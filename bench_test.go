package fixed_test

import (
	"math"
	"testing"

	"github.com/dhannyell/fixed"
)

var benchSinkQ32 int64
var benchSinkQ16 int64

// BenchmarkSinCosTurns uses a non-sequential step to exercise table access.
func BenchmarkSinCosTurns(b *testing.B) {
	var acc int64
	u := int64(0)
	for range b.N {
		a := fixed.Q32FromRaw(u)
		acc += fixed.SinTurns(a).Raw() + fixed.CosTurns(a).Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkRotFromTurns(b *testing.B) {
	var acc int64
	u := int64(0)
	for range b.N {
		r := fixed.RotFromTurns(fixed.Q32FromRaw(u))
		acc += r.Sin.Raw() + r.Cos.Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkAtan2Turns(b *testing.B) {
	var acc int64
	u := int64(1)
	for range b.N {
		acc += fixed.Atan2Turns(fixed.Q32FromRaw(u), fixed.Q32FromRaw(u*31)).Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

// Throughput benchmarks accumulate into a sink, so iterations overlap in
// the pipeline. Latency benchmarks feed each result into the next call and
// measure the dependent cost one caller pays. Do not compare the two.

func BenchmarkQ32AddThroughput(b *testing.B) {
	step := fixed.Q32FromRaw(1)
	var acc int64
	u := int64(1)
	for range b.N {
		acc += fixed.Q32FromRaw(u).Add(step).Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkQ32AddLatency(b *testing.B) {
	// The step is one raw unit, so x stays far from saturation.
	step := fixed.Q32FromRaw(1)
	x := fixed.Q32Zero()
	for range b.N {
		x = x.Add(step)
	}
	benchSinkQ32 = x.Raw()
}

func BenchmarkQ32MulThroughput(b *testing.B) {
	factor := fixed.Q32FromRatio(255, 256)
	var acc int64
	u := int64(1)
	for range b.N {
		acc += fixed.Q32FromRaw(u).Mul(factor).Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkQ32MulLatency(b *testing.B) {
	// x converges to a fixed point near 256 and stays in domain.
	factor := fixed.Q32FromRatio(255, 256)
	one := fixed.Q32One()
	x := fixed.Q32FromInt(3)
	for range b.N {
		x = x.Mul(factor).Add(one)
	}
	benchSinkQ32 = x.Raw()
}

func BenchmarkQ32DivThroughput(b *testing.B) {
	divisor := fixed.Q32FromInt(3)
	var acc int64
	u := int64(1)
	for range b.N {
		acc += fixed.Q32FromRaw(u).Div(divisor).Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkQ32DivLatency(b *testing.B) {
	// The chain holds x at exactly 3: 3/1.5 + 1 = 3 in Q32.32.
	divisor := fixed.Q32FromRatio(3, 2)
	one := fixed.Q32One()
	x := fixed.Q32FromInt(3)
	for range b.N {
		x = x.Div(divisor).Add(one)
	}
	benchSinkQ32 = x.Raw()
}

func BenchmarkQ32SqrtThroughput(b *testing.B) {
	var acc int64
	u := int64(1)
	for range b.N {
		acc += fixed.Q32FromRaw(u & math.MaxInt64).Sqrt().Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkQ32SqrtLatency(b *testing.B) {
	// x stays near one million; the low bits vary so the operand is not
	// constant between iterations.
	scale := fixed.Q32FromInt(1000)
	x := fixed.Q32FromInt(1_000_000)
	u := int64(0)
	for range b.N {
		x = x.Sqrt().Mul(scale).Add(fixed.Q32FromRaw(u & 0xFFFF))
		u += 2654435761
	}
	benchSinkQ32 = x.Raw()
}

func BenchmarkQ16AddThroughput(b *testing.B) {
	step := fixed.Q16FromRaw(1)
	var acc int64
	u := int64(1)
	for range b.N {
		acc += int64(fixed.Q16FromRaw(int32(u)).Add(step).Raw())
		u += 2654435761
	}
	benchSinkQ16 = acc
}

func BenchmarkQ16AddLatency(b *testing.B) {
	step := fixed.Q16FromRaw(1)
	x := fixed.Q16Zero()
	for range b.N {
		x = x.Add(step)
	}
	benchSinkQ16 = int64(x.Raw())
}

func BenchmarkQ16MulThroughput(b *testing.B) {
	factor := fixed.Q16FromRatio(255, 256)
	var acc int64
	u := int64(1)
	for range b.N {
		acc += int64(fixed.Q16FromRaw(int32(u)).Mul(factor).Raw())
		u += 2654435761
	}
	benchSinkQ16 = acc
}

func BenchmarkQ16MulLatency(b *testing.B) {
	factor := fixed.Q16FromRatio(255, 256)
	one := fixed.Q16One()
	x := fixed.Q16FromInt(3)
	for range b.N {
		x = x.Mul(factor).Add(one)
	}
	benchSinkQ16 = int64(x.Raw())
}

func BenchmarkQ16DivThroughput(b *testing.B) {
	divisor := fixed.Q16FromInt(3)
	var acc int64
	u := int64(1)
	for range b.N {
		acc += int64(fixed.Q16FromRaw(int32(u)).Div(divisor).Raw())
		u += 2654435761
	}
	benchSinkQ16 = acc
}

func BenchmarkQ16DivLatency(b *testing.B) {
	divisor := fixed.Q16FromRatio(3, 2)
	one := fixed.Q16One()
	x := fixed.Q16FromInt(3)
	for range b.N {
		x = x.Div(divisor).Add(one)
	}
	benchSinkQ16 = int64(x.Raw())
}

func BenchmarkQ16SqrtThroughput(b *testing.B) {
	var acc int64
	u := int64(1)
	for range b.N {
		acc += int64(fixed.Q16FromRaw(int32(u) & math.MaxInt32).Sqrt().Raw())
		u += 2654435761
	}
	benchSinkQ16 = acc
}

func BenchmarkQ16SqrtLatency(b *testing.B) {
	scale := fixed.Q16FromInt(32)
	x := fixed.Q16FromInt(1000)
	u := int64(0)
	for range b.N {
		x = x.Sqrt().Mul(scale).Add(fixed.Q16FromRaw(int32(u) & 0xFF))
		u += 2654435761
	}
	benchSinkQ16 = int64(x.Raw())
}

func BenchmarkVec2DotThroughput(b *testing.B) {
	// The base is inside the unit box, so no product or sum saturates.
	base := fixed.Vec2{X: fixed.Q32FromRatio(255, 256), Y: fixed.Q32FromRatio(1, 256)}
	var acc int64
	u := int64(1)
	for range b.N {
		v := fixed.Vec2{X: fixed.Q32FromRaw(u), Y: fixed.Q32FromRaw(u * 31)}
		acc += v.Dot(base).Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkVec2DotLatency(b *testing.B) {
	// x converges to a fixed point near 1 and stays in domain.
	base := fixed.Vec2{X: fixed.Q32FromRatio(255, 256), Y: fixed.Q32FromRatio(1, 256)}
	v := fixed.Vec2{X: fixed.Q32FromInt(3), Y: fixed.Q32One()}
	for range b.N {
		v = fixed.Vec2{X: v.Dot(base), Y: fixed.Q32One()}
	}
	benchSinkQ32 = v.X.Raw()
}

func BenchmarkVec2LenThroughput(b *testing.B) {
	var acc int64
	u := int64(1)
	for range b.N {
		v := fixed.Vec2{X: fixed.Q32FromRaw(u), Y: fixed.Q32FromRaw(u * 31)}
		acc += v.Len().Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkVec2LenLatency(b *testing.B) {
	// A 3-4-5 triangle: Len rebuilds the same vector, so the chain is
	// dependent while the magnitude stays put.
	cx := fixed.Q32FromRatio(3, 5)
	cy := fixed.Q32FromRatio(4, 5)
	v := fixed.Vec2{X: fixed.Q32FromInt(300), Y: fixed.Q32FromInt(400)}
	for range b.N {
		l := v.Len()
		v = fixed.Vec2{X: l.Mul(cx), Y: l.Mul(cy)}
	}
	benchSinkQ32 = v.X.Raw()
}

func BenchmarkVec2NormalizeThroughput(b *testing.B) {
	var acc int64
	u := int64(1)
	for range b.N {
		v := fixed.Vec2{X: fixed.Q32FromRaw(u), Y: fixed.Q32FromRaw(u * 31)}
		n := v.Normalize()
		acc += n.X.Raw() + n.Y.Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkVec2NormalizeLatency(b *testing.B) {
	scale := fixed.Q32FromInt(300)
	v := fixed.Vec2{X: fixed.Q32FromInt(3), Y: fixed.Q32FromInt(4)}
	for range b.N {
		n := v.Normalize()
		v = fixed.Vec2{X: n.Y.Mul(scale), Y: n.X.Mul(scale)}
	}
	benchSinkQ32 = v.X.Raw()
}

func BenchmarkVec2NormalizeAxialThroughput(b *testing.B) {
	var acc int64
	u := int64(1)
	for range b.N {
		v := fixed.Vec2{X: fixed.Q32FromRaw(u), Y: fixed.Q32Zero()}
		n := v.Normalize()
		acc += n.X.Raw() + n.Y.Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkVec2NormalizeAxialLatency(b *testing.B) {
	v := fixed.Vec2{X: fixed.Q32FromInt(300), Y: fixed.Q32Zero()}
	for range b.N {
		v = fixed.Vec2{X: v.Normalize().X.Mul(fixed.Q32FromInt(300)), Y: fixed.Q32Zero()}
	}
	benchSinkQ32 = v.X.Raw()
}

func BenchmarkRotNormalizeThroughput(b *testing.B) {
	var acc int64
	u := int64(1)
	for range b.N {
		r := fixed.Rot{Sin: fixed.Q32FromRaw(u), Cos: fixed.Q32FromRaw(u * 31)}
		n := r.Normalize()
		acc += n.Sin.Raw() + n.Cos.Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkRotNormalizeLatency(b *testing.B) {
	scale := fixed.Q32FromInt(300)
	r := fixed.Rot{Sin: fixed.Q32FromInt(3), Cos: fixed.Q32FromInt(4)}
	for range b.N {
		n := r.Normalize()
		r = fixed.Rot{Sin: n.Cos.Mul(scale), Cos: n.Sin.Mul(scale)}
	}
	benchSinkQ32 = r.Sin.Raw()
}

func BenchmarkRotApplyThroughput(b *testing.B) {
	r := fixed.RotFromTurns(fixed.Q32FromRatio(1, 12))
	var acc int64
	u := int64(1)
	for range b.N {
		v := fixed.Vec2{X: fixed.Q32FromRaw(u), Y: fixed.Q32FromRaw(-u)}
		w := r.Apply(v)
		acc += w.X.Raw() + w.Y.Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkRotApplyLatency(b *testing.B) {
	// Rotation preserves the length and each floor can only shrink it,
	// so v stays in domain.
	r := fixed.RotFromTurns(fixed.Q32FromRatio(1, 12))
	v := fixed.Vec2{X: fixed.Q32One(), Y: fixed.Q32Half()}
	for range b.N {
		v = r.Apply(v)
	}
	benchSinkQ32 = v.X.Raw() + v.Y.Raw()
}

func BenchmarkRotMulThroughput(b *testing.B) {
	step := fixed.RotFromTurns(fixed.Q32FromRatio(1, 12))
	var acc int64
	u := int64(1)
	for range b.N {
		r := fixed.Rot{Sin: fixed.Q32FromRaw(u), Cos: fixed.Q32FromRaw(-u)}
		n := r.Mul(step)
		acc += n.Sin.Raw() + n.Cos.Raw()
		u += 2654435761
	}
	benchSinkQ32 = acc
}

func BenchmarkRotMulLatency(b *testing.B) {
	// Unit rotations compose into unit rotations; floors only shrink.
	step := fixed.RotFromTurns(fixed.Q32FromRatio(1, 12))
	r := fixed.RotIdentity()
	for range b.N {
		r = r.Mul(step)
	}
	benchSinkQ32 = r.Sin.Raw() + r.Cos.Raw()
}
