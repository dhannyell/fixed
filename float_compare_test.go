package fixed_test

import (
	"math"
	"testing"

	"github.com/dhannyell/fixed"
)

// Four rotating accumulators keep the result sink from forming one serial
// reduction. Each iteration still evaluates one operation; final reduction
// happens after the timed loop. Loop and sink overhead remain in ns/op.
const compareMask = 255

var benchSinkFloat32 float32
var benchSinkFloat64 float64

func compareQ16Inputs() ([compareMask + 1]fixed.Q16, [compareMask + 1]float32) {
	var q [compareMask + 1]fixed.Q16
	var f [compareMask + 1]float32
	for i := range q {
		raw := int32((i + 1) * 7919)
		q[i] = fixed.Q16FromRaw(raw)
		f[i] = float32(raw) * 0x1p-16
	}
	return q, f
}

func compareQ32Inputs() ([compareMask + 1]fixed.Q32, [compareMask + 1]float64) {
	var q [compareMask + 1]fixed.Q32
	var f [compareMask + 1]float64
	for i := range q {
		raw := int64(i+1) * 2654435761
		q[i] = fixed.Q32FromRaw(raw)
		f[i] = float64(raw) * 0x1p-32
	}
	return q, f
}

func BenchmarkCompareQ16Add(b *testing.B) {
	q, f := compareQ16Inputs()
	b.Run("fixed", func(b *testing.B) {
		step := fixed.Q16FromRaw(1)
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(int64(q[i&compareMask].Add(step).Raw()))
		}
		b.StopTimer()
		benchSinkQ16 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float32", func(b *testing.B) {
		const step = float32(0x1p-16)
		var acc0, acc1, acc2, acc3 float32
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[i&compareMask]+step)
		}
		b.StopTimer()
		benchSinkFloat32 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ16Mul(b *testing.B) {
	q, f := compareQ16Inputs()
	b.Run("fixed", func(b *testing.B) {
		factor := fixed.Q16FromRatio(255, 256)
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(int64(q[i&compareMask].Mul(factor).Raw()))
		}
		b.StopTimer()
		benchSinkQ16 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float32", func(b *testing.B) {
		const factor = float32(255.0 / 256.0)
		var acc0, acc1, acc2, acc3 float32
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[i&compareMask]*factor)
		}
		b.StopTimer()
		benchSinkFloat32 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ16Div(b *testing.B) {
	q, f := compareQ16Inputs()
	b.Run("fixed", func(b *testing.B) {
		divisor := fixed.Q16FromInt(3)
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(int64(q[i&compareMask].Div(divisor).Raw()))
		}
		b.StopTimer()
		benchSinkQ16 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float32", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float32
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[i&compareMask]/3)
		}
		b.StopTimer()
		benchSinkFloat32 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ16Sqrt(b *testing.B) {
	q, f := compareQ16Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(int64(q[i&compareMask].Sqrt().Raw()))
		}
		b.StopTimer()
		benchSinkQ16 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float32", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float32
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(float32(math.Sqrt(float64(f[i&compareMask]))))
		}
		b.StopTimer()
		benchSinkFloat32 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ32Add(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		step := fixed.Q32FromRaw(1)
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(q[i&compareMask].Add(step).Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		const step = 0x1p-32
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[i&compareMask]+step)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ32Mul(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		factor := fixed.Q32FromRatio(255, 256)
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(q[i&compareMask].Mul(factor).Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		const factor = 255.0 / 256.0
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[i&compareMask]*factor)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ32Div(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		divisor := fixed.Q32FromInt(3)
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(q[i&compareMask].Div(divisor).Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[i&compareMask]/3)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ32Sqrt(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(q[i&compareMask].Sqrt().Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(math.Sqrt(f[i&compareMask]))
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareVec2Len(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			j := i & compareMask
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+((fixed.Vec2{X: q[j], Y: q[(j*31)&compareMask]}).Len().Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			j := i & compareMask
			x, y := f[j], f[(j*31)&compareMask]
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(math.Sqrt(x*x+y*y))
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareVec2Normalize(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			j := i & compareMask
			n := (fixed.Vec2{X: q[j], Y: q[(j*31)&compareMask]}).Normalize()
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(n.X.Raw()+n.Y.Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			j := i & compareMask
			x, y := f[j], f[(j*31)&compareMask]
			length := math.Sqrt(x*x + y*y)
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(x/length+y/length)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareVec2NormalizeAxial(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			n := (fixed.Vec2{X: q[i&compareMask]}).Normalize()
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(n.X.Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			x := f[i&compareMask]
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(x/math.Sqrt(x*x))
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareVec2Dot(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		base := fixed.Vec2{X: fixed.Q32FromRatio(255, 256), Y: fixed.Q32FromRatio(1, 256)}
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			j := i & compareMask
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+((fixed.Vec2{X: q[j], Y: q[(j*31)&compareMask]}).Dot(base).Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		const bx, by = 255.0 / 256.0, 1.0 / 256.0
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			j := i & compareMask
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[j]*bx+f[(j*31)&compareMask]*by)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareRotApply(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		r := fixed.RotFromTurns(fixed.Q32FromRatio(1, 12))
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			j := i & compareMask
			w := r.Apply(fixed.Vec2{X: q[j], Y: q[(j*31)&compareMask]})
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(w.X.Raw()+w.Y.Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		sin, cos := math.Sincos(2 * math.Pi / 12)
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			j := i & compareMask
			x, y := f[j], f[(j*31)&compareMask]
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(cos*x-sin*y+sin*x+cos*y)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareRotMul(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		step := fixed.RotFromTurns(fixed.Q32FromRatio(1, 12))
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			j := i & compareMask
			n := (fixed.Rot{Sin: q[j], Cos: q[(j*31)&compareMask]}).Mul(step)
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(n.Sin.Raw()+n.Cos.Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		sin2, cos2 := math.Sincos(2 * math.Pi / 12)
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			j := i & compareMask
			sin1, cos1 := f[j], f[(j*31)&compareMask]
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(sin1*cos2+cos1*sin2+cos1*cos2-sin1*sin2)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareRotNormalize(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			j := i & compareMask
			n := (fixed.Rot{Sin: q[j], Cos: q[(j*31)&compareMask]}).Normalize()
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(n.Sin.Raw()+n.Cos.Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			j := i & compareMask
			sin, cos := f[j], f[(j*31)&compareMask]
			length := math.Sqrt(sin*sin + cos*cos)
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(sin/length+cos/length)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareSinCosTurns(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			a := q[i&compareMask]
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(fixed.SinTurns(a).Raw()+fixed.CosTurns(a).Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			sin, cos := math.Sincos(f[i&compareMask] * 2 * math.Pi)
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(sin+cos)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareRotFromTurns(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			r := fixed.RotFromTurns(q[i&compareMask])
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(r.Sin.Raw()+r.Cos.Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			sin, cos := math.Sincos(f[i&compareMask] * 2 * math.Pi)
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(sin+cos)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareAtan2Turns(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			j := i & compareMask
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(fixed.Atan2Turns(q[j], q[(j*31)&compareMask]).Raw())
		}
		b.StopTimer()
		benchSinkQ32 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		const invTau = 1 / (2 * math.Pi)
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			j := i & compareMask
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(math.Atan2(f[j], f[(j*31)&compareMask])*invTau)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func compareQ48Inputs() ([compareMask + 1]fixed.Q48, [compareMask + 1]float64) {
	var q [compareMask + 1]fixed.Q48
	var f [compareMask + 1]float64
	for i := range q {
		raw := int64(i+1) * 2654435761
		q[i] = fixed.Q48FromRaw(raw)
		f[i] = float64(raw) * 0x1p-16
	}
	return q, f
}

func BenchmarkCompareQ48Add(b *testing.B) {
	q, f := compareQ48Inputs()
	b.Run("fixed", func(b *testing.B) {
		step := fixed.Q48FromRaw(1)
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(q[i&compareMask].Add(step).Raw())
		}
		b.StopTimer()
		benchSinkQ48 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		const step = 0x1p-16
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[i&compareMask]+step)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ48Mul(b *testing.B) {
	q, f := compareQ48Inputs()
	b.Run("fixed", func(b *testing.B) {
		factor := fixed.Q48FromRatio(255, 256)
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(q[i&compareMask].Mul(factor).Raw())
		}
		b.StopTimer()
		benchSinkQ48 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		const factor = 255.0 / 256.0
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[i&compareMask]*factor)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

// BenchmarkCompareQ48MulAdd16 pairs the accumulator with a float64 sum of
// float32 products, the shape a float solver would use for the same dot
// product.
func BenchmarkCompareQ48MulAdd16(b *testing.B) {
	q, f := compareQ16Inputs()
	b.Run("fixed", func(b *testing.B) {
		factor := fixed.Q16FromRatio(255, 256)
		var acc0, acc1, acc2, acc3 fixed.Q48
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0.MulAdd16(q[i&compareMask], factor)
		}
		b.StopTimer()
		benchSinkQ48 = acc0.Raw() + acc1.Raw() + acc2.Raw() + acc3.Raw()
	})
	b.Run("float64", func(b *testing.B) {
		const factor = float32(255.0 / 256.0)
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(float64(f[i&compareMask]*factor))
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ48Div(b *testing.B) {
	q, f := compareQ48Inputs()
	b.Run("fixed", func(b *testing.B) {
		divisor := fixed.Q48FromInt(3)
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(q[i&compareMask].Div(divisor).Raw())
		}
		b.StopTimer()
		benchSinkQ48 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(f[i&compareMask]/3)
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}

func BenchmarkCompareQ48Sqrt(b *testing.B) {
	q, f := compareQ48Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 int64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(q[i&compareMask].Sqrt().Raw())
		}
		b.StopTimer()
		benchSinkQ48 = acc0 + acc1 + acc2 + acc3
	})
	b.Run("float64", func(b *testing.B) {
		var acc0, acc1, acc2, acc3 float64
		for i := range b.N {
			acc0, acc1, acc2, acc3 = acc1, acc2, acc3, acc0+(math.Sqrt(f[i&compareMask]))
		}
		b.StopTimer()
		benchSinkFloat64 = acc0 + acc1 + acc2 + acc3
	})
}
