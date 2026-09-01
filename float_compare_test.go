package fixed_test

import (
	"math"
	"testing"

	"github.com/dhannyell/fixed"
)

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
		var acc int64
		for i := range b.N {
			acc += int64(q[i&compareMask].Add(step).Raw())
		}
		benchSinkQ16 = acc
	})
	b.Run("float32", func(b *testing.B) {
		const step = float32(0x1p-16)
		var acc float32
		for i := range b.N {
			acc += f[i&compareMask] + step
		}
		benchSinkFloat32 = acc
	})
}

func BenchmarkCompareQ16Mul(b *testing.B) {
	q, f := compareQ16Inputs()
	b.Run("fixed", func(b *testing.B) {
		factor := fixed.Q16FromRatio(255, 256)
		var acc int64
		for i := range b.N {
			acc += int64(q[i&compareMask].Mul(factor).Raw())
		}
		benchSinkQ16 = acc
	})
	b.Run("float32", func(b *testing.B) {
		const factor = float32(255.0 / 256.0)
		var acc float32
		for i := range b.N {
			acc += f[i&compareMask] * factor
		}
		benchSinkFloat32 = acc
	})
}

func BenchmarkCompareQ16Div(b *testing.B) {
	q, f := compareQ16Inputs()
	b.Run("fixed", func(b *testing.B) {
		divisor := fixed.Q16FromInt(3)
		var acc int64
		for i := range b.N {
			acc += int64(q[i&compareMask].Div(divisor).Raw())
		}
		benchSinkQ16 = acc
	})
	b.Run("float32", func(b *testing.B) {
		var acc float32
		for i := range b.N {
			acc += f[i&compareMask] / 3
		}
		benchSinkFloat32 = acc
	})
}

func BenchmarkCompareQ16Sqrt(b *testing.B) {
	q, f := compareQ16Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc int64
		for i := range b.N {
			acc += int64(q[i&compareMask].Sqrt().Raw())
		}
		benchSinkQ16 = acc
	})
	b.Run("float32", func(b *testing.B) {
		var acc float32
		for i := range b.N {
			acc += float32(math.Sqrt(float64(f[i&compareMask])))
		}
		benchSinkFloat32 = acc
	})
}

func BenchmarkCompareQ32Add(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		step := fixed.Q32FromRaw(1)
		var acc int64
		for i := range b.N {
			acc += q[i&compareMask].Add(step).Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		const step = 0x1p-32
		var acc float64
		for i := range b.N {
			acc += f[i&compareMask] + step
		}
		benchSinkFloat64 = acc
	})
}

func BenchmarkCompareQ32Mul(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		factor := fixed.Q32FromRatio(255, 256)
		var acc int64
		for i := range b.N {
			acc += q[i&compareMask].Mul(factor).Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		const factor = 255.0 / 256.0
		var acc float64
		for i := range b.N {
			acc += f[i&compareMask] * factor
		}
		benchSinkFloat64 = acc
	})
}

func BenchmarkCompareQ32Div(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		divisor := fixed.Q32FromInt(3)
		var acc int64
		for i := range b.N {
			acc += q[i&compareMask].Div(divisor).Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		var acc float64
		for i := range b.N {
			acc += f[i&compareMask] / 3
		}
		benchSinkFloat64 = acc
	})
}

func BenchmarkCompareQ32Sqrt(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc int64
		for i := range b.N {
			acc += q[i&compareMask].Sqrt().Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		var acc float64
		for i := range b.N {
			acc += math.Sqrt(f[i&compareMask])
		}
		benchSinkFloat64 = acc
	})
}

func BenchmarkCompareVec2Len(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc int64
		for i := range b.N {
			j := i & compareMask
			acc += (fixed.Vec2{X: q[j], Y: q[(j*31)&compareMask]}).Len().Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		var acc float64
		for i := range b.N {
			j := i & compareMask
			x, y := f[j], f[(j*31)&compareMask]
			acc += math.Sqrt(x*x + y*y)
		}
		benchSinkFloat64 = acc
	})
}

func BenchmarkCompareVec2Normalize(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc int64
		for i := range b.N {
			j := i & compareMask
			n := (fixed.Vec2{X: q[j], Y: q[(j*31)&compareMask]}).Normalize()
			acc += n.X.Raw() + n.Y.Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		var acc float64
		for i := range b.N {
			j := i & compareMask
			x, y := f[j], f[(j*31)&compareMask]
			length := math.Sqrt(x*x + y*y)
			acc += x/length + y/length
		}
		benchSinkFloat64 = acc
	})
}

func BenchmarkCompareVec2NormalizeAxial(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc int64
		for i := range b.N {
			n := (fixed.Vec2{X: q[i&compareMask]}).Normalize()
			acc += n.X.Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		var acc float64
		for i := range b.N {
			x := f[i&compareMask]
			acc += x / math.Sqrt(x*x)
		}
		benchSinkFloat64 = acc
	})
}

func BenchmarkCompareRotNormalize(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc int64
		for i := range b.N {
			j := i & compareMask
			n := (fixed.Rot{Sin: q[j], Cos: q[(j*31)&compareMask]}).Normalize()
			acc += n.Sin.Raw() + n.Cos.Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		var acc float64
		for i := range b.N {
			j := i & compareMask
			sin, cos := f[j], f[(j*31)&compareMask]
			length := math.Sqrt(sin*sin + cos*cos)
			acc += sin/length + cos/length
		}
		benchSinkFloat64 = acc
	})
}

func BenchmarkCompareSinCosTurns(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc int64
		for i := range b.N {
			a := q[i&compareMask]
			acc += fixed.SinTurns(a).Raw() + fixed.CosTurns(a).Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		var acc float64
		for i := range b.N {
			sin, cos := math.Sincos(f[i&compareMask] * 2 * math.Pi)
			acc += sin + cos
		}
		benchSinkFloat64 = acc
	})
}

func BenchmarkCompareRotFromTurns(b *testing.B) {
	q, f := compareQ32Inputs()
	b.Run("fixed", func(b *testing.B) {
		var acc int64
		for i := range b.N {
			r := fixed.RotFromTurns(q[i&compareMask])
			acc += r.Sin.Raw() + r.Cos.Raw()
		}
		benchSinkQ32 = acc
	})
	b.Run("float64", func(b *testing.B) {
		var acc float64
		for i := range b.N {
			sin, cos := math.Sincos(f[i&compareMask] * 2 * math.Pi)
			acc += sin + cos
		}
		benchSinkFloat64 = acc
	})
}
