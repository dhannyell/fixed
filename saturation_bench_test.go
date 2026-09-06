package fixed_test

import (
	"testing"

	"github.com/dhannyell/fixed"
)

// Run the same workloads with and without fixed_nosatcounter. Safe inputs
// distinguish diagnostic overhead from the arithmetic needed to clamp results.
func BenchmarkSaturationOverhead(b *testing.B) {
	for _, saturating := range []bool{false, true} {
		name := "safe"
		if saturating {
			name = "saturated"
		}
		b.Run(name, func(b *testing.B) {
			var a [256]fixed.Q32
			var x, y, dst [256]fixed.Q16
			for i := range a {
				a[i] = fixed.Q32FromRaw(int64(i))
				x[i], y[i] = fixed.Q16FromRaw(int32(i)), fixed.Q16One()
				if saturating {
					a[i] = fixed.Q32FromRaw(fixed.Q32MaxValue().Raw() - int64(i))
					x[i] = fixed.Q16FromRaw(fixed.Q16MaxValue().Raw() - int32(i))
				}
			}
			b.Run("Q32Add", func(b *testing.B) {
				step := fixed.Q32One()
				var s0, s1, s2, s3 int64
				b.ResetTimer()
				for i := range b.N {
					s0, s1, s2, s3 = s1, s2, s3, s0+a[i&255].Add(step).Raw()
				}
				b.StopTimer()
				benchSinkQ32 = s0 + s1 + s2 + s3
			})
			b.Run("BatchAdd16x256", func(b *testing.B) {
				b.ResetTimer()
				for range b.N {
					fixed.BatchAdd16(dst[:], x[:], y[:])
				}
				b.StopTimer()
				var sum int64
				for _, v := range dst {
					sum += int64(v.Raw())
				}
				benchSinkQ16 = sum
			})
		})
	}
}
