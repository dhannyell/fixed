package fixed

import (
	"strconv"
	"testing"
)

// benchSinkBatch drains batch results so the compiler keeps the work.
var benchSinkBatch int64

// benchSizes spans one cache line's worth of elements, an L1-resident run and
// a run that must stream from memory.
var benchSizes = []int{64, 1024, 65536}

// benchInputs builds operands with the house stride, narrowed so no element
// saturates. The probe measures the hot path; a workload that saturates often
// changes the scalar cost and must be measured with its own profile.
func benchInputs(n int) (a, b []Q16) {
	a, b = make([]Q16, n), make([]Q16, n)
	u := uint32(1)
	next := func() int32 {
		u = u*batchStride + 1
		return int32(u) >> 10
	}
	for i := range n {
		a[i] = Q16{raw: next()}
		b[i] = Q16{raw: next()}
	}
	return a, b
}

func drain(dst []Q16) {
	if len(dst) != 0 {
		benchSinkBatch += int64(dst[0].raw) + int64(dst[len(dst)-1].raw)
	}
}

// withKernels runs fn with the package dispatch table pointed at one kernel
// per operation, so the exported functions carry the full call cost of that
// path: the length check, the indirect call and the counter update.
func withKernels(k batchKernels, fn func()) {
	saved := kernels
	kernels = k
	defer func() { kernels = saved }()
	fn()
}

// runBatch names one benchmark case as op/impl/n so benchstat can group it.
func runBatch(b *testing.B, op, impl string, n int, run func()) {
	b.Run("op="+op+"/impl="+impl+"/n="+strconv.Itoa(n), func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		b.ResetTimer()
		for range b.N {
			run()
		}
	})
}

func BenchmarkBatch(b *testing.B) {
	for _, n := range benchSizes {
		a, y := benchInputs(n)
		dst := make([]Q16, n)

		runBatch(b, "add", "percall", n, func() {
			for i := range a {
				dst[i] = a[i].Add(y[i])
			}
			drain(dst)
		})
		for _, k := range addKernels {
			table := kernels
			table.add = k.fn
			withKernels(table, func() {
				runBatch(b, "add", k.name, n, func() {
					BatchAdd16(dst, a, y)
					drain(dst)
				})
			})
		}
		runBatch(b, "sub", "percall", n, func() {
			for i := range a {
				dst[i] = a[i].Sub(y[i])
			}
			drain(dst)
		})
		for _, k := range subKernels {
			table := kernels
			table.sub = k.fn
			withKernels(table, func() {
				runBatch(b, "sub", k.name, n, func() {
					BatchSub16(dst, a, y)
					drain(dst)
				})
			})
		}

		runBatch(b, "mul", "percall", n, func() {
			for i := range a {
				dst[i] = a[i].Mul(y[i])
			}
			drain(dst)
		})
		for _, k := range mulKernels {
			table := kernels
			table.mul = k.fn
			withKernels(table, func() {
				runBatch(b, "mul", k.name, n, func() {
					BatchMul16(dst, a, y)
					drain(dst)
				})
			})
		}

		lo, hi := Q16{raw: -q16RawOne}, Q16{raw: q16RawOne}
		runBatch(b, "clamp", "percall", n, func() {
			for i := range a {
				dst[i] = a[i].Clamp(lo, hi)
			}
			drain(dst)
		})
		for _, k := range clampKernels {
			table := kernels
			table.clamp = k.fn
			withKernels(table, func() {
				runBatch(b, "clamp", k.name, n, func() {
					BatchClamp16(dst, a, lo, hi)
					drain(dst)
				})
			})
		}

		benchmarkConversions(b, n, a, dst)
	}
}

// benchmarkConversions measures the format boundary. Each element moves 8
// bytes on one side and 4 on the other, so a case touches 12 bytes.
func benchmarkConversions(b *testing.B, n int, a, dst []Q16) {
	wide := make([]Q32, n)
	for i := range wide {
		wide[i] = a[i].ToQ32()
	}
	wideDst := make([]Q32, n)

	b.Run("op=q32fromq16/impl=percall/n="+strconv.Itoa(n), func(b *testing.B) {
		b.SetBytes(int64(n) * 12)
		for range b.N {
			for i := range a {
				wideDst[i] = a[i].ToQ32()
			}
			benchSinkBatch += wideDst[n-1].raw
		}
	})
	for _, k := range widenKernels {
		table := kernels
		table.q32FromQ16 = k.fn
		withKernels(table, func() {
			b.Run("op=q32fromq16/impl="+k.name+"/n="+strconv.Itoa(n), func(b *testing.B) {
				b.SetBytes(int64(n) * 12)
				for range b.N {
					BatchQ32FromQ16(wideDst, a)
					benchSinkBatch += wideDst[n-1].raw
				}
			})
		})
	}

	b.Run("op=q16fromq32/impl=percall/n="+strconv.Itoa(n), func(b *testing.B) {
		b.SetBytes(int64(n) * 12)
		for range b.N {
			for i := range wide {
				dst[i] = wide[i].ToQ16()
			}
			drain(dst)
		}
	})
	for _, k := range narrowKernels {
		table := kernels
		table.q16FromQ32 = k.fn
		withKernels(table, func() {
			b.Run("op=q16fromq32/impl="+k.name+"/n="+strconv.Itoa(n), func(b *testing.B) {
				b.SetBytes(int64(n) * 12)
				for range b.N {
					BatchQ16FromQ32(dst, wide)
					drain(dst)
				}
			})
		})
	}
}
