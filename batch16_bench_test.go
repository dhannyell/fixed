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

// benchInputs builds deterministic, non-saturating operands for hot-path
// measurements. Saturating workloads need separate measurements.
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
		benchmarkBatch48(b, n, a, y)
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

// benchmarkBatch48 measures the Q48 batch functions. dot16 has no store;
// q48mul16 reads a Q48 and a Q16 and writes a Q48 per element.
func benchmarkBatch48(b *testing.B, n int, a, y []Q16) {
	var sink int64
	runBatch(b, "dot16", "percall", n, func() {
		var d Q48
		for i := range a {
			d = d.MulAdd16(a[i], y[i])
		}
		sink += d.raw
	})
	for _, k := range dot16Kernels {
		table := kernels
		table.dot16 = k.fn
		withKernels(table, func() {
			runBatch(b, "dot16", k.name, n, func() {
				sink += BatchDot16(a, y).raw
			})
		})
	}

	q, f := benchMul16Inputs(n)
	prod := make([]Q48, n)
	runBatch(b, "q48mul16", "percall", n, func() {
		for i := range q {
			prod[i] = q[i].Mul16(f[i])
		}
		sink += prod[n-1].raw
	})
	for _, k := range q48Mul16Kernels {
		table := kernels
		table.q48Mul16 = k.fn
		withKernels(table, func() {
			runBatch(b, "q48mul16", k.name, n, func() {
				BatchQ48Mul16(prod, q, f)
				sink += prod[n-1].raw
			})
		})
	}
	benchSinkBatch += sink
}
