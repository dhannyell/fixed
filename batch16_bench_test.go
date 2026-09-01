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
			runBatch(b, "add", k.name, n, func() {
				k.fn(dst, a, y)
				drain(dst)
			})
		}
		for _, k := range wrapKernels {
			runBatch(b, "addwrap", k.name, n, func() {
				k.fn(dst, a, y)
				drain(dst)
			})
		}

		runBatch(b, "mul", "percall", n, func() {
			for i := range a {
				dst[i] = a[i].Mul(y[i])
			}
			drain(dst)
		})
		for _, k := range mulKernels {
			runBatch(b, "mul", k.name, n, func() {
				k.fn(dst, a, y)
				drain(dst)
			})
		}
	}

	benchmarkAxpy(b)
}

func benchmarkAxpy(b *testing.B) {
	bias := Q16{raw: q16RawOne / 3}
	lo := Q16{raw: -8 * q16RawOne}
	hi := Q16{raw: 8 * q16RawOne}

	for _, n := range benchSizes {
		x, m, v := axpyInputs(n, 10)
		biasVec := make([]Q16, n)
		for i := range biasVec {
			biasVec[i] = bias
		}
		tmp := make([]Q16, n)

		runBatch(b, "axpy", "percall", n, func() {
			axpyClampPerCall(x, m, v, bias, lo, hi)
			drain(x)
		})
		runBatch(b, "axpy", "multipass", n, func() {
			axpyClampMultipass(x, m, v, biasVec, tmp, lo, hi)
			drain(x)
		})
		for _, f := range axpyForms {
			if f.name == "multipass" {
				// The registry form allocates its scratch; the case above
				// hoists it out of the measured loop, as the house rule asks.
				continue
			}
			runBatch(b, "axpy", f.name, n, func() {
				f.run(x, m, v, bias, lo, hi)
				drain(x)
			})
		}
	}
}
