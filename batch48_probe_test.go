package fixed

// This file is a probe. It measures Q48 batch candidates that are not part of
// the API: the per-element accumulation and the Q16<->Q48 conversions. Every
// kernel here is unexported and lives in a test file. The scalar kernels
// define the bits; the vector kernels in the architecture files must match.

import (
	"strconv"
	"strings"
	"testing"
)

// mulAcc48Kernel computes acc[i] += a[i]*b[i] and returns the saturated count.
type mulAcc48Kernel struct {
	name string
	fn   func(acc []Q48, a, b []Q16) uint64
}

type widen48Kernel struct {
	name string
	fn   func(dst []Q48, a []Q16)
}

type narrow48Kernel struct {
	name string
	fn   func(dst []Q16, a []Q48) uint64
}

// The architecture files append their vector kernels when the CPU supports
// them.
var (
	mulAcc48Kernels = []mulAcc48Kernel{{"scalar", mulAcc48Scalar}}
	widen48Kernels  = []widen48Kernel{{"scalar", q48FromQ16Scalar}}
	narrow48Kernels = []narrow48Kernel{{"scalar", q16FromQ48Scalar}}
)

func mulAcc48Scalar(acc []Q48, a, b []Q16) uint64 {
	acc = acc[:len(a)]
	b = b[:len(a)]
	var events uint64
	for i := range a {
		p := (int64(a[i].raw) * int64(b[i].raw)) >> 16
		r, ovf := q48AddSat(acc[i].raw, p)
		if ovf {
			events++
		}
		acc[i] = Q48{raw: r}
	}
	return events
}

func q48FromQ16Scalar(dst []Q48, a []Q16) {
	dst = dst[:len(a)]
	for i := range a {
		dst[i] = Q48{raw: int64(a[i].raw)}
	}
}

func q16FromQ48Scalar(dst []Q16, a []Q48) uint64 {
	dst = dst[:len(a)]
	var events uint64
	for i := range a {
		v := a[i].raw
		r := int32(v)
		if int64(r) != v {
			r = q16SaturateWide(v)
			events++
		}
		dst[i] = Q16{raw: r}
	}
	return events
}

func TestProbe48MulAccMatchesTheScalarPath(t *testing.T) {
	a, b := batchPairs()
	for _, k := range mulAcc48Kernels[1:] {
		if strings.HasSuffix(k.name, "wrap") {
			continue // A wrap kernel is a measurement, not a candidate.
		}
		for _, n := range batch48Sizes {
			if n > len(a) {
				continue
			}
			want := batch48Accs(n)
			got := batch48Accs(n)
			wantEvents := mulAcc48Scalar(want, a[:n], b[:n])
			gotEvents := k.fn(got, a[:n], b[:n])
			for i := range n {
				if got[i] != want[i] {
					t.Fatalf("%s n=%d: element %d = %d, scalar says %d", k.name, n, i, got[i].raw, want[i].raw)
				}
			}
			if gotEvents != wantEvents {
				t.Fatalf("%s n=%d: events = %d, scalar says %d", k.name, n, gotEvents, wantEvents)
			}
		}
	}
	// Away from saturation every kernel, wrap included, must agree.
	x, y := benchInputs(169)
	want := make([]Q48, len(x))
	mulAcc48Scalar(want, x, y)
	for _, k := range mulAcc48Kernels {
		got := make([]Q48, len(x))
		if e := k.fn(got, x, y); e != 0 {
			t.Fatalf("%s: %d events on non-saturating input", k.name, e)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s: element %d = %d, scalar says %d", k.name, i, got[i].raw, want[i].raw)
			}
		}
	}
}

func TestProbe48ConversionsMatchTheScalarPath(t *testing.T) {
	a, _ := batchPairs()
	wide := batch48Accs(len(a))
	for _, n := range batch48Sizes {
		for _, k := range widen48Kernels[1:] {
			want, got := make([]Q48, n), make([]Q48, n)
			q48FromQ16Scalar(want, a[:n])
			k.fn(got, a[:n])
			for i := range n {
				if got[i] != want[i] {
					t.Fatalf("%s n=%d: element %d = %d, want %d", k.name, n, i, got[i].raw, want[i].raw)
				}
			}
		}
		for _, k := range narrow48Kernels[1:] {
			want, got := make([]Q16, n), make([]Q16, n)
			wantEvents := q16FromQ48Scalar(want, wide[:n])
			gotEvents := k.fn(got, wide[:n])
			for i := range n {
				if got[i] != want[i] {
					t.Fatalf("%s n=%d: element %d = %d, want %d", k.name, n, i, got[i].raw, want[i].raw)
				}
			}
			if gotEvents != wantEvents {
				t.Fatalf("%s n=%d: events = %d, want %d", k.name, n, gotEvents, wantEvents)
			}
		}
	}
}

func runProbe48(b *testing.B, op, impl string, n, bytesPerElem int, run func()) {
	b.Run("op="+op+"/impl="+impl+"/n="+strconv.Itoa(n), func(b *testing.B) {
		b.SetBytes(int64(n) * int64(bytesPerElem))
		b.ResetTimer()
		for range b.N {
			run()
		}
	})
}

// BenchmarkProbe48 is the measurement matrix. mulacc touches two Q16 reads
// and one Q48 read-modify-write per element.
func BenchmarkProbe48(b *testing.B) {
	for _, n := range benchSizes {
		a, y := benchInputs(n)
		acc := make([]Q48, n)
		var sink int64

		runProbe48(b, "mulacc", "percall", n, 16, func() {
			for i := range a {
				acc[i] = acc[i].MulAdd16(a[i], y[i])
			}
			sink += acc[len(acc)-1].raw
		})
		for _, k := range mulAcc48Kernels {
			runProbe48(b, "mulacc", k.name, n, 16, func() {
				if e := k.fn(acc, a, y); e != 0 {
					saturationEvents.Add(e)
				}
				sink += acc[len(acc)-1].raw
			})
		}

		wide := make([]Q48, n)
		runProbe48(b, "q48fromq16", "percall", n, 12, func() {
			for i := range a {
				wide[i] = a[i].ToQ48()
			}
			sink += wide[n-1].raw
		})
		for _, k := range widen48Kernels {
			runProbe48(b, "q48fromq16", k.name, n, 12, func() {
				k.fn(wide, a)
				sink += wide[n-1].raw
			})
		}
		narrow := make([]Q16, n)
		runProbe48(b, "q16fromq48", "percall", n, 12, func() {
			for i := range wide {
				narrow[i] = wide[i].ToQ16()
			}
			sink += int64(narrow[n-1].raw)
		})
		for _, k := range narrow48Kernels {
			runProbe48(b, "q16fromq48", k.name, n, 12, func() {
				if e := k.fn(narrow, wide); e != 0 {
					saturationEvents.Add(e)
				}
				sink += int64(narrow[n-1].raw)
			})
		}
		benchSinkBatch += sink
	}
}
