package fixed

import "testing"

type dot16Kernel struct {
	name string
	fn   func(a, b []Q16) (Q48, uint64)
}

type q48Mul16Kernel struct {
	name string
	fn   func(dst, q []Q48, f []Q16) uint64
}

// The architecture files append their vector kernels when the CPU supports
// them.
var (
	dot16Kernels    = []dot16Kernel{{"scalar", dot16Scalar}}
	q48Mul16Kernels = []q48Mul16Kernel{{"scalar", q48Mul16Scalar}}
)

// batch48Sizes covers empty input, sub-block input, exact blocks and tails.
var batch48Sizes = []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 64, 169}

// batch48Accs returns Q48 values near both saturation edges and around zero,
// with low words of both signs, so every carry and saturation path runs.
func batch48Accs(n int) []Q48 {
	seeds := []int64{0, 1, -1, q48RawMax, q48RawMin, q48RawMax - 1<<31, q48RawMin + 1<<31, 1 << 40, -1 << 40, 1<<47 | 0x8000_0000}
	acc := make([]Q48, n)
	for i := range acc {
		acc[i] = Q48{raw: seeds[i%len(seeds)]}
	}
	return acc
}

// benchMul16Inputs widens the Q16 bench values by 2^20 so the Q48 operand
// uses its high word, while the products stay inside the range.
func benchMul16Inputs(n int) (q []Q48, f []Q16) {
	a, f := benchInputs(n)
	q = make([]Q48, n)
	for i := range a {
		q[i] = Q48{raw: int64(a[i].raw) << 20}
	}
	return q, f
}

func TestBatchDot16KernelsMatchTheCanonicalOrder(t *testing.T) {
	a, b := batchPairs()
	for _, k := range dot16Kernels[1:] {
		for _, n := range batch48Sizes {
			if n > len(a) {
				continue
			}
			want, wantEvents := dot16Scalar(a[:n], b[:n])
			got, gotEvents := k.fn(a[:n], b[:n])
			if got != want || gotEvents != wantEvents {
				t.Fatalf("%s n=%d: (%d, %d events), scalar says (%d, %d events)", k.name, n, got.raw, gotEvents, want.raw, wantEvents)
			}
		}
	}
	// The pair grid saturates; a plain run must not, and must equal the
	// per-call accumulator, which is the order a caller would write by hand.
	x, y := benchInputs(169)
	var acc Q48
	for i := range x {
		acc = acc.MulAdd16(x[i], y[i])
	}
	for _, k := range dot16Kernels {
		got, events := k.fn(x, y)
		if got != acc || events != 0 {
			t.Fatalf("%s: unsaturated dot = %d (%d events), per-call says %d", k.name, got.raw, events, acc.raw)
		}
	}
}

func TestBatchQ48Mul16KernelsMatchTheScalarPath(t *testing.T) {
	f, _ := batchPairs()
	q := batch48Accs(len(f))
	for _, k := range q48Mul16Kernels[1:] {
		for _, n := range batch48Sizes {
			if n > len(f) {
				continue
			}
			want, got := make([]Q48, n), make([]Q48, n)
			wantEvents := q48Mul16Scalar(want, q[:n], f[:n])
			gotEvents := k.fn(got, q[:n], f[:n])
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
	// Every kernel must equal the per-call Mul16, and Mul16 must equal Mul
	// on the widened factor, on the saturating grid.
	for i := range f {
		want := q[i].Mul(f[i].ToQ48())
		if got := q[i].Mul16(f[i]); got != want {
			t.Fatalf("Mul16(%d, %d) = %d, Mul says %d", q[i].raw, f[i].raw, got.raw, want.raw)
		}
	}
	for _, k := range q48Mul16Kernels {
		got := make([]Q48, len(f))
		k.fn(got, q, f)
		for i := range got {
			if want := q[i].Mul16(f[i]); got[i] != want {
				t.Fatalf("%s: element %d = %d, per-call says %d", k.name, i, got[i].raw, want.raw)
			}
		}
	}
}

// TestBatch48ExportedCallsCountAndCheck covers what the wrappers add: the
// length check, the counter update and in-place operation.
func TestBatch48ExportedCallsCountAndCheck(t *testing.T) {
	a, b := batchPairs()
	q := batch48Accs(len(a))

	before := SaturationCount()
	_, wantDot := dot16Scalar(a, b)
	got := BatchDot16(a, b)
	if want, _ := dot16Scalar(a, b); got != want {
		t.Fatalf("BatchDot16 = %d, scalar says %d", got.raw, want.raw)
	}
	if d := SaturationCount() - before; d != wantDot {
		t.Fatalf("BatchDot16 added %d events, want %d", d, wantDot)
	}

	before = SaturationCount()
	want := make([]Q48, len(q))
	wantEvents := q48Mul16Scalar(want, q, a)
	inPlace := append([]Q48(nil), q...)
	BatchQ48Mul16(inPlace, inPlace, a)
	for i := range want {
		if inPlace[i] != want[i] {
			t.Fatalf("in-place element %d = %d, want %d", i, inPlace[i].raw, want[i].raw)
		}
	}
	if d := SaturationCount() - before; d != wantEvents {
		t.Fatalf("BatchQ48Mul16 added %d events, want %d", d, wantEvents)
	}

	expectBatchPanic(t, "BatchDot16 length", func() { BatchDot16(a[:3], b[:2]) })
	expectBatchPanic(t, "BatchQ48Mul16 dst length", func() { BatchQ48Mul16(want[:2], q[:3], a[:3]) })
	expectBatchPanic(t, "BatchQ48Mul16 f length", func() { BatchQ48Mul16(want[:3], q[:3], a[:2]) })
}

func expectBatchPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: no panic", name)
		}
	}()
	fn()
}

func FuzzBatchDot16VsScalar(f *testing.F) {
	f.Add(int32(q16RawMax), int32(q16RawMax), uint8(200))
	f.Add(int32(q16RawMin), int32(q16RawMax), uint8(17))
	f.Add(int32(1), int32(-1), uint8(9))

	f.Fuzz(func(t *testing.T, seedA, seedB int32, size uint8) {
		n := int(size)
		a := make([]Q16, n)
		b := make([]Q16, n)
		ua, ub := uint32(seedA), uint32(seedB)
		for i := range n {
			a[i] = Q16{raw: int32(ua)}
			b[i] = Q16{raw: int32(ub)}
			ua = ua*batchStride + 1
			ub = ub*batchStride - 1
		}

		want, wantEvents := dot16Scalar(a, b)
		for _, k := range dot16Kernels {
			got, gotEvents := k.fn(a, b)
			if got != want || gotEvents != wantEvents {
				t.Fatalf("%s = (%d, %d events), oracle says (%d, %d events)",
					k.name, got.raw, gotEvents, want.raw, wantEvents)
			}
		}
	})
}

func FuzzBatchQ48Mul16VsScalar(f *testing.F) {
	f.Add(int64(1<<47), int32(q16RawOne), uint8(9))
	f.Add(int64(q48RawMin), int32(-1), uint8(8))
	f.Add(int64(1<<47|0x8000_0000), int32(q16RawMax), uint8(33))

	f.Fuzz(func(t *testing.T, seedQ int64, seedF int32, size uint8) {
		n := int(size)
		q := make([]Q48, n)
		fs := make([]Q16, n)
		uq, uf := uint64(seedQ), uint32(seedF)
		for i := range n {
			q[i] = Q48{raw: int64(uq)}
			fs[i] = Q16{raw: int32(uf)}
			uq = uq*uint64(batchStride) + 1
			uf = uf*batchStride - 1
		}

		want := make([]Q48, n)
		wantEvents := q48Mul16Scalar(want, q, fs)
		for _, k := range q48Mul16Kernels {
			got := make([]Q48, n)
			gotEvents := k.fn(got, q, fs)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s element %d = %d, oracle says %d",
						k.name, i, got[i].raw, want[i].raw)
				}
			}
			if gotEvents != wantEvents {
				t.Fatalf("%s recorded %d events, oracle recorded %d",
					k.name, gotEvents, wantEvents)
			}
		}
	})
}
