package fixed

import (
	"strconv"
	"testing"
)

// batchKernel names one add or mul implementation for the parity grid.
type batchKernel struct {
	name string
	fn   func(dst, a, b []Q16) uint64
}

// addKernels and mulKernels list every path this build can reach. The
// architecture files append their vector kernels when the CPU supports them.
var (
	addKernels = []batchKernel{{"scalar", add16Scalar}}
	mulKernels = []batchKernel{{"scalar", mul16Scalar}}
)

// batchWrapKernel names one wrapping add. A wrapping kernel records no events.
type batchWrapKernel struct {
	name string
	fn   func(dst, a, b []Q16)
}

var wrapKernels = []batchWrapKernel{{"scalar", add16WrapScalar}}

// axpyForm names one traversal of the composite kernel. Every form publishes
// its own saturation events, so a caller compares them through
// SaturationCount.
type axpyForm struct {
	name string
	run  func(x, m, v []Q16, bias, lo, hi Q16)
}

// axpyForms lists the traversals this build can reach. The architecture files
// append their vector kernels.
var axpyForms = []axpyForm{
	{"fused", axpyClampFused},
	{"multipass", func(x, m, v []Q16, bias, lo, hi Q16) {
		biasVec := make([]Q16, len(x))
		for i := range biasVec {
			biasVec[i] = bias
		}
		axpyClampMultipass(x, m, v, biasVec, make([]Q16, len(x)), lo, hi)
	}},
}

// batchGrid returns raw values that reach both saturation edges.
func batchGrid() []int32 {
	return []int32{
		q16RawMin, q16RawMin + 1, q16RawMin / 2,
		-q16RawOne, -0x8000, -1, 0, 1, 0x7FFF, q16RawOne,
		q16RawMax / 2, q16RawMax - 1, q16RawMax,
	}
}

// batchPairs expands the grid into every ordered pair, so one run covers full
// vector blocks and a remainder tail.
func batchPairs() (a, b []Q16) {
	g := batchGrid()
	for _, x := range g {
		for _, y := range g {
			a = append(a, Q16{raw: x})
			b = append(b, Q16{raw: y})
		}
	}
	return a, b
}

// batchSizes covers empty input, sub-block input, exact blocks and tails for
// both the 8-lane and the 4-lane kernels.
var batchSizes = []int{0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 64, 169}

func TestBatchOpsMatchTheScalarPath(t *testing.T) {
	a, b := batchPairs()
	ops := []struct {
		op      string
		kernels []batchKernel
		oracle  func(dst, a, b []Q16) uint64
	}{
		{"add", addKernels, add16Scalar},
		{"mul", mulKernels, mul16Scalar},
	}
	for _, o := range ops {
		for _, k := range o.kernels {
			for _, n := range batchSizes {
				if n > len(a) {
					continue
				}
				t.Run(o.op+"/"+k.name+"/n="+strconv.Itoa(n), func(t *testing.T) {
					want := make([]Q16, n)
					wantEvents := o.oracle(want, a[:n], b[:n])

					got := make([]Q16, n)
					gotEvents := k.fn(got, a[:n], b[:n])

					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("element %d = %d, oracle says %d",
								i, got[i].raw, want[i].raw)
						}
					}
					if gotEvents != wantEvents {
						t.Errorf("saturation events = %d, oracle says %d",
							gotEvents, wantEvents)
					}
				})
			}
		}
	}
}

func TestBatchWrapOpsNeverSaturate(t *testing.T) {
	a, b := batchPairs()
	n := len(a)

	ResetSaturationCount()
	gotMul := make([]Q16, n)
	Mul16Wrap(gotMul, a, b)
	for _, k := range wrapKernels {
		for _, size := range batchSizes {
			if size > n {
				continue
			}
			got := make([]Q16, size)
			k.fn(got, a[:size], b[:size])
			for i := range got {
				if want := a[i].raw + b[i].raw; got[i].raw != want {
					t.Fatalf("%s element %d = %d, want %d",
						k.name, i, got[i].raw, want)
				}
			}
		}
	}
	if c := SaturationCount(); c != 0 {
		t.Errorf("wrap ops recorded %d saturation events, want 0", c)
	}

	for i := range a {
		want := int32((int64(a[i].raw) * int64(b[i].raw)) >> 16)
		if gotMul[i].raw != want {
			t.Fatalf("Mul16Wrap element %d = %d, want %d", i, gotMul[i].raw, want)
		}
	}
}

// TestBatchAliasingKeepsResults proves a kernel reads each lane before it
// writes it, so dst may name the same slice as a source.
func TestBatchAliasingKeepsResults(t *testing.T) {
	a, b := batchPairs()
	for _, k := range addKernels {
		t.Run(k.name, func(t *testing.T) {
			want := make([]Q16, len(a))
			add16Scalar(want, a, b)

			inPlace := append([]Q16(nil), a...)
			k.fn(inPlace, inPlace, b)
			for i := range want {
				if inPlace[i] != want[i] {
					t.Fatalf("element %d = %d, want %d",
						i, inPlace[i].raw, want[i].raw)
				}
			}
		})
	}
}

// TestBatchPublicWrappersPublishCounts checks the exported functions add the
// element count to the shared counter with one atomic update.
func TestBatchPublicWrappersPublishCounts(t *testing.T) {
	a, b := batchPairs()
	scratch := make([]Q16, len(a))

	want := add16Scalar(scratch, a, b)
	ResetSaturationCount()
	Add16(scratch, a, b)
	if got := SaturationCount(); got != want {
		t.Errorf("Add16 published %d events, want %d", got, want)
	}

	want = mul16Scalar(scratch, a, b)
	ResetSaturationCount()
	Mul16(scratch, a, b)
	if got := SaturationCount(); got != want {
		t.Errorf("Mul16 published %d events, want %d", got, want)
	}
}

func TestBatchLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Add16 with mismatched lengths did not panic")
		}
	}()
	Add16(make([]Q16, 2), make([]Q16, 3), make([]Q16, 3))
}

// batchStride is the house pseudo-random step. It runs in uint32 because the
// value does not fit in int32.
const batchStride = uint32(2654435761)

// axpyInputs builds the composite kernel operands with the house stride.
// shift narrows the operands: a wide shift keeps the run inside the hot regime,
// a shift of zero drives every saturation site.
func axpyInputs(n int, shift uint) (x, m, v []Q16) {
	u := uint32(1)
	next := func() Q16 {
		u = u*batchStride + 1
		return Q16{raw: int32(u) >> shift}
	}
	for range n {
		x = append(x, next())
		m = append(m, next())
		v = append(v, next())
	}
	return x, m, v
}

// TestAxpyFormsAgree proves the three traversals of the composite kernel
// produce one result and one saturation count.
func TestAxpyFormsAgree(t *testing.T) {
	bias := Q16{raw: q16RawOne / 3}
	lo := Q16{raw: -8 * q16RawOne}
	hi := Q16{raw: 8 * q16RawOne}

	// A saturating regime is required: without it the counter comparison is
	// vacuous, because every form reports zero.
	regimes := []struct {
		name  string
		shift uint
	}{{"hot", 8}, {"saturating", 0}}

	for _, r := range regimes {
		for _, n := range batchSizes {
			t.Run(r.name+"/n="+strconv.Itoa(n), func(t *testing.T) {
				runAxpyParity(t, n, r.shift, bias, lo, hi)
			})
		}
	}
}

func runAxpyParity(t *testing.T, n int, shift uint, bias, lo, hi Q16) {
	t.Helper()
	x, m, v := axpyInputs(n, shift)

	wantX := append([]Q16(nil), x...)
	ResetSaturationCount()
	axpyClampPerCall(wantX, m, v, bias, lo, hi)
	wantEvents := SaturationCount()

	for _, f := range axpyForms {
		gotX := append([]Q16(nil), x...)
		ResetSaturationCount()
		f.run(gotX, m, v, bias, lo, hi)
		gotEvents := SaturationCount()

		for i := range wantX {
			if gotX[i] != wantX[i] {
				t.Fatalf("%s element %d = %d, per-call says %d",
					f.name, i, gotX[i].raw, wantX[i].raw)
			}
		}
		if gotEvents != wantEvents {
			t.Errorf("%s recorded %d events, per-call recorded %d",
				f.name, gotEvents, wantEvents)
		}
	}
	if shift == 0 && n >= 8 && wantEvents == 0 {
		t.Errorf("the saturating regime produced no events at n=%d", n)
	}
}

func FuzzBatchAdd16VsScalar(f *testing.F) {
	f.Add(int32(1), int32(-1), uint8(9))
	f.Add(int32(q16RawMax), int32(1), uint8(8))
	f.Add(int32(q16RawMin), int32(-1), uint8(33))

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

		want := make([]Q16, n)
		wantEvents := add16Scalar(want, a, b)
		wantWrap := make([]Q16, n)
		Add16Wrap(wantWrap, a, b)

		for _, k := range addKernels {
			got := make([]Q16, n)
			gotEvents := k.fn(got, a, b)
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
		for i := range wantWrap {
			if w := a[i].raw + b[i].raw; wantWrap[i].raw != w {
				t.Fatalf("wrap element %d = %d, want %d", i, wantWrap[i].raw, w)
			}
		}
	})
}
