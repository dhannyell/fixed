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
	subKernels = []batchKernel{{"scalar", sub16Scalar}}
	mulKernels = []batchKernel{{"scalar", mul16Scalar}}
)

// batchClampKernel names one clamp implementation. A clamp records no events.
type batchClampKernel struct {
	name string
	fn   func(dst, a []Q16, lo, hi Q16)
}

var clampKernels = []batchClampKernel{{"scalar", clamp16Scalar}}

// batchWidenKernel names one Q16 to Q32 widening. The widening is exact, so
// it records no events.
type batchWidenKernel struct {
	name string
	fn   func(dst []Q32, a []Q16)
}

// batchNarrowKernel names one Q32 to Q16 narrowing.
type batchNarrowKernel struct {
	name string
	fn   func(dst []Q16, a []Q32) uint64
}

var (
	widenKernels  = []batchWidenKernel{{"scalar", q32FromQ16Scalar}}
	narrowKernels = []batchNarrowKernel{{"scalar", q16FromQ32Scalar}}
)

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
		{"sub", subKernels, sub16Scalar},
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

// TestBatchAliasingKeepsResults proves a kernel reads each lane before it
// writes it, so dst may name the same slice as a source.
func TestBatchAliasingKeepsResults(t *testing.T) {
	a, b := batchPairs()
	ops := []struct {
		op      string
		kernels []batchKernel
		oracle  func(dst, a, b []Q16) uint64
	}{
		{"add", addKernels, add16Scalar},
		{"sub", subKernels, sub16Scalar},
		{"mul", mulKernels, mul16Scalar},
	}
	for _, o := range ops {
		for _, k := range o.kernels {
			t.Run(o.op+"/"+k.name, func(t *testing.T) {
				want := make([]Q16, len(a))
				o.oracle(want, a, b)

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

	lo, hi := Q16{raw: -q16RawOne}, Q16{raw: q16RawOne}
	for _, k := range clampKernels {
		t.Run("clamp/"+k.name, func(t *testing.T) {
			want := make([]Q16, len(a))
			clamp16Scalar(want, a, lo, hi)

			inPlace := append([]Q16(nil), a...)
			k.fn(inPlace, inPlace, lo, hi)
			for i := range want {
				if inPlace[i] != want[i] {
					t.Fatalf("element %d = %d, want %d",
						i, inPlace[i].raw, want[i].raw)
				}
			}
		})
	}
}

// TestBatchClampMatchesTheScalarPath compares every clamp kernel against the
// oracle at a narrow window and at the full Q16 range, which clamps nothing.
func TestBatchClampMatchesTheScalarPath(t *testing.T) {
	a, _ := batchPairs()
	windows := []struct {
		name   string
		lo, hi Q16
	}{
		{"unit", Q16{raw: -q16RawOne}, Q16{raw: q16RawOne}},
		{"full", Q16{raw: q16RawMin}, Q16{raw: q16RawMax}},
	}
	for _, w := range windows {
		for _, k := range clampKernels {
			for _, n := range batchSizes {
				if n > len(a) {
					continue
				}
				t.Run(w.name+"/"+k.name+"/n="+strconv.Itoa(n), func(t *testing.T) {
					want := make([]Q16, n)
					clamp16Scalar(want, a[:n], w.lo, w.hi)

					got := make([]Q16, n)
					k.fn(got, a[:n], w.lo, w.hi)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("element %d = %d, oracle says %d",
								i, got[i].raw, want[i].raw)
						}
					}
				})
			}
		}
	}
	ResetSaturationCount()
	dst := make([]Q16, len(a))
	BatchClamp16(dst, a, Q16{raw: -q16RawOne}, Q16{raw: q16RawOne})
	if c := SaturationCount(); c != 0 {
		t.Errorf("BatchClamp16 recorded %d saturation events, want 0", c)
	}
}

// TestBatchPublicWrappersPublishCounts checks the exported functions add the
// element count to the shared counter with one atomic update.
func TestBatchPublicWrappersPublishCounts(t *testing.T) {
	a, b := batchPairs()
	scratch := make([]Q16, len(a))

	want := add16Scalar(scratch, a, b)
	ResetSaturationCount()
	BatchAdd16(scratch, a, b)
	if got := SaturationCount(); got != want {
		t.Errorf("BatchAdd16 published %d events, want %d", got, want)
	}

	want = sub16Scalar(scratch, a, b)
	ResetSaturationCount()
	BatchSub16(scratch, a, b)
	if got := SaturationCount(); got != want {
		t.Errorf("BatchSub16 published %d events, want %d", got, want)
	}

	want = mul16Scalar(scratch, a, b)
	ResetSaturationCount()
	BatchMul16(scratch, a, b)
	if got := SaturationCount(); got != want {
		t.Errorf("BatchMul16 published %d events, want %d", got, want)
	}
}

func TestBatchLengthMismatchPanics(t *testing.T) {
	calls := []struct {
		name string
		run  func()
	}{
		{"BatchAdd16", func() { BatchAdd16(make([]Q16, 2), make([]Q16, 3), make([]Q16, 3)) }},
		{"BatchSub16", func() { BatchSub16(make([]Q16, 2), make([]Q16, 3), make([]Q16, 3)) }},
		{"BatchMul16", func() { BatchMul16(make([]Q16, 2), make([]Q16, 3), make([]Q16, 3)) }},
		{"BatchClamp16", func() { BatchClamp16(make([]Q16, 2), make([]Q16, 3), Q16{}, Q16{}) }},
		{"BatchQ32FromQ16", func() { BatchQ32FromQ16(make([]Q32, 2), make([]Q16, 3)) }},
		{"BatchQ16FromQ32", func() { BatchQ16FromQ32(make([]Q16, 2), make([]Q32, 3)) }},
	}
	for _, c := range calls {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s with mismatched lengths did not panic", c.name)
				}
			}()
			c.run()
		})
	}
}

// TestBatchPathNamesAKnownFamily checks the introspection agrees with the
// families this package can build.
func TestBatchPathNamesAKnownFamily(t *testing.T) {
	switch p := BatchPath(); p {
	case "scalar", "avx2", "neon":
	default:
		t.Errorf("BatchPath() = %q, want scalar, avx2 or neon", p)
	}
}

// batchStride is the house pseudo-random step. It runs in uint32 because the
// value does not fit in int32.
const batchStride = uint32(2654435761)

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
	})
}

// conversionGrid returns Q32 raw values on both sides of the Q16 range and at
// the negative rounding edges, where a floor and a truncation disagree.
func conversionGrid() []int64 {
	return []int64{
		q32RawMin, q32RawMin + 1, -(1 << 47) - 1, -(1 << 47), -(1 << 47) + 1,
		-65537, -65536, -65535, -1, 0, 1, 65535, 65536, 65537,
		(1 << 47) - 1, 1 << 47, (1 << 47) + 1, q32RawMax - 1, q32RawMax,
	}
}

// TestBatchConversionsMatchTheScalarPath compares every conversion kernel
// against the scalar methods that define the two formats.
func TestBatchConversionsMatchTheScalarPath(t *testing.T) {
	narrowIn := make([]Q32, 0, len(conversionGrid()))
	for _, v := range conversionGrid() {
		narrowIn = append(narrowIn, Q32{raw: v})
	}
	widenIn, _ := batchPairs()

	for _, k := range widenKernels {
		for _, n := range batchSizes {
			if n > len(widenIn) {
				continue
			}
			t.Run("q32fromq16/"+k.name+"/n="+strconv.Itoa(n), func(t *testing.T) {
				got := make([]Q32, n)
				k.fn(got, widenIn[:n])
				for i := range got {
					if want := widenIn[i].ToQ32(); got[i] != want {
						t.Fatalf("element %d = %d, ToQ32 says %d",
							i, got[i].raw, want.raw)
					}
				}
			})
		}
	}

	for _, k := range narrowKernels {
		for _, n := range batchSizes {
			if n > len(narrowIn) {
				continue
			}
			t.Run("q16fromq32/"+k.name+"/n="+strconv.Itoa(n), func(t *testing.T) {
				ResetSaturationCount()
				want := make([]Q16, n)
				for i := range want {
					want[i] = narrowIn[i].ToQ16()
				}
				wantEvents := SaturationCount()

				got := make([]Q16, n)
				gotEvents := k.fn(got, narrowIn[:n])
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("element %d = %d, ToQ16 says %d",
							i, got[i].raw, want[i].raw)
					}
				}
				if gotEvents != wantEvents {
					t.Errorf("saturation events = %d, ToQ16 recorded %d",
						gotEvents, wantEvents)
				}
			})
		}
	}
}

// TestBatchConversionRoundTrip proves the widening loses nothing: every Q16
// survives a trip through Q32 unchanged.
func TestBatchConversionRoundTrip(t *testing.T) {
	a, _ := batchPairs()
	wide := make([]Q32, len(a))
	back := make([]Q16, len(a))

	ResetSaturationCount()
	BatchQ32FromQ16(wide, a)
	BatchQ16FromQ32(back, wide)
	for i := range a {
		if back[i] != a[i] {
			t.Fatalf("element %d = %d, want %d", i, back[i].raw, a[i].raw)
		}
	}
	if c := SaturationCount(); c != 0 {
		t.Errorf("the round trip recorded %d saturation events, want 0", c)
	}
}

func FuzzBatchMul16VsScalar(f *testing.F) {
	f.Add(int32(q16RawOne), int32(q16RawOne), uint8(9))
	f.Add(int32(q16RawMax), int32(q16RawMax), uint8(8))
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
		wantEvents := mul16Scalar(want, a, b)
		for _, k := range mulKernels {
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
	})
}

func FuzzBatchQ16FromQ32VsScalar(f *testing.F) {
	f.Add(int64(1<<47), uint8(9))
	f.Add(int64(-1<<47), uint8(8))
	f.Add(int64(-65537), uint8(33))

	f.Fuzz(func(t *testing.T, seed int64, size uint8) {
		n := int(size)
		a := make([]Q32, n)
		u := uint64(seed)
		for i := range n {
			a[i] = Q32{raw: int64(u)}
			u = u*uint64(batchStride) + 1
		}

		want := make([]Q16, n)
		wantEvents := q16FromQ32Scalar(want, a)
		for _, k := range narrowKernels {
			got := make([]Q16, n)
			gotEvents := k.fn(got, a)
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
