//go:build goexperiment.simd && amd64

package fixed

import (
	"simd/archsimd"
	"testing"
)

// init registers the vector kernels this CPU can run, so the parity grid
// compares every reachable path against the scalar oracle on one machine.
func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	addKernels = append(addKernels, batchKernel{"avx2", add16AVX2})
	subKernels = append(subKernels, batchKernel{"avx2", sub16AVX2})
	mulKernels = append(mulKernels, batchKernel{"avx2", mul16AVX2})
	clampKernels = append(clampKernels, batchClampKernel{"avx2", clamp16AVX2})
	widenKernels = append(widenKernels, batchWidenKernel{"avx2", q32FromQ16AVX2})
	narrowKernels = append(narrowKernels, batchNarrowKernel{"avx2", q16FromQ32AVX2})
}

// TestBatchPathNamesTheActiveKernels ties the introspection to the CPU check
// that selects the kernels.
func TestBatchPathNamesTheActiveKernels(t *testing.T) {
	want := "scalar"
	if archsimd.X86.AVX2() {
		want = "avx2"
	}
	if got := BatchPath(); got != want {
		t.Errorf("BatchPath() = %q, want %q", got, want)
	}
}
