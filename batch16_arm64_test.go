//go:build goexperiment.simd && arm64

package fixed

import "testing"

// init registers the vector kernels. NEON is mandatory on ARMv8, so the
// kernels below run on every arm64 CPU.
func init() {
	addKernels = append(addKernels, batchKernel{"neon", add16NEON})
	subKernels = append(subKernels, batchKernel{"neon", sub16NEON})
	clampKernels = append(clampKernels, batchClampKernel{"neon", clamp16NEON})
}

// TestBatchPathNamesTheActiveKernels ties the introspection to the kernels
// this architecture always selects.
func TestBatchPathNamesTheActiveKernels(t *testing.T) {
	if got := BatchPath(); got != "neon" {
		t.Errorf("BatchPath() = %q, want neon", got)
	}
}
