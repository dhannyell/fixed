//go:build goexperiment.simd && amd64

package fixed

import "simd/archsimd"

// init registers the vector kernels this CPU can run, so the parity grid
// compares every reachable path against the scalar oracle on one machine.
func init() {
	if archsimd.X86.AVX2() {
		addKernels = append(addKernels, batchKernel{"avx2", add16AVX2})
	}
}
