//go:build goexperiment.simd && arm64

package fixed

// init registers the vector kernels. NEON is mandatory on ARMv8, so the
// kernels below run on every arm64 CPU.
func init() {
	addKernels = append(addKernels, batchKernel{"neon", add16NEON})
	wrapKernels = append(wrapKernels, batchWrapKernel{"neon", add16WrapNEON})
}
