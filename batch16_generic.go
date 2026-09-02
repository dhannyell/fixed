//go:build !goexperiment.simd || (!amd64 && !arm64)

package fixed

// selectKernels returns the scalar kernels. This build has no vector path,
// either because the simd experiment is off or because the architecture
// offers no vectors. The scalar kernels own the result bits everywhere.
func selectKernels() batchKernels {
	return batchKernels{add: add16Scalar, mul: mul16Scalar}
}
