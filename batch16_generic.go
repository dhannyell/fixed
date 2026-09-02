//go:build !goexperiment.simd || (!amd64 && !arm64)

package fixed

// selectKernels returns the scalar kernels. This build has no vector path,
// either because the simd experiment is off or because the architecture
// offers no vectors. The scalar kernels own the result bits everywhere.
func selectKernels() batchKernels {
	return batchKernels{
		path:       "scalar",
		add:        add16Scalar,
		sub:        sub16Scalar,
		mul:        mul16Scalar,
		clamp:      clamp16Scalar,
		q32FromQ16: q32FromQ16Scalar,
		q16FromQ32: q16FromQ32Scalar,
	}
}
