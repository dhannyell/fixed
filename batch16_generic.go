//go:build !goexperiment.simd || !go1.27 || (!amd64 && !arm64)

package fixed

// selectKernels uses scalar kernels when SIMD is disabled or unsupported on
// the target architecture.
func selectKernels() batchKernels {
	return batchKernels{
		path:       "scalar",
		add:        add16Scalar,
		sub:        sub16Scalar,
		mul:        mul16Scalar,
		clamp:      clamp16Scalar,
		q32FromQ16: q32FromQ16Scalar,
		q16FromQ32: q16FromQ32Scalar,
		dot16:      dot16Scalar,
		q48Mul16:   q48Mul16Scalar,
	}
}
