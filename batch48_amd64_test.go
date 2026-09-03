//go:build goexperiment.simd && go1.27 && amd64

package fixed

import "simd/archsimd"

func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	dot16Kernels = append(dot16Kernels, dot16Kernel{"avx2", dot16AVX2})
	q48Mul16Kernels = append(q48Mul16Kernels, q48Mul16Kernel{"avx2", q48Mul16AVX2})
}
