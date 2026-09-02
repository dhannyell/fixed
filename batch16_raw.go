//go:build goexperiment.simd && go1.27 && (amd64 || arm64)

package fixed

import "unsafe"

// This file holds every unsafe operation in the package. The vector kernels
// need the raw words of a Q16 or Q32 slice, because archsimd loads from
// []int32 and []int64; a copy would cost more than the kernel saves.

// Q16 is struct{raw int32} and Q32 is struct{raw int64}, so each slice already
// has the layout required by the kernels. The paired uintptr differences fail
// to compile if either size changes. A one-field struct has its field's
// alignment, so equal sizes leave no other layout difference.
const (
	_ = unsafe.Sizeof(Q16{}) - unsafe.Sizeof(int32(0))
	_ = unsafe.Sizeof(int32(0)) - unsafe.Sizeof(Q16{})
	_ = unsafe.Sizeof(Q32{}) - unsafe.Sizeof(int64(0))
	_ = unsafe.Sizeof(int64(0)) - unsafe.Sizeof(Q32{})
)

// rawInt32 reinterprets a Q16 slice as int32. It borrows the caller's backing
// array, so the result stays valid exactly as long as the argument does.
func rawInt32(s []Q16) []int32 {
	return unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
}

// rawInt64 reinterprets a Q32 slice as int64. It borrows the caller's backing
// array, so the result stays valid exactly as long as the argument does.
func rawInt64(s []Q32) []int64 {
	return unsafe.Slice((*int64)(unsafe.Pointer(unsafe.SliceData(s))), len(s))
}
