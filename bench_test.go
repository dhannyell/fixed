package fixed_test

// Kernel measurement harness. External consumers reuse these
// benchmark names; do not delete or rename them.

import (
	"testing"

	"github.com/dhannyell/fixed"
)

var benchSink int64

// BenchmarkSinCosTurns sweeps the circle with a golden-ratio step so
// the table cannot ride a cache-friendly sequential access pattern.
func BenchmarkSinCosTurns(b *testing.B) {
	var acc int64
	u := int64(0)
	for range b.N {
		a := fixed.FromRaw(u)
		acc += fixed.SinTurns(a).Raw() + fixed.CosTurns(a).Raw()
		u += 2654435761
	}
	benchSink = acc
}
