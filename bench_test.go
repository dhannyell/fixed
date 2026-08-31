package fixed_test

import (
	"testing"

	"github.com/dhannyell/fixed"
)

var benchSink int64

// BenchmarkSinCosTurns uses a non-sequential step to exercise table access.
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
