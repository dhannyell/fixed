package fixed

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"testing"
)

// Pin both the tables and off-grid outputs: numerical error bounds alone
// do not protect the raw compatibility contract.
func TestTrigCompatibilityBits(t *testing.T) {
	h := sha256.New()
	var word [8]byte
	write := func(raw int64) {
		binary.LittleEndian.PutUint64(word[:], uint64(raw))
		_, _ = h.Write(word[:])
	}
	for _, table := range [][1025]int64{sineQuarter, atanUnit} {
		for _, raw := range table {
			write(raw)
		}
	}
	if got := fmt.Sprintf("%x", h.Sum(nil)); got != "7b76f214cd865a8352753402013e03c8ab64ea6a25d1e5ac3f57b8a775a38b9a" {
		t.Fatalf("trigonometric table bits changed: %s", got)
	}
	h.Reset()
	for i := range int64(4096) {
		u := i<<20 | 0x5a5a5
		write(SinTurns(Q32FromRaw(u)).Raw())
		write(CosTurns(Q32FromRaw(u)).Raw())
		x, y := int64(q32RawOne), u
		for _, signs := range [][2]int64{{1, 1}, {-1, 1}, {-1, -1}, {1, -1}} {
			write(Atan2Turns(Q32FromRaw(signs[1]*y), Q32FromRaw(signs[0]*x)).Raw())
			write(Atan2Turns(Q32FromRaw(signs[1]*x), Q32FromRaw(signs[0]*y)).Raw())
		}
	}
	if got := fmt.Sprintf("%x", h.Sum(nil)); got != "9c2c7ccc6347d0a4980d465fba2043ea075f6cd1fd9330f53130ba3f9c77bfa6" {
		t.Fatalf("trigonometric output bits changed: %s", got)
	}
}
