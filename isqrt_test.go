package fixed

import (
	"math/big"
	"math/bits"
	"testing"
)

// oracleIsqrt128 is the reference root: big.Int square root of the full
// 128-bit radicand.
func oracleIsqrt128(hi, lo uint64) uint64 {
	n := new(big.Int).Lsh(new(big.Int).SetUint64(hi), 64)
	n.Or(n, new(big.Int).SetUint64(lo))
	return n.Sqrt(n).Uint64()
}

// isqrtNewtonRef is a pure-integer root without any float path. It
// cross-checks the shipped kernel with an independent algorithm.
func isqrtNewtonRef(hi, lo uint64) uint64 {
	if hi == 0 && lo == 0 {
		return 0
	}
	if hi >= ^uint64(0)-1 {
		r := ^uint64(0)
		for {
			pHi, pLo := bits.Mul64(r, r)
			if pHi < hi || (pHi == hi && pLo <= lo) {
				return r
			}
			r--
		}
	}
	var z int
	if hi != 0 {
		z = bits.LeadingZeros64(hi)
	} else {
		z = 64 + bits.LeadingZeros64(lo)
	}
	z &^= 1
	// The seed only needs to start above the root; 2^(64-z/2) does.
	x := uint64(1) << (63 - z/2)
	if x2 := x << 1; x2 != 0 {
		x = x2
	} else {
		x = ^uint64(0) - 1
	}
	for {
		q, _ := bits.Div64(hi, lo, x)
		sum, carry := bits.Add64(x, q, 0)
		y := sum>>1 | carry<<63
		if y >= x {
			break
		}
		x = y
	}
	for {
		pHi, pLo := bits.Mul64(x, x)
		if pHi > hi || (pHi == hi && pLo > lo) {
			x--
			continue
		}
		break
	}
	for x < ^uint64(0) {
		y := x + 1
		pHi, pLo := bits.Mul64(y, y)
		if pHi > hi || (pHi == hi && pLo > lo) {
			break
		}
		x = y
	}
	return x
}

// splitmix64 walks a deterministic 64-bit sequence for random cases.
func splitmix64(state *uint64) uint64 {
	*state += 0x9e3779b97f4a7c15
	z := *state
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}

func isqrtCases() [][2]uint64 {
	cases := [][2]uint64{
		{0, 0}, {0, 1}, {0, 2}, {0, 3},
		{^uint64(0), ^uint64(0)},
		{^uint64(0), 0},
		{^uint64(0) - 1, ^uint64(0)},
		{^uint64(0) - 1, 0},
		{^uint64(0) - 2, ^uint64(0)},
	}
	// Powers of two and neighbors across the whole 128-bit range.
	for n := range 128 {
		hi, lo := uint64(0), uint64(0)
		if n >= 64 {
			hi = 1 << (n - 64)
		} else {
			lo = 1 << n
		}
		cases = append(cases, [2]uint64{hi, lo})
		loM, borrow := bits.Sub64(lo, 1, 0)
		cases = append(cases, [2]uint64{hi - borrow, loM})
		loP, carry := bits.Add64(lo, 1, 0)
		cases = append(cases, [2]uint64{hi + carry, loP})
	}
	// Perfect squares and neighbors: the floor must step exactly at r².
	roots := []uint64{1, 2, 3, 255, 256, 1 << 16, 1<<24 - 1, 1 << 32, 1<<48 + 12345, 1<<63 - 1, 1 << 63, ^uint64(0) - 1, ^uint64(0)}
	for _, r := range roots {
		hi, lo := bits.Mul64(r, r)
		cases = append(cases, [2]uint64{hi, lo})
		loM, borrow := bits.Sub64(lo, 1, 0)
		cases = append(cases, [2]uint64{hi - borrow, loM})
		loP, carry := bits.Add64(lo, 1, 0)
		cases = append(cases, [2]uint64{hi + carry, loP})
	}
	return cases
}

func TestIsqrt128MatchesOracle(t *testing.T) {
	for _, c := range isqrtCases() {
		want := oracleIsqrt128(c[0], c[1])
		if got := isqrt128(c[0], c[1]); got != want {
			t.Fatalf("isqrt128(%#x, %#x) = %d, want %d", c[0], c[1], got, want)
		}
		if ref := isqrtNewtonRef(c[0], c[1]); ref != want {
			t.Fatalf("isqrtNewtonRef(%#x, %#x) = %d, want %d", c[0], c[1], ref, want)
		}
	}
	state := uint64(1)
	for range 1_000_000 {
		hi, lo := splitmix64(&state), splitmix64(&state)
		// Sweep all widths, not only full 128-bit values.
		hi >>= splitmix64(&state) % 65
		want := oracleIsqrt128(hi, lo)
		if got := isqrt128(hi, lo); got != want {
			t.Fatalf("isqrt128(%#x, %#x) = %d, want %d", hi, lo, got, want)
		}
		if ref := isqrtNewtonRef(hi, lo); ref != want {
			t.Fatalf("isqrtNewtonRef(%#x, %#x) = %d, want %d", hi, lo, ref, want)
		}
	}
}

// isqrtOperands mixes magnitudes seen by Sqrt (96-bit radicands) and by
// hypotRaw (sums of squared components).
func isqrtOperands() [256][2]uint64 {
	var ops [256][2]uint64
	state := uint64(42)
	for i := range ops {
		raw := splitmix64(&state) >> (1 + splitmix64(&state)%24)
		switch i % 3 {
		case 0: // Sqrt radicand of a positive Q32 value.
			ops[i] = [2]uint64{raw >> 32, raw << 32}
		case 1: // hypotRaw radicand for two components near raw.
			xHi, xLo := bits.Mul64(raw, raw)
			yHi, yLo := bits.Mul64(raw/3+1, raw/3+1)
			lo, carry := bits.Add64(xLo, yLo, 0)
			ops[i] = [2]uint64{xHi + yHi + carry, lo}
		default: // Small value: exercises the leading-zero skip.
			ops[i] = [2]uint64{0, raw >> 16}
		}
	}
	return ops
}

var benchSinkRoot uint64

func BenchmarkIsqrt128(b *testing.B) {
	ops := isqrtOperands()
	var acc uint64
	idx := 0
	for range b.N {
		c := ops[idx&255]
		r := isqrt128(c[0], c[1])
		acc += r
		// The next index depends on the root, so calls chain.
		idx += int(r&7) + 1
	}
	benchSinkRoot = acc
}
