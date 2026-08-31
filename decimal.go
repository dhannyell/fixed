package fixed

// MustParse parses a decimal literal. It rounds to the nearest Q32 value, with
// exact halves away from zero. It saturates outside the Q32 range and panics on
// malformed input.
func MustParse(s string) Q32 {
	if s == "" {
		panic("fixed: malformed literal: " + s)
	}
	start := 0
	negative := s[0] == '-'
	if negative {
		start++
	}
	if start == len(s) {
		panic("fixed: malformed literal: " + s)
	}

	var intPart uint64
	dot := len(s)
	for i := start; i < len(s); i++ {
		c := s[i]
		if c == '.' {
			if dot != len(s) || i == start || i == len(s)-1 {
				panic("fixed: malformed literal: " + s)
			}
			dot = i
			continue
		}
		if c < '0' || c > '9' {
			panic("fixed: malformed literal: " + s)
		}
		// Freezing an out-of-range value prevents overflow on long literals.
		if dot == len(s) && intPart <= 1<<31 {
			intPart = intPart*10 + uint64(c-'0')
		}
	}
	if intPart > 1<<31 {
		saturationEvents.Add(1)
		if negative {
			return Q32{raw: rawMin}
		}
		return Q32{raw: rawMax}
	}

	fraction := ""
	if dot < len(s) {
		fraction = s[dot+1:]
	}
	raw := intPart<<32 + decimalFraction(fraction)
	if negative {
		if raw > 1<<63 {
			saturationEvents.Add(1)
			return Q32{raw: rawMin}
		}
		return Q32{raw: -int64(raw)} // A magnitude of 1<<63 converts to MinValue.
	}
	if raw > rawMax {
		saturationEvents.Add(1)
		return Q32{raw: rawMax}
	}
	return Q32{raw: int64(raw)}
}

// String returns the exact canonical decimal form of q, such as "-6.25".
// For every q, MustParse(q.String()) == q. Use Raw for the exact bit pattern.
func (q Q32) String() string {
	n := magnitude(q.raw)
	intPart := n >> 32
	fraction := n & (rawOne - 1)

	var buf []byte
	if q.raw < 0 {
		buf = append(buf, '-')
	}
	var digits [10]byte
	i := len(digits)
	for {
		i--
		digits[i] = byte('0' + intPart%10)
		intPart /= 10
		if intPart == 0 {
			break
		}
	}
	buf = append(buf, digits[i:]...)
	if fraction != 0 {
		buf = append(buf, '.')
		// A Q32.32 fraction has at most 32 decimal digits.
		for fraction != 0 {
			fraction *= 10
			buf = append(buf, byte('0'+fraction>>32))
			fraction &= rawOne - 1
		}
	}
	return string(buf)
}

func decimalFraction(s string) uint64 {
	if s == "" {
		return 0
	}
	digits := make([]byte, len(s))
	for i := range s {
		digits[i] = s[i] - '0'
	}

	var raw uint64
	for bit := range 33 {
		var carry byte
		for i := len(digits) - 1; i >= 0; i-- {
			v := digits[i]*2 + carry
			digits[i] = v % 10
			carry = v / 10
		}
		if bit < 32 {
			raw = raw<<1 | uint64(carry)
		} else if carry != 0 { // Bit 33 applies half-away-from-zero rounding.
			raw++
		}
	}
	return raw
}
