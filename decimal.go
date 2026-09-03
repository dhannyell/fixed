package fixed

// Q32MustParse parses a decimal literal. It rounds to the nearest Q32 value, with
// exact halves away from zero. It saturates outside the Q32 range and panics on
// malformed input.
func Q32MustParse(s string) Q32 {
	intPart, fraction, negative := scanDecimal(s, 1<<31)
	if intPart > 1<<31 {
		saturationEvents.Add(1)
		if negative {
			return Q32{raw: q32RawMin}
		}
		return Q32{raw: q32RawMax}
	}
	raw := intPart<<32 + decimalFraction(fraction, 32)
	if negative {
		if raw > 1<<63 {
			saturationEvents.Add(1)
			return Q32{raw: q32RawMin}
		}
		return Q32{raw: -int64(raw)} // A magnitude of 1<<63 converts to MinValue.
	}
	if raw > q32RawMax {
		saturationEvents.Add(1)
		return Q32{raw: q32RawMax}
	}
	return Q32{raw: int64(raw)}
}

// Q16MustParse parses a decimal literal. It rounds to the nearest Q16 value,
// with exact halves away from zero. It saturates outside the Q16 range and
// panics on malformed input.
func Q16MustParse(s string) Q16 {
	intPart, fraction, negative := scanDecimal(s, 1<<15)
	raw := int64(intPart)<<16 + int64(decimalFraction(fraction, 16))
	if negative {
		return q16Saturate(-raw)
	}
	return q16Saturate(raw)
}

// String returns the exact canonical decimal form of q. The widening
// conversion is exact, so the Q32 formatter emits the same value.
func (q Q16) String() string { return q.ToQ32().String() }

// Q48MustParse parses a decimal literal. It rounds to the nearest Q48 value,
// with exact halves away from zero. It saturates outside the Q48 range and
// panics on malformed input.
func Q48MustParse(s string) Q48 {
	intPart, fraction, negative := scanDecimal(s, 1<<47)
	if intPart > 1<<47 {
		saturationEvents.Add(1)
		if negative {
			return Q48{raw: q48RawMin}
		}
		return Q48{raw: q48RawMax}
	}
	raw := intPart<<16 + decimalFraction(fraction, 16)
	if negative {
		if raw > 1<<63 {
			saturationEvents.Add(1)
			return Q48{raw: q48RawMin}
		}
		return Q48{raw: -int64(raw)} // A magnitude of 1<<63 converts to MinValue.
	}
	if raw > q48RawMax {
		saturationEvents.Add(1)
		return Q48{raw: q48RawMax}
	}
	return Q48{raw: int64(raw)}
}

// String returns the exact canonical decimal form of q, such as "-6.25".
// For every q, MustParse(q.String()) == q. Use Raw for the exact bit pattern.
func (q Q32) String() string { return formatFixed(q.raw, 32) }

// String returns the exact canonical decimal form of q.
// For every q, Q48MustParse(q.String()) == q. Use Raw for the exact bit pattern.
func (q Q48) String() string { return formatFixed(q.raw, 16) }

// formatFixed writes a signed raw value with fractionBits bits of fraction.
func formatFixed(raw int64, fractionBits uint) string {
	n := magnitude(raw)
	mask := uint64(1)<<fractionBits - 1
	intPart := n >> fractionBits
	fraction := n & mask

	var buf []byte
	if raw < 0 {
		buf = append(buf, '-')
	}
	var digits [20]byte
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
		// A fraction of b bits has at most b decimal digits.
		for fraction != 0 {
			fraction *= 10
			buf = append(buf, byte('0'+fraction>>fractionBits))
			fraction &= mask
		}
	}
	return string(buf)
}

// scanDecimal validates a literal and splits it into parts. freeze bounds the
// integer accumulation; a frozen value only signals saturation.
func scanDecimal(s string, freeze uint64) (intPart uint64, fraction string, negative bool) {
	if s == "" {
		panic("fixed: malformed literal: " + s)
	}
	start := 0
	negative = s[0] == '-'
	if negative {
		start++
	}
	if start == len(s) {
		panic("fixed: malformed literal: " + s)
	}

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
		if dot == len(s) && intPart <= freeze {
			intPart = intPart*10 + uint64(c-'0')
		}
	}
	fraction = ""
	if dot < len(s) {
		fraction = s[dot+1:]
	}

	return
}

func decimalFraction(s string, fractionBits int) uint64 {
	if s == "" {
		return 0
	}
	digits := make([]byte, len(s))
	for i := range s {
		digits[i] = s[i] - '0'
	}

	var raw uint64
	for bit := range fractionBits + 1 {
		var carry byte
		for i := len(digits) - 1; i >= 0; i-- {
			v := digits[i]*2 + carry
			digits[i] = v % 10
			carry = v / 10
		}
		if bit < fractionBits {
			raw = raw<<1 | uint64(carry)
		} else if carry != 0 { // The extra bit applies half-away-from-zero rounding.
			raw++
		}
	}
	return raw
}
