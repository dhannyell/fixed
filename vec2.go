package fixed

import "math/bits"

// Vec2 is a 2D vector of Q32 components.
type Vec2 struct {
	X, Y Q32
}

// Add returns the sum of two vectors.
func (v Vec2) Add(o Vec2) Vec2 {
	return Vec2{X: v.X.Add(o.X), Y: v.Y.Add(o.Y)}
}

// Sub returns the difference between two vectors.
func (v Vec2) Sub(o Vec2) Vec2 {
	return Vec2{X: v.X.Sub(o.X), Y: v.Y.Sub(o.Y)}
}

// Mul returns v scaled by s.
func (v Vec2) Mul(s Q32) Vec2 {
	return Vec2{X: v.X.Mul(s), Y: v.Y.Mul(s)}
}

// Div returns v divided by s. It panics when s is zero.
func (v Vec2) Div(s Q32) Vec2 {
	return Vec2{X: v.X.Div(s), Y: v.Y.Div(s)}
}

// Dot returns the dot product of two vectors.
func (v Vec2) Dot(o Vec2) Q32 {
	return v.X.Mul(o.X).Add(v.Y.Mul(o.Y))
}

// LenSq returns the squared length of the vector. It uses the scalar
// multiplication and addition rules, including saturation.
func (v Vec2) LenSq() Q32 {
	return v.Dot(v)
}

// Len returns the length of the vector, floored to the Q32.32 grid.
func (v Vec2) Len() Q32 {
	raw := hypotRaw(v.X.raw, v.Y.raw)
	if raw > uint64(q32RawMax) {
		saturationEvents.Add(1)
		return Q32MaxValue()
	}
	return Q32{raw: int64(raw)}
}

// Normalize returns a unit vector with the same direction as v. The
// zero vector returns the zero vector.
func (v Vec2) Normalize() Vec2 {
	x, y := unitPair(v.X.raw, v.Y.raw)
	return Vec2{X: x, Y: y}
}

// Distance returns the distance between v and o.
func (v Vec2) Distance(o Vec2) Q32 {
	return v.Sub(o).Len()
}

// DistanceSq returns the squared distance between v and o.
func (v Vec2) DistanceSq(o Vec2) Q32 {
	return v.Sub(o).LenSq()
}

// Lerp linearly interpolates between v and target by t. t is not
// clamped.
func (v Vec2) Lerp(target Vec2, t Q32) Vec2 {
	return v.Add(target.Sub(v).Mul(t))
}

// hypotRaw returns floor(sqrt(x²+y²)) for raw Q32.32 values. The
// intermediate sum uses 128 bits so Len does not saturate early.
func hypotRaw(x, y int64) uint64 {
	xHi, xLo := bits.Mul64(magnitude(x), magnitude(x))
	yHi, yLo := bits.Mul64(magnitude(y), magnitude(y))
	lo, carry := bits.Add64(xLo, yLo, 0)
	hi, _ := bits.Add64(xHi, yHi, carry)
	return isqrt128(hi, lo)
}

// unitPair scales before squaring so the pair cannot underflow to zero.
func unitPair(x, y int64) (Q32, Q32) {
	mx, my := magnitude(x), magnitude(y)
	scale := max(mx, my)
	if scale == 0 {
		return Q32{}, Q32{}
	}
	var sx, sy Q32
	switch {
	case mx == my:
		sx = Q32{raw: signedUnit(x < 0)}
		sy = Q32{raw: signedUnit(y < 0)}
	case mx > my:
		sx = Q32{raw: signedUnit(x < 0)}
		sy = divMag(my, scale, y < 0)
	default:
		sx = divMag(mx, scale, x < 0)
		sy = Q32{raw: signedUnit(y < 0)}
	}
	n := sx.Mul(sx).Add(sy.Mul(sy)).Sqrt()
	return sx.Div(n), sy.Div(n)
}

func signedUnit(neg bool) int64 {
	if neg {
		return -q32RawOne
	}
	return q32RawOne
}
