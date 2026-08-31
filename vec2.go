package fixed

import "math/bits"

// Vec2 is a 2D vector of Q components.
type Vec2 struct {
	X, Y Q
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
func (v Vec2) Mul(s Q) Vec2 {
	return Vec2{X: v.X.Mul(s), Y: v.Y.Mul(s)}
}

// Div returns v divided by s. It panics when s is zero.
func (v Vec2) Div(s Q) Vec2 {
	return Vec2{X: v.X.Div(s), Y: v.Y.Div(s)}
}

// Dot returns the dot product of two vectors.
func (v Vec2) Dot(o Vec2) Q {
	return v.X.Mul(o.X).Add(v.Y.Mul(o.Y))
}

// LenSq returns the squared length of the vector. It uses the scalar
// multiplication and addition rules, including saturation.
func (v Vec2) LenSq() Q {
	return v.Dot(v)
}

// Len returns the length of the vector, floored to the Q32.32 grid.
func (v Vec2) Len() Q {
	raw := hypotRaw(v.X.raw, v.Y.raw)
	if raw > uint64(rawMax) {
		saturationEvents.Add(1)
		return MaxValue()
	}
	return Q{raw: int64(raw)}
}

// Normalize returns a unit vector with the same direction as v. The
// zero vector returns the zero vector.
func (v Vec2) Normalize() Vec2 {
	x, y := unitPair(v.X.raw, v.Y.raw)
	return Vec2{X: x, Y: y}
}

// Distance returns the distance between v and o.
func (v Vec2) Distance(o Vec2) Q {
	return v.Sub(o).Len()
}

// DistanceSq returns the squared distance between v and o.
func (v Vec2) DistanceSq(o Vec2) Q {
	return v.Sub(o).LenSq()
}

// Lerp linearly interpolates between v and target by t. t is not
// clamped.
func (v Vec2) Lerp(target Vec2, t Q) Vec2 {
	return v.Add(target.Sub(v).Mul(t))
}

// hypotRaw returns floor(sqrt(x²+y²)) for raw Q32.32 values. The
// intermediate sum uses 128 bits so Len does not saturate early.
func hypotRaw(x, y int64) uint64 {
	xHi, xLo := bits.Mul64(magnitude(x), magnitude(x))
	yHi, yLo := bits.Mul64(magnitude(y), magnitude(y))
	lo, carry := bits.Add64(xLo, yLo, 0)
	hi, _ := bits.Add64(xHi, yHi, carry)

	loRoot, hiRoot := uint64(0), ^uint64(0)
	for loRoot < hiRoot {
		mid := loRoot + (hiRoot-loRoot)/2 + 1
		midHi, midLo := bits.Mul64(mid, mid)
		if midHi < hi || (midHi == hi && midLo <= lo) {
			loRoot = mid
		} else {
			hiRoot = mid - 1
		}
	}
	return loRoot
}

// unitPair scales before squaring so the pair cannot underflow to zero.
func unitPair(x, y int64) (Q, Q) {
	mx, my := magnitude(x), magnitude(y)
	scale := max(mx, my)
	if scale == 0 {
		return Q{}, Q{}
	}
	sx := divMag(mx, scale, x < 0)
	sy := divMag(my, scale, y < 0)
	n := sx.Mul(sx).Add(sy.Mul(sy)).Sqrt()
	return sx.Div(n), sy.Div(n)
}
