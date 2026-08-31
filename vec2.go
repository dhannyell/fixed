package fixed

// Vec2 is a 2D vector of Q components. Every operation composes the
// scalar contracts: saturation clamps and counts, division by zero
// panics.
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

// Div returns v divided by s. A zero s panics: it is a logic bug, not
// overflow.
func (v Vec2) Div(s Q) Vec2 {
	return Vec2{X: v.X.Div(s), Y: v.Y.Div(s)}
}

// Dot returns the dot product of two vectors.
func (v Vec2) Dot(o Vec2) Q {
	return v.X.Mul(o.X).Add(v.Y.Mul(o.Y))
}

// LenSq returns the squared length of the vector.
func (v Vec2) LenSq() Q {
	return v.Dot(v)
}

// Len returns the length of the vector. A component above ~46340 whole
// units saturates LenSq and the result loses meaning; keep vectors with
// headroom.
func (v Vec2) Len() Q {
	return v.LenSq().Sqrt()
}

// Normalize returns a unit vector with the same direction as v. The
// zero vector returns the zero vector.
func (v Vec2) Normalize() Vec2 {
	length := v.Len()
	if length.Eq(Zero()) {
		return Vec2{}
	}
	return v.Div(length)
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
