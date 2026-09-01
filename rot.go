package fixed

// Rot is a 2D rotation stored as its sine and cosine.
//
// The zero Rot is not a valid rotation; start from RotIdentity or
// RotFromTurns.
type Rot struct {
	Sin, Cos Q32
}

// RotIdentity returns the rotation by zero turns.
func RotIdentity() Rot {
	return Rot{Sin: Q32{}, Cos: Q32One()}
}

// RotFromTurns returns the rotation by the angle t in turns. It shares
// the kernel and the contract of SinTurns and CosTurns.
func RotFromTurns(t Q32) Rot {
	sin, cos := sinCosFrac(uint32(t.raw))
	return Rot{Sin: Q32{raw: sin}, Cos: Q32{raw: cos}}
}

// Apply rotates the vector v by r.
func (r Rot) Apply(v Vec2) Vec2 {
	return Vec2{
		X: r.Cos.Mul(v.X).Sub(r.Sin.Mul(v.Y)),
		Y: r.Sin.Mul(v.X).Add(r.Cos.Mul(v.Y)),
	}
}

// Mul composes the rotations: the result rotates by r then by o. It
// uses the angle-sum identities, so each product floors once.
func (r Rot) Mul(o Rot) Rot {
	return Rot{
		Sin: r.Sin.Mul(o.Cos).Add(r.Cos.Mul(o.Sin)),
		Cos: r.Cos.Mul(o.Cos).Sub(r.Sin.Mul(o.Sin)),
	}
}

// Inv returns the inverse rotation. For a unit rotation the inverse is
// the conjugate, so no division is needed.
func (r Rot) Inv() Rot {
	return Rot{Sin: r.Sin.Neg(), Cos: r.Cos}
}

// Normalize rescales r to unit length. A zero r returns the identity.
func (r Rot) Normalize() Rot {
	if r.Sin.raw == 0 && r.Cos.raw == 0 {
		return RotIdentity()
	}
	sin, cos := unitPair(r.Sin.raw, r.Cos.raw)
	return Rot{Sin: sin, Cos: cos}
}
