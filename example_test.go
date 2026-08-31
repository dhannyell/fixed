package fixed_test

import (
	"fmt"

	"github.com/dhannyell/fixed"
)

func ExampleFromInt() {
	fmt.Println(fixed.FromInt(3))
	fmt.Println(fixed.FromInt(-2))
	// Output:
	// 3
	// -2
}

func ExampleFromRatio() {
	// Use a ratio for exact fractional constants. There is no FromFloat.
	q := fixed.FromRatio(5, 2)
	fmt.Println(q)
	// Output:
	// 2.5
}

func ExampleMustParse() {
	fmt.Println(fixed.MustParse("6.25"))
	fmt.Println(fixed.MustParse("-0.001"))
	// Output:
	// 6.25
	// -0.00099999993108212947845458984375
}

func ExampleQ_Mul() {
	area := fixed.FromRatio(5, 2).Mul(fixed.FromInt(4))
	fmt.Println(area)
	// Output:
	// 10
}

func ExampleQ_String() {
	q := fixed.FromRaw(1)
	text := q.String()
	fmt.Println(text)
	fmt.Println(fixed.MustParse(text).Eq(q))
	// Output:
	// 0.00000000023283064365386962890625
	// true
}

func ExampleVec2_Normalize() {
	v := fixed.Vec2{X: fixed.Zero(), Y: fixed.FromInt(-7)}
	u := v.Normalize()
	fmt.Println(u.X, u.Y)
	// Output:
	// 0 -1
}

func ExampleRotFromTurns() {
	// A quarter turn sends (1, 0) to (0, 1) exactly.
	r := fixed.RotFromTurns(fixed.FromRatio(1, 4))
	v := r.Apply(fixed.Vec2{X: fixed.One(), Y: fixed.Zero()})
	fmt.Println(v.X, v.Y)
	// Output:
	// 0 1
}

func ExampleQ_Clamp() {
	speed := fixed.FromInt(150)
	limited := speed.Clamp(fixed.Zero(), fixed.FromInt(100))
	fmt.Println(limited)
	// Output:
	// 100
}
