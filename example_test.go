package fixed_test

import (
	"fmt"

	"github.com/dhannyell/fixed"
)

func ExampleQ32FromInt() {
	fmt.Println(fixed.Q32FromInt(3))
	fmt.Println(fixed.Q32FromInt(-2))
	// Output:
	// 3
	// -2
}

func ExampleQ32FromRatio() {
	// Use a ratio for exact fractional constants. There is no FromFloat.
	q := fixed.Q32FromRatio(5, 2)
	fmt.Println(q)
	// Output:
	// 2.5
}

func ExampleQ32MustParse() {
	fmt.Println(fixed.Q32MustParse("6.25"))
	fmt.Println(fixed.Q32MustParse("-0.001"))
	// Output:
	// 6.25
	// -0.00099999993108212947845458984375
}

func ExampleQ32_Mul() {
	area := fixed.Q32FromRatio(5, 2).Mul(fixed.Q32FromInt(4))
	fmt.Println(area)
	// Output:
	// 10
}

func ExampleQ32_String() {
	q := fixed.Q32FromRaw(1)
	text := q.String()
	fmt.Println(text)
	fmt.Println(fixed.Q32MustParse(text).Eq(q))
	// Output:
	// 0.00000000023283064365386962890625
	// true
}

func ExampleVec2_Normalize() {
	v := fixed.Vec2{X: fixed.Q32Zero(), Y: fixed.Q32FromInt(-7)}
	u := v.Normalize()
	fmt.Println(u.X, u.Y)
	// Output:
	// 0 -1
}

func ExampleRotFromTurns() {
	// A quarter turn sends (1, 0) to (0, 1) exactly.
	r := fixed.RotFromTurns(fixed.Q32FromRatio(1, 4))
	v := r.Apply(fixed.Vec2{X: fixed.Q32One(), Y: fixed.Q32Zero()})
	fmt.Println(v.X, v.Y)
	// Output:
	// 0 1
}

func ExampleQ32_Clamp() {
	speed := fixed.Q32FromInt(150)
	limited := speed.Clamp(fixed.Q32Zero(), fixed.Q32FromInt(100))
	fmt.Println(limited)
	// Output:
	// 100
}
