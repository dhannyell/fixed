package fixed_test

import (
	"fmt"

	fixed "github.com/dhannyell/fixed"
)

func ExampleFromInt() {
	fmt.Println(fixed.FromInt(3))
	fmt.Println(fixed.FromInt(-2))
	// Output:
	// 3
	// -2
}

func ExampleFromRatio() {
	// The idiom for 2.5 is a ratio of integers; there is no FromFloat.
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
	fmt.Println(fixed.FromRatio(7, 2))
	fmt.Println(fixed.FromRatio(-1, 4))
	fmt.Println(fixed.FromInt(3))
	// Output:
	// 3.5
	// -0.25
	// 3
}

func ExampleQ_Clamp() {
	speed := fixed.FromInt(150)
	limited := speed.Clamp(fixed.Zero(), fixed.FromInt(100))
	fmt.Println(limited)
	// Output:
	// 100
}
