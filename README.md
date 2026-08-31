# fixed

`fixed` is a small Go package for signed Q32.32 arithmetic. Equal inputs produce
the same result bits on every supported architecture.

A `Q` value uses one `int64`: 32 bits for the signed integer part and 32 bits
for the fraction. This gives a resolution of 2⁻³² and a range from -2³¹ to
2³¹ - 2⁻³².

The module is pre-v1. Its import path may change before the first stable
release. It requires Go 1.26.4 or newer.

## Install

```sh
go get github.com/dhannyell/fixed
```

## Quick start

```go
package main

import (
	"fmt"

	"github.com/dhannyell/fixed"
)

func main() {
	price := fixed.MustParse("6.25")
	quantity := fixed.FromRatio(3, 2)
	total := price.Mul(quantity)

	fmt.Println(total)       // 9.375
	fmt.Println(total.Raw()) // 40265318400
}
```

## Why there is no float constructor

Floating-point input can already contain small differences caused by an
earlier computation. The package cannot recover the intended exact value from
those bits. For this reason, at least initially, support for float input is not planned, 
`fixed` accepts only explicit inputs:

- `FromInt` for integers.
- `FromRatio` for exact ratios.
- `MustParse` for decimal literals.
- `FromRaw` for an exact Q32.32 bit pattern.

`String` provides the inverse text boundary. It emits a canonical decimal
representation, and every value satisfies:

```go
fixed.MustParse(q.String()) == q
```

## Arithmetic contract

The package uses one explicit rule for each operation:

| Operation | Result |
| --- | --- |
| `Add`, `Sub` | Exact Q32.32 result, with saturation on overflow |
| `Mul` | Product floored to Q32.32, with saturation on overflow |
| `Div`, `FromRatio` | Quotient truncated toward zero, with saturation on overflow |
| `Sqrt` | Square root floored to Q32.32 |
| `Round`, `MustParse` | Nearest representable value; exact halves round away from zero |

Division by zero panics. `Sqrt` of a negative value also panics.

Saturated addition is not associative near the limits. Reordering a sum can
therefore change its result. Accumulators should have enough headroom to avoid
saturation.

Every saturation increments a process-wide atomic counter. `SaturationCount`
provides diagnostics without changing any `Q` value or operation result.

## Architecture

`fixed` is a leaf module. Its production code imports only `math/bits` and
`sync/atomic`. This small dependency surface lets applications use the numeric
type without importing unrelated systems.

The `Q` type is opaque. Constructors control how values enter the package,
operations own saturation and rounding, and `Raw` is the boundary for exact
bit access.

The module is one flat package by design. Every public type shares one
contract, so subpackages would only split the documentation and add import
noise. File names carry the layers: `q*`/`decimal*` for the scalar, `vec2*`
and `rot*` for the plane, `trig*` for the kernel. Directories exist only for
content outside the package interface: `internal/` for tools and `.github/`
for CI.

The Go implementation defines the bit-level contract. An independent
implementation must preserve the rounding, saturation, and raw representation
before it exchanges values with this package. A change to one of these rules is
a semantic change, not an internal refactor.

## Development

Run the standard checks before submitting a change:

```sh
go test ./...
go vet ./...
golangci-lint run ./...
```

See the [package documentation](https://pkg.go.dev/github.com/dhannyell/fixed)
for the API reference and the full behavioral contract.

## License

`fixed` is available under the [MIT License](LICENSE).
