# fixed

`fixed` is a small Go package for signed fixed-point arithmetic. Equal inputs
produce the same result bits on every supported architecture.

The package provides two formats without choosing a default. `Q32` stores
Q32.32 in an `int64`, with resolution 2⁻³² and range
[-2³¹, 2³¹ - 2⁻³²]. `Q16` stores Q16.16 in an `int32`, with resolution 2⁻¹⁶
and range [-2¹⁵, 2¹⁵ - 2⁻¹⁶]. Consumers choose the format that fits their
range and storage requirements.

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
	a := fixed.Vec2{X: fixed.Q32FromInt(1), Y: fixed.Q32FromInt(2)}
	b := fixed.Vec2{X: fixed.Q32FromInt(4), Y: fixed.Q32FromInt(6)}
	distance := a.Distance(b)

	quarterTurn := fixed.RotFromTurns(fixed.Q32FromRatio(1, 4))
	direction := quarterTurn.Apply(fixed.Vec2{X: fixed.Q32One()})

	fmt.Println(distance)                 // 5
	fmt.Println(direction.X, direction.Y) // 0 1
}
```

## Why there is no float constructor

Floating-point input can already contain small differences caused by an
earlier computation. The package cannot recover the intended exact value from
those bits. For this reason, `fixed` accepts only explicit inputs:

- `Q32FromInt` and `Q16FromInt` for integers.
- `Q32FromRatio` and `Q16FromRatio` for exact ratios.
- `Q32MustParse` and `Q16MustParse` for decimal literals.
- `Q32FromRaw` and `Q16FromRaw` for exact bit patterns.

`String` provides the inverse text boundary. It emits a canonical decimal
representation, and every value satisfies:

```go
fixed.Q32MustParse(q32.String()) == q32
fixed.Q16MustParse(q16.String()) == q16
```

## Arithmetic contract

Both formats use the same explicit rule for each operation:

| Operation | Result |
| --- | --- |
| `Add`, `Sub` | Exact result in the selected format, with saturation on overflow |
| `Mul` | Product floored to the selected format, with saturation on overflow |
| `Div`, `Q32FromRatio`, `Q16FromRatio` | Quotient truncated toward zero, with saturation on overflow |
| `Sqrt` | Square root floored to the selected format |
| `Round`, `Q32MustParse`, `Q16MustParse` | Nearest representable value; exact halves round away from zero |
| `Q32.ToQ16` | Floored to Q16.16, with saturation on overflow |

`Q16.ToQ32` is exact and never saturates. Narrowing and division deliberately
use different rules: narrowing floors, while division truncates toward zero.

Division by zero panics. `Sqrt` of a negative value also panics.

Saturated addition is not associative near the limits. Reordering a sum can
therefore change its result. Accumulators should have enough headroom to avoid
saturation.

Every saturation increments a process-wide atomic counter. `SaturationCount`
provides diagnostics without changing any fixed-point value or operation
result.

## Cost of operations

The contract promises bits, not speed. The numbers below are a guide for
design choices on one machine: AMD Ryzen 7 5800X3D (amd64), Go 1.26.4. They
are medians of ten runs with `-benchtime=500ms`; Windows scheduling produced
occasional high outliers that the median excludes.

| Operation | Latency (ns) | Throughput (ns) | Throughput (M op/s) | Fixed / float throughput |
| --- | --- | --- | --- | --- |
| `Q16.Add` | 0.5 | 0.5 | 2,000 | 0.83× |
| `Q16.Mul` | 1.7 | 0.8 | 1,200 | 1.30× |
| `Q16.Div` | 3.3 | 1.1 | 910 | 1.35× |
| `Q16.Sqrt` | 10.3 | 3.2 | 310 | 3.05× |
| `Q32.Add` | 0.4 | 0.5 | 1,900 | 0.73× |
| `Q32.Mul` | 2.8 | 1.4 | 690 | 2.14× |
| `Q32.Div` | 4.7 | 3.1 | 330 | 2.38× |
| `Q32.Sqrt` | 13.6 | 6.1 | 160 | 2.65× |
| `Vec2.Dot` | 3.3 | 3.4 | 290 | 5.35× |
| `Vec2.Len` | 16.0 | 13.1 | 76 | 3.47× |
| `Vec2.Normalize` | 26.4 | 18.3 | 55 | 4.82× |
| `Vec2.Normalize` axial | 12.5 | 11.4 | 87 | 3.80× |
| `Rot.Apply` | 5.5 | 5.9 | 170 | 7.88× |
| `Rot.Mul` | 5.6 | 6.1 | 160 | 8.18× |
| `Rot.Normalize` | 26.7 | 18.5 | 54 | 4.79× |
| `SinTurns` + `CosTurns` | — | 4.3 per pair | 230 pairs | 0.39× |
| `RotFromTurns` | — | 3.5 | 280 | 0.36× |

Read each column alone; the columns measure different situations. Latency is
the cost when each result feeds the next operation, as in an iterative
solver. Throughput is the cost when independent operations overlap in the
pipeline, as in a loop over many values. The rate column is the reciprocal of
the throughput column, rounded to two digits; use it to size a frame budget.
Each latency chain also contains one cheap companion operation that keeps the
value in domain; `bench_test.go` shows the exact chains.

The comparison column comes from paired benchmarks over the same prebuilt
inputs: Q16 is compared with `float32`, while Q32, vectors, and rotations are
compared with `float64`. A value below 1× means fixed was faster; a value above
1× is the fixed-point penalty. These safe-domain float kernels do not reproduce
the package's saturation, rounding, or cross-architecture bit contract. Run
them with:

```sh
go test -run '^$' -bench '^BenchmarkCompare' -benchtime=500ms -count=10
```

Two portability notes. `Div` costs more on arm64, because the 128-bit
division is a software routine there. `Sqrt` does not divide on any
architecture: its hardware seed plus integer corrections stay within
multiplications.

## Vectors and angles

`Vec2` provides the usual 2D operations over `Q32`: addition, scaling, dot
product, length, normalization, distance, and interpolation. `LenSq` follows
the scalar operation order and can saturate even when the length still fits.
`Len` uses a 128-bit intermediate and saturates only when the final magnitude
does not fit. `Normalize` scales the components before squaring them, which
avoids intermediate overflow and underflow.

Angles use turns instead of radians. `Q32One()` is one complete revolution,
`Q32Half()` is half a revolution, and `Q32FromRatio(1, 4)` is a quarter turn.
This maps the fractional bits of `Q32` directly onto the circle and avoids
reduction through an approximation of pi.

`SinTurns`, `CosTurns`, and `Atan2Turns` use committed lookup tables and linear
interpolation. Their maximum absolute error is 2⁻²⁰. `Rot` stores a rotation as
its sine and cosine, which makes application, composition, and inversion
available without another trigonometric lookup. The zero value of `Rot` is not
a valid rotation; start with `RotIdentity` or `RotFromTurns`.

## Architecture

`fixed` is a leaf module. The package imports only `math`, `math/bits`, and
`sync/atomic`. The `math` import provides hardware seeds; exact integer
comparisons close every result, so floating point never decides a bit. This
small dependency surface lets applications use the numeric type without
importing unrelated systems.

The `Q32` and `Q16` types are opaque. Their prefixed constructors make the
chosen format explicit. Operations own saturation and rounding, and `Raw` is
the boundary for exact bit access. Consumers that standardize on one format can
define local aliases without imposing that choice on other users of the
library.

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
