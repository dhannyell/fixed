# fixed

`fixed` is a small Go package for signed fixed-point arithmetic. Equal inputs
produce the same result bits on every supported architecture.

The package provides three formats without choosing a default. `Q32` stores
Q32.32 in an `int64`, with resolution 2⁻³² and range
[-2³¹, 2³¹ - 2⁻³²]. `Q16` stores Q16.16 in an `int32`, with resolution 2⁻¹⁶
and range [-2¹⁵, 2¹⁵ - 2⁻¹⁶]. `Q48` stores Q48.16 in an `int64`, with
resolution 2⁻¹⁶ and range [-2⁴⁷, 2⁴⁷ - 2⁻¹⁶]; it accumulates `Q16` products.
Consumers choose the format that fits their range and storage requirements.

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

- `Q32FromInt`, `Q16FromInt`, and `Q48FromInt` for integers.
- `Q32FromRatio`, `Q16FromRatio`, and `Q48FromRatio` for exact ratios.
- `Q32MustParse`, `Q16MustParse`, and `Q48MustParse` for decimal literals.
- `Q32FromRaw`, `Q16FromRaw`, and `Q48FromRaw` for exact bit patterns.

`String` provides the inverse text boundary. It emits a canonical decimal
representation, and every value satisfies:

```go
fixed.Q32MustParse(q32.String()) == q32
fixed.Q16MustParse(q16.String()) == q16
fixed.Q48MustParse(q48.String()) == q48
```

## Arithmetic contract

All formats use the same explicit rule for each operation:

| Operation | Result |
| --- | --- |
| `Add`, `Sub` | Exact result in the selected format, with saturation on overflow |
| `Mul` | Product floored to the selected format, with saturation on overflow |
| `Div`, `FromRatio` | Quotient truncated toward zero, with saturation on overflow |
| `Sqrt` | Square root floored to the selected format |
| `Round`, `MustParse` | Nearest representable value; exact halves round away from zero |
| `Q48.MulAdd16` | Exact `Q16` product floored to Q48.16, then added with saturation on overflow |
| `Q48.Mul16` | `Q48` times `Q16`, floored to Q48.16 with saturation on overflow; same bits as `Mul` on the widened factor |
| `Q16.ToQ32`, `Q16.ToQ48` | Exact; the grid gets finer and the range gets wider |
| `Q32.ToQ16` | Floored to the coarser grid, with saturation outside the narrower range |
| `Q32.ToQ48` | Floored to the coarser grid; the range gets wider, so no saturation |
| `Q48.ToQ16` | Exact on the shared grid, with saturation outside the narrower range |
| `Q48.ToQ32` | Exact on the finer grid, with saturation outside the narrower range |

A conversion applies two independent rules. A finer fraction grid is exact
and a coarser one floors. A wider integer range never saturates and a
narrower one saturates. `Q48.ToQ32` refines the grid and narrows the range
at the same time. Conversion and division deliberately use different
rounding: a coarser grid floors, while division truncates toward zero.

`Q48.Int`, `Q48FromInt`, and `Q48FromRatio` use `int64`. The 48-bit integer
range does not fit an `int` on 32-bit architectures, and a truncated `int`
would break determinism between architectures. `Q32` and `Q16` keep `int`,
because their integer range fits 32 bits.

`Q48` exists for sums of `Q16` products. A product has at most 32 integer
bits, and `MulAdd16` keeps 16 bits of headroom above it, so a sum of up to
2¹⁶ full-range products cannot saturate:

```go
var acc fixed.Q48
for i := range a {
	acc = acc.MulAdd16(a[i], b[i])
}
dot := acc.ToQ16() // Narrow only when the value is stored.
```

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

| Operation | Latency (ns) | Throughput (ns) | Throughput (M op/s) | Fixed time vs float |
| --- | --- | --- | --- | --- |
| `Q16.Add` | 0.5 | 0.5 | 2,000 | −17% (faster) |
| `Q16.Mul` | 1.7 | 0.8 | 1,200 | +30% (slower) |
| `Q16.Div` | 3.3 | 1.1 | 910 | +35% (slower) |
| `Q16.Sqrt` | 10.3 | 3.2 | 310 | +205% (slower) |
| `Q32.Add` | 0.4 | 0.5 | 1,900 | −27% (faster) |
| `Q32.Mul` | 2.8 | 1.4 | 690 | +114% (slower) |
| `Q32.Div` | 4.7 | 3.1 | 330 | +138% (slower) |
| `Q32.Sqrt` | 13.6 | 6.1 | 160 | +165% (slower) |
| `Q48.Add` | 0.5 | 0.5 | 2,100 | −30% (faster) |
| `Q48.Mul` | 2.6 | 1.5 | 680 | +105% (slower) |
| `Q48.MulAdd16` | 1.0 | 0.7 | 1,500 | +73% (slower) |
| `Q48.Div` | 4.3 | 5.1 | 200 | +270% (slower) |
| `Q48.Sqrt` | 11.4 | 5.5 | 180 | +128% (slower) |
| `Vec2.Dot` | 3.3 | 3.4 | 290 | +435% (slower) |
| `Vec2.Len` | 16.0 | 13.1 | 76 | +247% (slower) |
| `Vec2.Normalize` | 26.4 | 18.3 | 55 | +382% (slower) |
| `Vec2.Normalize` axial | 12.5 | 11.4 | 87 | +280% (slower) |
| `Rot.Apply` | 5.5 | 5.9 | 170 | +688% (slower) |
| `Rot.Mul` | 5.6 | 6.1 | 160 | +718% (slower) |
| `Rot.Normalize` | 26.7 | 18.5 | 54 | +379% (slower) |
| `SinTurns` + `CosTurns` | — | 4.3 per pair | 230 pairs | −61% (faster) |
| `RotFromTurns` | — | 3.5 | 280 | −64% (faster) |
| `Atan2Turns` | — | 3.9 | 250 | −54% (faster) |

Read each column alone; the columns measure different situations. Latency is
the cost when each result feeds the next operation, as in an iterative
solver. Throughput is the cost when independent operations overlap in the
pipeline, as in a loop over many values. The rate column is the reciprocal of
the throughput column, rounded to two digits; use it to size a frame budget.
Each latency chain also contains one cheap companion operation that keeps the
value in domain; `bench_test.go` shows the exact chains.

The comparison column reports the raw change in throughput time:
`(fixed time / float time - 1) × 100`. A negative value means fixed was faster;
a positive value means it was slower. The paired benchmarks use the same
prebuilt inputs. Q16 is compared with `float32`; Q32, Q48, vectors, and
rotations are compared with `float64`. `Q48.MulAdd16` is compared with a
`float64` sum of `float32` products, the shape a float solver uses for the
same dot product. These safe-domain float kernels do not reproduce
the package's saturation, rounding, or cross-architecture bit contract. Run
them with:

```sh
go test -run '^$' -bench '^BenchmarkCompare' -benchtime=500ms -count=10
```

The batch functions are measured separately, because they are compared with a
loop rather than with a float. The numbers are nanoseconds per element at 1024
elements, medians of twenty runs over two sessions on the same machine. The
`per-call loop` column writes `dst[i] = a[i].Op(b[i])` by hand; `scalar` is the
exported batch function in a default build; `avx2` is the same exported call
in a build with `GOEXPERIMENT=simd` on Go 1.27. The benchmark goes through the
exported function, so the length check and the counter update are inside the
number.

| Operation | per-call loop | scalar | avx2 |
| --- | --- | --- | --- |
| `BatchAdd16` | 0.98 | 0.71 | 0.35 |
| `BatchSub16` | 0.97 | 0.72 | 0.35 |
| `BatchMul16` | 0.96 | 0.85 | 0.44 |
| `BatchClamp16` | 1.30 | 1.06 | 0.16 |
| `BatchQ32FromQ16` | 0.50 | 0.38 | 0.27 |
| `BatchQ16FromQ32` | 0.75 | 0.42 | 0.38 |
| `BatchDot16` | 1.18 | 1.40 | 0.57 |
| `BatchQ48Mul16` | 1.51 | 1.48 | 0.73 |

In these measurements, every scalar `Q16` batch function was faster than its
hand-written loop. The default build therefore had no batch abstraction penalty
for this workload. `BatchDot16` is the exception: its scalar kernel keeps eight
partial sums so that every path shares one order, and that costs more than a
serial loop when nothing saturates. The `per-call loop` for `BatchDot16` is a
serial `Q48.MulAdd16` accumulator; for `BatchQ48Mul16` it is `Q48.Mul16`. arm64 has a NEON path for the six `Q16` functions. Its numbers are not
published here because only shared CI runners have measured it, and a shared
runner cannot support the comparison above.

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

## Batch operations

`BatchAdd16`, `BatchSub16`, `BatchMul16`, and `BatchClamp16` apply one
operation across whole slices of `Q16`. Every slice in a call must share one
length. The destination may be the same slice as a source, so an operation can
run in place; any other overlap is undefined. `BatchQ32FromQ16` and
`BatchQ16FromQ32` move whole slices across the format boundary and follow the
conversion rules of `Q16.ToQ32` and `Q32.ToQ16`. A batch call adds the number
of saturated elements to the saturation counter in one update, so
`SaturationCount` reports the same total as a loop over the scalar methods.

`BatchDot16` sums `Q16` products into one `Q48`. Its order is fixed: element
`i` joins partial sum `i mod 8`, and the eight partials reduce as a balanced
tree. Without saturation this equals a loop over `Q48.MulAdd16`. With
saturation the order decides the bits, so every kernel keeps it, and the result
is the same on every host. `BatchQ48Mul16` scales a `Q48` slice by a `Q16`
slice with the rules of `Q48.Mul16`.

Every build runs the scalar kernels. Building with `GOEXPERIMENT=simd` on Go
1.27 or later selects vector kernels at package initialization: AVX2 on amd64
when the CPU reports it, and NEON on arm64. `BatchPath` returns `"scalar"`,
`"avx2"`, or `"neon"` so a program can report which family it got.

```sh
GOEXPERIMENT=simd go build ./...
```

CI compares the result bits and saturation counts of the AVX2 and NEON kernels
with the scalar kernels.

## Architecture

`fixed` is a leaf module. The portable files import only `math`, `math/bits`,
and `sync/atomic`; files behind the `goexperiment.simd` build tag also use
`unsafe` and `simd/archsimd`. Every `unsafe` operation lives in one file,
`batch16_raw.go`, which only reinterprets a `Q16` or `Q32` slice as the raw
words the vector loads take, under compile-time assertions on the layout. The
`math` import provides hardware seeds; exact integer
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
