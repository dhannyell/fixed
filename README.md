# fixed

`fixed` is a Go library for signed fixed-point arithmetic. It gives the same
result bits for the same inputs on every supported architecture, with explicit
rules for rounding and overflow.

It includes three numeric formats, 2D vectors, rotations, trigonometry, and
batch operations. Choose the format that fits your data; the library has no
default numeric type.

Requires **Go 1.26.4 or newer**. The module is pre-v1, and its import path may
change before the first stable release.

```sh
go get github.com/dhannyell/fixed
```

This lib has a WGSL version that reproduces the same bits as the GO version on GPU [fixed-wgsl](https://github.com/dhannyell/fixed-wgsl).

## Getting started

Construct values from integers, ratios, decimal strings, or raw bits. Methods
return new values, so arithmetic reads much like the calculation itself:

```go
package main

import (
	"fmt"

	"github.com/dhannyell/fixed"
)

func main() {
	a := fixed.Vec2{X: fixed.Q32FromInt(1), Y: fixed.Q32FromInt(2)}
	b := fixed.Vec2{X: fixed.Q32FromInt(4), Y: fixed.Q32FromInt(6)}

	fmt.Println(a.Distance(b)) // 5

	quarterTurn := fixed.RotFromTurns(fixed.Q32FromRatio(1, 4))
	direction := quarterTurn.Apply(fixed.Vec2{X: fixed.Q32One()})
	fmt.Println(direction.X, direction.Y) // 0 1
}
```

See the [package documentation](https://pkg.go.dev/github.com/dhannyell/fixed)
for the full API.

## Choosing a format

A fixed-point value is an integer with a fixed scale. For example, one raw
unit in `Q16` represents 1/65536. The format determines both the smallest step
you can represent and how large a value can get.

| Type | Format | Storage | Smallest step | Range |
| --- | --- | --- | --- | --- |
| `Q16` | Q16.16 | `int32` | 2⁻¹⁶ | −2¹⁵ to 2¹⁵ − 2⁻¹⁶ |
| `Q32` | Q32.32 | `int64` | 2⁻³² | −2³¹ to 2³¹ − 2⁻³² |
| `Q48` | Q48.16 | `int64` | 2⁻¹⁶ | −2⁴⁷ to 2⁴⁷ − 2⁻¹⁶ |

Use `Q16` when compact storage and 16 fractional bits are enough. `Q32` gives
you more fractional precision and is the format used by vectors and rotations.
`Q48` keeps the resolution of `Q16` while providing more room for large values
and accumulated products.

The underlying fields are private. Use constructors and `Raw` to move between
numeric values and their stored representation.

### Creating and printing values

Each format has the same constructor names:

| Input | Q32 example | Meaning |
| --- | --- | --- |
| Integer | `fixed.Q32FromInt(3)` | The whole number 3 |
| Ratio | `fixed.Q32FromRatio(1, 3)` | 1/3, truncated to the Q32 grid |
| Decimal | `fixed.Q32MustParse("0.25")` | A decimal rounded to the nearest representable value |
| Raw bits | `fixed.Q32FromRaw(1)` | One raw unit, or 2⁻³² |

Replace `Q32` with `Q16` or `Q48` to use another format. `Q48FromInt` and
`Q48FromRatio` take `int64` arguments, and `Q48.Int` returns `int64`; its range
does not fit a 32-bit `int`. The corresponding Q16 and Q32 APIs use `int`.

There is no float constructor. A floating-point calculation may already have
introduced rounding differences before the value reaches this library.
Integers, ratios, text, and raw bits make that input boundary explicit.

`String` produces a canonical decimal that parses back to the same value:

```go
fixed.Q32MustParse(q32.String()) == q32
fixed.Q16MustParse(q16.String()) == q16
fixed.Q48MustParse(q48.String()) == q48
```

## Rounding and overflow

Overflow clamps a result to the format's minimum or maximum value. This is
called *saturation*. A build with the `fixed_satcounter` tag also increments
the process-wide `SaturationCount` counter, which you can use for diagnostics
without affecting the calculation.

### Enabling the saturation counter

Counting is off by default. Add the `fixed_satcounter` build tag when you want
the diagnostic counter:

```sh
go build -tags=fixed_satcounter ./...
```

The same tag works for WebAssembly. In PowerShell:

```powershell
$env:GOOS = "js"
$env:GOARCH = "wasm"
go build -tags=fixed_satcounter ./...
```

Overflow clamps to the same limits and every numeric result keeps the same
bits either way. The tag adds diagnostic increments, counter access, and local
event counting in batch kernels. The checks that saturate arithmetic are
always present. There is no runtime switch to check on each operation.

On `js/wasm` and `wasip1` the counter is a plain variable rather than an
atomic. Those targets run on one thread without asynchronous preemption, so
an increment cannot be interrupted, and the atomic call would otherwise keep
the scalar methods from inlining on WebAssembly.

`SaturationCountingEnabled` is a compile-time constant. Without the tag it is
`false`, `SaturationCount()` always returns zero, and `ResetSaturationCount()`
does nothing. You can combine the tag with `GOEXPERIMENT=simd`.

Before v0.9.0 the counter was on by default and `fixed_nosatcounter` removed
it. If you read `SaturationCount()`, add `fixed_satcounter`. If you passed
`fixed_nosatcounter`, drop it; the new default already leaves the counter out.

To measure what the counter costs on your target, run the same benchmark both
ways:

```sh
go test -run '^$' -bench '^BenchmarkSaturationOverhead' -count=10 .
go test -tags=fixed_satcounter -run '^$' -bench '^BenchmarkSaturationOverhead' -count=10 .
```

These tests include safe and saturating inputs. Scalar counting happens only
when an operation saturates; batch kernels can also spend time collecting
events locally. The cost depends on the workload and target.

### Arithmetic rules

| Operation | Rounding rule |
| --- | --- |
| `Add`, `Sub` | Exact when the result fits |
| `Mul` | Round down to the format's grid |
| `Q16.MulRound` | Nearest step; exact halves go toward positive infinity |
| `Div`, `FromRatio` | Truncate toward zero |
| `Sqrt` | Round down to the format's grid |
| `Round` | Nearest integer; exact halves go away from zero |
| `MustParse` | Nearest representable value; exact halves go away from zero |
| `Int` | Integer part, truncated toward zero |

Rounding down and truncating toward zero differ for negative values. The
library keeps that distinction: multiplication rounds down, while division
truncates toward zero. Results outside the target range saturate.

The nearest-step operations break an exact half toward positive infinity, not
away from zero like `Round`. A negative half therefore moves toward zero.

Division by zero, a zero denominator in `FromRatio`, and the square root of a
negative value panic.

Saturation also makes the order of addition matter. Near the limits,
`a.Add(b).Add(c)` can differ from `a.Add(b.Add(c))`. Leave enough room in an
accumulator if you need to avoid that effect.

### Converting between formats

Conversions preserve a value exactly when the destination has enough range
and fractional precision. Dropping fractional bits rounds down; exceeding the
destination's range saturates.

| Conversion | Behavior |
| --- | --- |
| `Q16.ToQ32` | Exact; more range and finer resolution |
| `Q16.ToQ48` | Exact; more range, same resolution |
| `Q32.ToQ16` | Rounds down; saturates outside the Q16 range |
| `Q32.ToQ16Round` | Nearest step; saturates outside the Q16 range |
| `Q32.ToQ48` | Rounds down; the wider range needs no saturation |
| `Q32.ToQ48Round` | Nearest step; the wider range needs no saturation |
| `Q48.ToQ16` | Same resolution; saturates outside the Q16 range |
| `Q48.ToQ32` | Exact when in range; saturates outside the Q32 range |

### Accumulating Q16 products

`Q48.MulAdd16` multiplies two Q16 values, rounds the product down to Q48.16,
and adds it to the accumulator with saturation. Starting from zero, a sum of
up to 2¹⁶ full-range Q16 products fits in Q48.

```go
var acc fixed.Q48
for i := range a {
	acc = acc.MulAdd16(a[i], b[i])
}
dot := acc.ToQ16() // Convert back when you need the narrower value.
```

`Q48.Mul16` multiplies a Q48 value by a Q16 factor. It rounds down and
saturates just like `Mul` with the factor converted to Q48.

## Vectors, rotations, and angles

`Vec2` holds two Q32 components and supports addition, scaling, dot products,
length, normalization, distance, and interpolation.

Three details matter near the numeric limits:

- `LenSq` uses scalar multiplication and addition, so it can saturate even
  when the length itself fits. `Len` uses a 128-bit intermediate and saturates
  only if the final magnitude is too large.
- `Normalize` rescales components before squaring them to avoid intermediate
  overflow and underflow.
- `Lerp` evaluates `Sub`, `Mul`, and `Add` with their usual saturation rules.
  If `target - v` overflows, `t=1` may miss the target and `t=0` still records
  that intermediate overflow. Keep the intermediate values in range when
  exact endpoints matter.

Angles are measured in **turns**: `Q32One()` is a full revolution,
`Q32Half()` is a half turn, and `Q32FromRatio(1, 4)` is a quarter turn.
This represents a circle directly in the fractional bits without reducing
angles through an approximation of pi.

`SinTurns`, `CosTurns`, and `Atan2Turns` use committed lookup tables with
linear interpolation. Sine and cosine have a maximum absolute error of 2⁻²⁰;
the angle returned by `Atan2Turns` has a maximum absolute error of 2⁻²⁰ turns.
Its output is in `[-1/2, 1/2]`: the negative x axis returns `+1/2`, while
values just below it can round to `-1/2`.

`Rot` stores sine and cosine together. You can apply, compose, or invert a
rotation without another trigonometric lookup. Construct it with
`RotIdentity` or `RotFromTurns`; the zero value is not a valid rotation.
`Inv` is the conjugate and acts as an inverse for a unit rotation. Use
`InvNormalized` when accumulated rounding drift also needs to be corrected.

## Working with slices

The batch API applies arithmetic or conversions to whole slices:

| Functions | Work performed |
| --- | --- |
| `BatchAdd16`, `BatchSub16`, `BatchMul16`, `BatchClamp16` | Element-wise Q16 arithmetic and clamping |
| `BatchQ32FromQ16`, `BatchQ16FromQ32` | Format conversion using the scalar conversion rules |
| `BatchDot16` | Sum of Q16 products in a Q48 accumulator |
| `BatchQ48Mul16` | Element-wise Q48 multiplication by Q16 factors |

All slices in a call must have the same length. For element-wise operations
on the same format, the destination can be exactly the same slice as a source.
Other overlap is undefined.

Batch arithmetic records saturation events in one counter update per call
when needed. Element-wise results and saturation totals match the scalar
operations.

`BatchDot16` uses eight partial sums: element `i` goes into partial `i mod 8`,
then the partials are combined in a balanced tree. Every implementation uses
this order. It gives the same result as a serial `MulAdd16` loop when no
intermediate sum saturates; saturation can make those two orders differ.

### SIMD support

Default builds use scalar kernels. With Go 1.27 or newer and
`GOEXPERIMENT=simd`, the package selects AVX2 on supported amd64 CPUs or NEON
on arm64 at initialization:

```sh
GOEXPERIMENT=simd go build ./...
```

`BatchPath()` reports `"scalar"`, `"avx2"`, or `"neon"`. NEON covers the six
Q16 arithmetic and conversion functions; `BatchDot16` and `BatchQ48Mul16`
remain scalar on arm64. CI checks that vector kernels preserve the scalar
results and saturation counts.

## Lanes

The lane API keeps a fixed number of independent values together while a
calculation is in progress. Load an array or splat one value, compose lane
operations, then store the result:

```go
var a, b, dst [fixed.LaneWidth]fixed.Q16

la := fixed.LoadLane16(&a)
lb := fixed.LoadLane16(&b)
bias := fixed.SplatLane16(fixed.Q16Half())
la.Mul(lb).Add(bias).Store(&dst)
```

`LaneWidth` is 8 for the AVX2 path and 4 for the NEON and generic paths.
`LanePath()` reports `"avx2"`, `"neon"`, or `"generic"`. In an amd64 SIMD
build, check `LanesAvailable()` before using the lane API; it reports false on
a CPU without AVX2, where calling lane operations is undefined.

| Type | Operations |
| --- | --- |
| `Lane16` | `SplatLane16`, `LoadLane16`, `Store`, `Add`, `Sub`, `Neg`, `AddWrap`, `SubWrap`, `Mul`, `MulRound`, `ScaleDown`, `ScaleDownRound`, `MulAdd`, `MulSub`, `Min`, `Max`, `SymClamp`, `Greater`, `Equals`, `ToLane48` |
| `Mask16` | `Or`, `AllZero`, `BlendLane16` |
| `Shift16` | `SplatShift16`, `LoadShift16` |
| `Lane48` | `SplatLane48`, `LoadLane48`, `Store`, `Add`, `Sub`, `AddWrap`, `SubWrap`, `MulAdd16`, `MulAdd16Round`, `ToLane16` |

`MulAdd` and `MulSub` perform two ordered operations rather than a fused
operation. `Mul` and `MulAdd16` round down, while the `Round` forms take the
nearest step. Every operation follows its scalar Q16 or Q48 saturation rules.

`ScaleDown` and `ScaleDownRound` divide by a power of two that each lane picks
for itself, which a single instruction does on both SIMD paths. Up to an
amount of 16 they give the same bits as `Mul` and `MulRound` by that power of
two. Past 16 the divisor leaves the Q16 grid: the multiply gives zero, while
the shift keeps going. An amount above 31 acts like 31.

`AddWrap` and `SubWrap` skip the overflow check on both types. A result
outside the format wraps and records no saturation event, so the caller must
bound the operands itself. They serve a caller that can prove the bound from
the surrounding code and does not want to pay for the check. The AVX2, NEON, and generic paths produce
identical result bits and saturation counts.

## Performance

Performance depends on how the operations are used. The tables below separate
three questions:

| Measurement | What it tells you |
| --- | --- |
| Dependent operations | The time for a step that needs the previous result |
| Independent inputs | The throughput of a loop whose arithmetic can overlap |
| Representative workloads | The cost of a complete numerical task, including loops and data access |

All measurements used an AMD Ryzen 7 5800X3D, Windows/amd64, and Go 1.26.4.
The first two tables were measured on 2026-09-05; the workloads on 2026-09-06.
Each value is the median of ten runs at 500 ms per benchmark. Batch results
use a separate procedure described below.
All published tables below were measured with saturation counting enabled.

These are measurements of specific loops and inputs. They include compiler
and machine effects, and do not establish that fixed-point arithmetic is
generally faster or slower than float. Sub-nanosecond results are particularly
sensitive to loop overhead and code placement.

### Dependent operations

In [`Benchmark*Latency`](bench_test.go), each result feeds the next iteration.
Some tests need extra arithmetic to keep values in range: the multiplication
test, for example, measures `Mul + Add`. The last column lists the complete
step. Loop control is included in every row.

| Chain | Fixed ns/step | Work included in each step |
| --- | --- | --- |
| `Q16.Add` | 0.543 | Add |
| `Q16.Mul` | 1.89 | Mul + Add |
| `Q16.Div` | 3.37 | Div + Add |
| `Q16.Sqrt` | 9.9 | Sqrt + Mul + Add + input update |
| `Q32.Add` | 0.382 | Add |
| `Q32.Mul` | 2.74 | Mul + Add |
| `Q32.Div` | 4.76 | Div + Add |
| `Q32.Sqrt` | 11.3 | Sqrt + Mul + Add + input update |
| `Q48.Add` | 0.507 | Add |
| `Q48.Mul` | 2.6 | Mul + Add |
| `Q48.MulAdd16` | 0.86 | MulAdd16 + input update |
| `Q48.Div` | 4.49 | Div + Add |
| `Q48.Sqrt` | 11.1 | Sqrt + Mul + Add + input update |
| `Vec2.Dot` | 4.51 | Dot + vector reconstruction |
| `Vec2.Len` | 16.5 | Len + two Mul + vector reconstruction |
| `Vec2.Normalize` | 27.1 | Normalize + two Mul + component swap |
| `Vec2.Normalize` axial | 3.97 | Normalize + Mul + vector reconstruction |
| `Rot.Apply` | 5.89 | Apply |
| `Rot.Mul` | 6.05 | Mul |
| `Rot.Normalize` | 26.4 | Normalize + two Mul + component swap |

These are chain costs, not isolated instruction latencies. This table has no
float comparison.

```sh
go test -run '^$' -bench '^Benchmark(Q16|Q32|Q48|Vec2|Rot).*Latency$' -benchtime=500ms -count=10 .
```

### Independent inputs

[`BenchmarkCompare`](float_compare_test.go) reads prepared inputs and rotates
between four accumulators, allowing more work to overlap. Each iteration
evaluates one named operation; the sine/cosine row evaluates one pair.
Array reads, indexing, accumulation, and loop control are timed. Combining
the four accumulators at the end is excluded.

The accumulators still have dependencies. In particular, `Q48.MulAdd16` uses
them as operation inputs, so its row measures four interleaved accumulation
chains. The table measures the throughput of these loops, not bare arithmetic
instructions. To convert ns/iteration to millions of iterations per second,
use `1000 / ns`.

| Independent-input kernel | Fixed ns/iteration | Float ns/iteration | Fixed time vs float in this loop |
| --- | --- | --- | --- |
| `Q16.Add` | 0.885 | 0.767 | +15% (slower) |
| `Q16.Mul` | 1.25 | 0.534 | +134% (slower) |
| `Q16.Div` | 1.29 | 0.827 | +56% (slower) |
| `Q16.Sqrt` | 2.61 | 1.21 | +115% (slower) |
| `Q32.Add` | 0.654 | 0.632 | +3% (slower) |
| `Q32.Mul` | 1.96 | 0.597 | +228% (slower) |
| `Q32.Div` | 3.93 | 1.12 | +251% (slower) |
| `Q32.Sqrt` | 3.96 | 2.09 | +89% (slower) |
| `Q48.Add` | 0.691 | 0.625 | +11% (slower) |
| `Q48.Mul` | 2.02 | 0.619 | +227% (slower) |
| `Q48.MulAdd16` | 1.47 | 0.583 | +151% (slower) |
| `Q48.Div` | 4.09 | 1.08 | +277% (slower) |
| `Q48.Sqrt` | 4.26 | 2.09 | +103% (slower) |
| `Vec2.Dot` | 5.54 | 0.945 | +486% (slower) |
| `Vec2.Len` | 9.51 | 2.17 | +338% (slower) |
| `Vec2.Normalize` | 20.7 | 4.32 | +378% (slower) |
| `Vec2.Normalize` axial | 3.69 | 3.21 | +15% (slower) |
| `Rot.Apply` | 8.16 | 1.22 | +567% (slower) |
| `Rot.Mul` | 8.59 | 1.19 | +620% (slower) |
| `Rot.Normalize` | 21.5 | 4.34 | +396% (slower) |
| `SinTurns` + `CosTurns` | 5.22 | 11.5 | -54% (faster) |
| `RotFromTurns` | 4.5 | 11.7 | -62% (faster) |
| `Atan2Turns` | 5.75 | 12.4 | -53% (faster) |

The percentage is `(fixed median / float median - 1) × 100`, calculated before
rounding the displayed times. Negative means fixed took less time; positive
means it took more time in this test.

Q16 is compared with float32. The other formats, vectors, and rotations use
float64. The Q48.MulAdd16 reference multiplies in float32 and accumulates the
products in float64. These float versions do not reproduce fixed-point
rounding, saturation, or bit guarantees.

Older README results used a single accumulator, which limited float in
particular. Those percentages are not directly comparable with this table.
The older `Benchmark*Throughput` tests in `bench_test.go` also include input
generation and a serial accumulation; they are not used here.

```sh
go test -run '^$' -bench '^BenchmarkCompare' -benchtime=500ms -count=10 .
```

### Representative workloads

The [workload benchmarks](workload_bench_test.go) measure two complete
numerical tasks. Both reuse 256-element arrays that fit in cache, avoid
saturation, and allocated no memory during the measured work. They model
common usage patterns; they are not measurements of a complete application.

**Transform256** rotates and translates 256 Q32 points into an output array.
The float64 version uses the same coordinates and rotation coefficients.
The measurement includes arithmetic, calls, loops, reads, and writes. Input
preparation and construction of the rotation are excluded.

**Dot256** reduces 256 Q16 pairs into a Q48 accumulator. Each task starts
from zero and uses one accumulator, because that dependency is part of this
calculation. The float version reads float32 arrays, promotes the inputs,
and multiplies and accumulates in float64. The fixed version rounds each
product down to Q48.16, so the results are not bit-equivalent. Input
preparation and the final benchmark result assignment are excluded.

| Task (256 elements) | Fixed ns/task | Float ns/task | Fixed task time vs float |
| --- | --- | --- | --- |
| Transform256 | 2110 | 271 | +677% (slower) |
| Dot256 | 351 | 193 | +82% (slower) |

```sh
go test -run '^$' -bench '^BenchmarkWorkload' -benchmem -benchtime=500ms -count=10 .
```

For larger datasets, frequent saturation, or application-level decisions,
measure the workload you actually intend to run. To compare library versions,
use the same benchmark source and Go version, alternate runs on the same
machine, and analyze repeated samples with a tool such as benchstat. Keep the
absolute fixed and float times: a changed percentage alone does not show which
side changed.

### Batch operations

This table compares batch calls with hand-written loops. Values are **ns per
element** for 1024 elements, using medians of twenty runs over two sessions
on the same machine. Scalar results use the default build; AVX2 results use
Go 1.27 with `GOEXPERIMENT=simd`. The batch measurements include the exported
call, length checks, and any saturation-counter update.

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

The hand-written loops call the corresponding scalar method for each element.
For `BatchDot16`, the reference is a serial `Q48.MulAdd16` loop; for
`BatchQ48Mul16`, it is `Q48.Mul16`.

Scalar batches took less time than those loops in this workload, except for
`BatchDot16`. Its eight partial sums preserve the same reduction order across
implementations, but cost more here than the serial reference.

NEON timings are omitted because the available measurements came from shared
CI runners. `BatchDot16` and `BatchQ48Mul16` currently stay scalar on arm64;
the tested vector candidates did not meet the project's twofold speedup
threshold. Scalar division also costs more on arm64 because its 128-bit
division uses a software routine. Square root uses a hardware seed followed
by integer corrections, without division.

## Implementation and compatibility

The library is one package with no external runtime dependencies. Portable
code imports `math`, `math/bits`, and `sync/atomic`. Some operations use a
floating-point seed, then check and correct it with integer arithmetic so the
final bits follow the fixed-point contract.

SIMD builds also use `simd/archsimd` and `unsafe`. All unsafe operations are
in [`batch16_raw.go`](batch16_raw.go), where Q16, Q32, and Q48 slices are viewed
as their underlying integer words. Compile-time checks enforce their sizes.

Scalar arithmetic and decimal conversion live in `q*` and `decimal*` files;
vectors, rotations, and trigonometry live in `vec2*`, `rot*`, and `trig*`.
The constructors keep the format explicit. Applications can define local
aliases if they choose to standardize on one type.

Raw representation, rounding, and saturation are part of the public contract.
An independent implementation must preserve all three to exchange values
reliably. Changing them is a compatibility change.

## Development

Run the checks with:

```sh
go test ./...
go vet ./...
go run ./internal/gentable -check
golangci-lint run ./...
```

The trigonometric tables and outputs between table entries have fixed bit
checksums. `gentable -check` regenerates tables for comparison without changing
the files; CI runs it on amd64. Updating the checksums requires an intentional
compatibility change, even if the numerical error remains within tolerance.

Batch benchmarks cover steady workloads, empty slices, vector boundaries,
scalar tails, and different amounts of saturation:

```sh
go test -run '^$' -bench '^BenchmarkBatch$' -benchmem ./...
go test -run '^$' -bench '^BenchmarkBatch(Boundaries|Saturation)$' -benchmem ./...
```

Their MB/s metric counts logical slice reads and writes, not physical memory
traffic: 12 bytes per element for add/sub/mul and conversions, 8 for clamp and
dot, and 20 for Q48Mul16. The dot result and benchmark result storage are
excluded from that count.

## License

[MIT](LICENSE).
