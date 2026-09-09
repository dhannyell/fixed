// Command fixedtrace writes one text trace per operation. Every line carries
// the raw inputs and the raw results of one case.
//
// Usage:
//
//	go run github.com/dhannyell/fixed/cmd/fixedtrace@v0.8.0 -o dir
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/dhannyell/fixed"
)

type column string

const (
	colQ16  column = "q16"
	colQ32  column = "q32"
	colQ48  column = "q48"
	colMask column = "mask"
	colSat  column = "sat"
)

// Q16.16 raw values around the sign, the unit, and the ends of the range.
var edges16 = []int64{
	-2147483648, -2147483647, -0x1_0000, -0x8000, -1, 0,
	1, 0x8000, 0x1_0000, 2147483646, 2147483647, 0x1234_5678,
}

// Q48.16 raw values around the sign, the Q16 range, and the 48-bit range.
var edges48 = []int64{
	-9223372036854775808, -9223372036854775807,
	-(1 << 48), -(1 << 47),
	-2147483649, -2147483648, -2147483647,
	-0x1_0000, -0x8000, -1, 0, 1, 0x8000, 0x1_0000,
	2147483646, 2147483647, 2147483648,
	1 << 47, 1 << 48,
	9223372036854775806, 9223372036854775807,
	0x1234_5678_9abc_def0,
}

// Q32.32 raw values around the sign, the unit, and the Q48 boundary.
var edges32 = []int64{
	-9223372036854775808, -9223372036854775807,
	-(1 << 48), -(1 << 32), -(1 << 31), -1, 0, 1,
	1 << 31, 1 << 32, 1 << 48,
	9223372036854775806, 9223372036854775807,
	0x1234_5678_9abc_def0,
}

type traceOp struct {
	name string
	in   []column
	out  []column
	skip func(in []int64) bool
	eval func(in []int64) []int64
}

func q16(v int64) fixed.Q16 { return fixed.Q16FromRaw(int32(v)) }
func q32(v int64) fixed.Q32 { return fixed.Q32FromRaw(v) }
func q48(v int64) fixed.Q48 { return fixed.Q48FromRaw(v) }

func mask(b bool) []int64 {
	if b {
		return []int64{-1}
	}
	return []int64{0}
}

var ops = []traceOp{
	{name: "add16", in: []column{colQ16, colQ16}, out: []column{colQ16, colSat},
		eval: func(v []int64) []int64 { return []int64{int64(q16(v[0]).Add(q16(v[1])).Raw())} }},
	{name: "sub16", in: []column{colQ16, colQ16}, out: []column{colQ16, colSat},
		eval: func(v []int64) []int64 { return []int64{int64(q16(v[0]).Sub(q16(v[1])).Raw())} }},
	{name: "mul16", in: []column{colQ16, colQ16}, out: []column{colQ16, colSat},
		eval: func(v []int64) []int64 { return []int64{int64(q16(v[0]).Mul(q16(v[1])).Raw())} }},
	{name: "div16", in: []column{colQ16, colQ16}, out: []column{colQ16, colSat},
		skip: func(v []int64) bool { return v[1] == 0 },
		eval: func(v []int64) []int64 { return []int64{int64(q16(v[0]).Div(q16(v[1])).Raw())} }},
	{name: "sqrt16", in: []column{colQ16}, out: []column{colQ16, colSat},
		skip: func(v []int64) bool { return v[0] < 0 },
		eval: func(v []int64) []int64 { return []int64{int64(q16(v[0]).Sqrt().Raw())} }},
	{name: "min16", in: []column{colQ16, colQ16}, out: []column{colQ16, colSat},
		eval: func(v []int64) []int64 { return []int64{int64(q16(v[0]).Min(q16(v[1])).Raw())} }},
	{name: "max16", in: []column{colQ16, colQ16}, out: []column{colQ16, colSat},
		eval: func(v []int64) []int64 { return []int64{int64(q16(v[0]).Max(q16(v[1])).Raw())} }},
	{name: "greater16", in: []column{colQ16, colQ16}, out: []column{colMask, colSat},
		eval: func(v []int64) []int64 { return mask(q16(v[0]).Greater(q16(v[1]))) }},
	{name: "equals16", in: []column{colQ16, colQ16}, out: []column{colMask, colSat},
		eval: func(v []int64) []int64 { return mask(q16(v[0]).Eq(q16(v[1]))) }},
	// The shader blends with the mask from greater, which is Max by definition.
	{name: "blend16", in: []column{colQ16, colQ16}, out: []column{colQ16, colSat},
		eval: func(v []int64) []int64 { return []int64{int64(q16(v[0]).Max(q16(v[1])).Raw())} }},

	{name: "add48", in: []column{colQ48, colQ48}, out: []column{colQ48, colSat},
		eval: func(v []int64) []int64 { return []int64{q48(v[0]).Add(q48(v[1])).Raw()} }},
	{name: "sub48", in: []column{colQ48, colQ48}, out: []column{colQ48, colSat},
		eval: func(v []int64) []int64 { return []int64{q48(v[0]).Sub(q48(v[1])).Raw()} }},
	{name: "mul_add48", in: []column{colQ48, colQ16, colQ16}, out: []column{colQ48, colSat},
		eval: func(v []int64) []int64 {
			return []int64{q48(v[0]).MulAdd16(q16(v[1]), q16(v[2])).Raw()}
		}},

	{name: "to16_48", in: []column{colQ48}, out: []column{colQ16, colSat},
		eval: func(v []int64) []int64 { return []int64{int64(q48(v[0]).ToQ16().Raw())} }},
	{name: "to16_32", in: []column{colQ32}, out: []column{colQ16, colSat},
		eval: func(v []int64) []int64 { return []int64{int64(q32(v[0]).ToQ16().Raw())} }},
	{name: "to48_16", in: []column{colQ16}, out: []column{colQ48, colSat},
		eval: func(v []int64) []int64 { return []int64{q16(v[0]).ToQ48().Raw()} }},
	{name: "to48_32", in: []column{colQ32}, out: []column{colQ48, colSat},
		eval: func(v []int64) []int64 { return []int64{q32(v[0]).ToQ48().Raw()} }},
	{name: "to32_48", in: []column{colQ48}, out: []column{colQ32, colSat},
		eval: func(v []int64) []int64 { return []int64{q48(v[0]).ToQ32().Raw()} }},
}

func main() {
	dir := flag.String("o", "traces", "directory to write the trace files into")
	flag.Parse()

	if !fixed.SaturationCountingEnabled {
		fmt.Fprintln(os.Stderr, "fixedtrace: the sat column needs the saturation counter")
		os.Exit(1)
	}

	if err := os.MkdirAll(*dir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "fixedtrace:", err)
		os.Exit(1)
	}

	for _, op := range ops {
		path := filepath.Join(*dir, op.name+".txt")
		lines, err := write(path, op)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fixedtrace:", err)
			os.Exit(1)
		}
		fmt.Printf("%s %d\n", path, lines)
	}
}

func write(path string, op traceOp) (int, error) {
	file, err := os.Create(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	out := bufio.NewWriter(file)
	fmt.Fprintf(out, "# fixed trace %s\n", op.name)
	fmt.Fprintf(out, "# in %s\n", join(op.in))
	fmt.Fprintf(out, "# out %s\n", join(op.out))

	lines := 0
	emit := func(in []int64) {
		if op.skip != nil && op.skip(in) {
			return
		}

		fixed.ResetSaturationCount()
		results := op.eval(in)
		saturations := int64(fixed.SaturationCount())

		fields := make([]string, 0, len(op.in)+len(op.out))
		for i, c := range op.in {
			fields = append(fields, formatValue(in[i], c))
		}
		for i, c := range op.out {
			if c == colSat {
				fields = append(fields, formatValue(saturations, c))
				continue
			}
			fields = append(fields, formatValue(results[i], c))
		}

		fmt.Fprintln(out, strings.Join(fields, " "))
		lines++
	}

	enumerate(op.in, emit)
	sample(op.in, emit)

	return lines, out.Flush()
}

// The edge cross product finds the boundaries. These cases find the rest: the
// carry chains and the middle of the range that no edge touches.
const samples = 65535

// sample visits pseudo-random cases in four shapes, so the same sweep covers
// wide operands, operands too small to overflow, and each operand paired with
// an edge.
func sample(in []column, visit func(v []int64)) {
	next := lcg()
	values := make([]int64, len(in))

	for i := range samples {
		shape := i % 4
		for j, c := range in {
			values[j] = randomRaw(next(), c, shape == 1)
		}

		switch shape {
		case 2:
			last := len(in) - 1
			e := edges(in[last])
			values[last] = e[i%len(e)]
		case 3:
			e := edges(in[0])
			values[0] = e[i/4%len(e)]
		}

		visit(values)
	}
}

func lcg() func() uint64 {
	state := uint64(0x4D59_5DF4_D0F3_3173)
	return func() uint64 {
		state = state*6364136223846793005 + 1
		return state
	}
}

// randomRaw cuts the value to the width of the column. A small value cannot
// overflow, so it exercises the carry alone.
func randomRaw(bits uint64, c column, small bool) int64 {
	if c == colQ16 || c == colMask {
		v := int64(int32(bits))
		if small {
			v >>= 10
		}
		return v
	}

	v := int64(bits)
	if small {
		v >>= 20
	}
	return v
}

// enumerate visits the cross product of the edge vectors, last column fastest.
func enumerate(in []column, visit func(v []int64)) {
	values := make([]int64, len(in))

	var step func(i int)
	step = func(i int) {
		if i == len(in) {
			visit(values)
			return
		}
		for _, v := range edges(in[i]) {
			values[i] = v
			step(i + 1)
		}
	}

	step(0)
}

func edges(c column) []int64 {
	switch c {
	case colQ16, colMask:
		return edges16
	case colQ32:
		return edges32
	case colQ48:
		return edges48
	}
	panic("fixedtrace: no edge vector for column " + string(c))
}

func join(cols []column) string {
	names := make([]string, len(cols))
	for i, c := range cols {
		names[i] = string(c)
	}
	return strings.Join(names, " ")
}

// formatValue renders one raw value as the text that goes in the file.
//
// A negative value must appear as two's complement at the width of its column,
// never with a minus sign. The reader in the twin then copies the digits into a
// buffer without knowing anything about sign. The widths are 8 hex digits for
// q16 and mask, 16 for q32 and q48. The sat column is a decimal count.
func formatValue(v int64, c column) string {
	switch c {
	case colSat:
		return fmt.Sprintf("%d", v)
	case colQ16, colMask:
		return fmt.Sprintf("%08x", uint32(v))
	case colQ32, colQ48:
		return fmt.Sprintf("%016x", uint64(v))
	}
	panic("fixedtrace: no format for column " + string(c))
}
