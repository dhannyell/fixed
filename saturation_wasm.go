//go:build (js || wasip1) && !fixed_nosatcounter

package fixed

// SaturationCountingEnabled reports whether this build records saturation
// events. Build with -tags=fixed_nosatcounter to disable diagnostic counting.
// Arithmetic still saturates and produces the same result bits in either mode.
const SaturationCountingEnabled = true

// plainSaturationCounter is a non-atomic counter for single-threaded targets.
// js/wasm and wasip1 run goroutines on one thread without asynchronous
// preemption, so an increment cannot be interrupted. An atomic add is not a
// compiler intrinsic on these targets; its call cost pushed the scalar
// methods past the inlining budget.
type plainSaturationCounter struct{ n uint64 }

func (c *plainSaturationCounter) Add(n uint64) { c.n += n }

// saturationEvents records diagnostic data. It does not affect fixed-point
// values or operation results.
var saturationEvents plainSaturationCounter

// SaturationCount reports the number of saturation events since the last reset.
func SaturationCount() uint64 { return saturationEvents.n }

// ResetSaturationCount zeroes the saturation counter.
func ResetSaturationCount() { saturationEvents.n = 0 }
