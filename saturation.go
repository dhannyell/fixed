//go:build !fixed_nosatcounter && !js && !wasip1

package fixed

import "sync/atomic"

// SaturationCountingEnabled reports whether this build records saturation
// events. Build with -tags=fixed_nosatcounter to disable diagnostic counting.
// Arithmetic still saturates and produces the same result bits in either mode.
const SaturationCountingEnabled = true

// saturationEvents records diagnostic data. It does not affect fixed-point
// values or operation results.
var saturationEvents atomic.Uint64

// SaturationCount reports the number of saturation events since the last reset.
func SaturationCount() uint64 { return saturationEvents.Load() }

// ResetSaturationCount zeroes the saturation counter.
func ResetSaturationCount() { saturationEvents.Store(0) }
