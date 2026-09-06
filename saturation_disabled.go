//go:build fixed_nosatcounter

package fixed

// SaturationCountingEnabled reports whether this build records saturation
// events. Build with -tags=fixed_nosatcounter to disable diagnostic counting.
// Arithmetic still saturates and produces the same result bits in either mode.
const SaturationCountingEnabled = false

// The compiler inlines and removes Add at the existing scalar call sites.
type disabledSaturationCounter struct{}

func (disabledSaturationCounter) Add(uint64) {}

var saturationEvents disabledSaturationCounter

// SaturationCount returns zero when built with fixed_nosatcounter.
func SaturationCount() uint64 { return 0 }

// ResetSaturationCount has no effect when built with fixed_nosatcounter.
func ResetSaturationCount() {}
