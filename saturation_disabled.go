//go:build !fixed_satcounter

package fixed

// SaturationCountingEnabled reports whether this build records saturation
// events. Build with -tags=fixed_satcounter to enable diagnostic counting.
// Arithmetic still saturates and produces the same result bits in either mode.
const SaturationCountingEnabled = false

// The compiler inlines and removes Add at the existing scalar call sites.
type disabledSaturationCounter struct{}

func (disabledSaturationCounter) Add(uint64) {}

var saturationEvents disabledSaturationCounter

// SaturationCount returns zero unless the build enables counting.
func SaturationCount() uint64 { return 0 }

// ResetSaturationCount has no effect unless the build enables counting.
func ResetSaturationCount() {}
