package fixed

// Lane16 holds LaneWidth Q16 values for register-resident lane-wise work.
// Its methods produce the same bits as the corresponding scalar operations.
// Mul rounds down with the BatchMul16 arithmetic-shift rule, which is also the
// Q16.Mul rule. Each saturated lane adds one event to SaturationCount when
// counting is enabled. MulAdd and MulSub execute two operations and count the
// multiplication and the addition or subtraction separately.
//
// In an amd64 SIMD build, callers must check LanesAvailable before constructing
// or using a Lane16. Calling a lane operation without AVX2 is undefined.
//
// The value is opaque: use the methods; the layout differs by path.
type Lane16 struct{ v lane16Data }

// Mask16 holds LaneWidth comparison results. Each lane is all ones when set
// and all zeros when clear.
//
// The value is opaque: use the methods; the layout differs by path.
type Mask16 struct{ v mask16Data }

// Lane48 holds LaneWidth Q48 values for register-resident accumulation. Its
// methods produce the same bits and saturation events as the corresponding
// scalar Q48 operations applied LaneWidth times.
//
// The value is opaque: use the methods; the layout differs by path.
type Lane48 struct{ v lane48Data }

// LanesAvailable reports whether lane operations may be called. It is false
// only in an amd64 SIMD build on a CPU without AVX2. Calling any lane operation
// in that case is undefined. Other build paths always return true.
func LanesAvailable() bool { return lanesAvailable() }

// LanePath reports the compiled lane implementation: "avx2", "neon", or
// "generic". Use LanesAvailable before using the amd64 SIMD implementation.
func LanePath() string { return lanePath() }

// SplatLane16 returns a Lane16 with every lane set to q.
func SplatLane16(q Q16) Lane16 { return splatLane16(q) }

// LoadLane16 loads LaneWidth Q16 values from p.
func LoadLane16(p *[LaneWidth]Q16) Lane16 { return loadLane16(p) }

// Store writes every lane of a to p.
func (a Lane16) Store(p *[LaneWidth]Q16) { storeLane16(a, p) }

// Add returns a+b lane-wise, with Q16 saturation.
func (a Lane16) Add(b Lane16) Lane16 { return addLane16(a, b) }

// Sub returns a-b lane-wise, with Q16 saturation.
func (a Lane16) Sub(b Lane16) Lane16 { return subLane16(a, b) }

// Mul returns a*b lane-wise, rounded down to Q16.16 and saturated.
func (a Lane16) Mul(b Lane16) Lane16 { return mulLane16(a, b) }

// MulAdd returns a.Add(b.Mul(c)). It never fuses the two operations.
func (a Lane16) MulAdd(b, c Lane16) Lane16 { return a.Add(b.Mul(c)) }

// MulSub returns a.Sub(b.Mul(c)). It never fuses the two operations.
func (a Lane16) MulSub(b, c Lane16) Lane16 { return a.Sub(b.Mul(c)) }

// Min returns the smaller value in each lane.
func (a Lane16) Min(b Lane16) Lane16 { return minLane16(a, b) }

// Max returns the larger value in each lane.
func (a Lane16) Max(b Lane16) Lane16 { return maxLane16(a, b) }

// SymClamp returns Max(-limit, Min(a, limit)) lane-wise. Negating a minimum
// limit saturates to Q16MaxValue and records one event for that lane.
func (a Lane16) SymClamp(limit Lane16) Lane16 { return symClampLane16(a, limit) }

// Greater reports a>b in each lane.
func (a Lane16) Greater(b Lane16) Mask16 { return greaterLane16(a, b) }

// Equals reports a==b in each lane.
func (a Lane16) Equals(b Lane16) Mask16 { return equalsLane16(a, b) }

// Or returns the lane-wise union of m and n.
func (m Mask16) Or(n Mask16) Mask16 { return orMask16(m, n) }

// AllZero reports whether every mask lane is clear.
func (m Mask16) AllZero() bool { return allZeroMask16(m) }

// BlendLane16 selects a where m is set and b where m is clear.
func BlendLane16(m Mask16, a, b Lane16) Lane16 { return blendLane16(m, a, b) }

// ToLane48 widens every lane to Q48 exactly.
func (a Lane16) ToLane48() Lane48 { return lane16ToLane48(a) }

// SplatLane48 returns a Lane48 with every lane set to q.
func SplatLane48(q Q48) Lane48 { return splatLane48(q) }

// LoadLane48 loads LaneWidth Q48 values from p.
func LoadLane48(p *[LaneWidth]Q48) Lane48 { return loadLane48(p) }

// Store writes every lane of a to p.
func (a Lane48) Store(p *[LaneWidth]Q48) { storeLane48(a, p) }

// Add returns a+b lane-wise, with Q48 saturation.
func (a Lane48) Add(b Lane48) Lane48 { return addLane48(a, b) }

// Sub returns a-b lane-wise, with Q48 saturation.
func (a Lane48) Sub(b Lane48) Lane48 { return subLane48(a, b) }

// MulAdd16 adds the exact b*c Q16 product to each Q48 accumulator lane. Each
// product is rounded down to the shared Q48.16 grid before the saturating add.
func (a Lane48) MulAdd16(b, c Lane16) Lane48 { return mulAdd16Lane48(a, b, c) }

// ToLane16 narrows every lane with Q48.ToQ16 saturation semantics.
func (a Lane48) ToLane16() Lane16 { return lane48ToLane16(a) }
