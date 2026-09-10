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

// Shift16 holds LaneWidth shift amounts, one for each lane of a Lane16.
//
// The value is opaque: build it with SplatShift16 or LoadShift16.
type Shift16 struct{ v shift16Data }

// maxShiftCount is the largest amount a 32-bit lane can distinguish. Every
// larger amount gives the same result, so the constructors store this one.
const maxShiftCount = 31

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

// Neg returns -a lane-wise, with Q16 saturation. It subtracts from zero
// rather than multiplying by -1, which costs a widening product on the SIMD
// paths. The two agree in every lane, saturation included.
func (a Lane16) Neg() Lane16 { return subLane16(SplatLane16(Q16Zero()), a) }

// Mul returns a*b lane-wise, rounded down to Q16.16 and saturated.
func (a Lane16) Mul(b Lane16) Lane16 { return mulLane16(a, b) }

// MulRound returns a*b lane-wise, rounded to the nearest Q16.16 step with
// exact ties toward positive infinity, and saturated.
func (a Lane16) MulRound(b Lane16) Lane16 { return mulRoundLane16(a, b) }

// SplatShift16 returns a Shift16 with every lane set to n. An amount above 31
// is stored as 31; both shift a lane to zero, or to -1 raw when it is negative.
func SplatShift16(n uint8) Shift16 { return splatShift16(min(n, maxShiftCount)) }

// LoadShift16 loads LaneWidth shift amounts from p, with the SplatShift16 rule
// for an amount above 31. It widens every amount, so it is not a single load.
func LoadShift16(p *[LaneWidth]uint8) Shift16 {
	var n [LaneWidth]uint8
	for i := range LaneWidth {
		n[i] = min(p[i], maxShiftCount)
	}
	return loadShift16(&n)
}

// ScaleDown returns a divided by two raised to the matching lane of s. It
// rounds toward negative infinity, which is the Mul rule, and it cannot
// overflow or saturate.
//
// For an amount up to 16 the result is the same bits as Mul by the Q16 value
// of two raised to minus that amount. Past 16 that value is not on the Q16
// grid and the two disagree: Mul gives zero, ScaleDown keeps shifting.
func (a Lane16) ScaleDown(s Shift16) Lane16 { return scaleDownLane16(a, s) }

// ScaleDownRound returns a divided by two raised to the matching lane of s,
// rounded to the nearest step with exact ties toward positive infinity. It
// cannot overflow or saturate.
//
// For an amount up to 16 the result is the same bits as MulRound by the Q16
// value of two raised to minus that amount. ScaleDown is the truncating form
// and costs one instruction, so prefer it when the rounding does not matter.
func (a Lane16) ScaleDownRound(s Shift16) Lane16 { return scaleDownRoundLane16(a, s) }

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

// MulAdd16Round rounds each Q16 product to the nearest Q48.16 step with exact
// ties toward positive infinity before the saturating lane-wise add.
func (a Lane48) MulAdd16Round(b, c Lane16) Lane48 { return mulAdd16RoundLane48(a, b, c) }

// ToLane16 narrows every lane with Q48.ToQ16 saturation semantics.
func (a Lane48) ToLane16() Lane16 { return lane48ToLane16(a) }
