# Unstable Periodic System: Right Boundary Analysis Summary

## System Dynamics

The unstable periodic system has the following characteristics:

1. **ODE**: dx/dt = 1.0, dy/dt = 0.0 (constant rightward motion)
2. **Guard Set**: g(x,y) = 1 - x = 0 (vertical line at x = 1)
3. **Reset Map**: When x reaches 1, the system jumps to:
   - x⁺ = 0
   - y⁺ = y⁻ × 1.75 (magnifies y by factor of 1.75)

## Key Findings from Box Map Analysis

### 1. Right Boundary Behavior
- All 100 boxes touching the right boundary (x ∈ [0.99, 1.0]) have non-empty images
- Average of 4.8 image boxes per source box
- Time horizon τ = 1.42 is sufficient for trajectories to reach and potentially cross the boundary

### 2. Image Distribution
- **296 mappings to middle region** (0.2 ≤ x ≤ 0.8)
- **186 mappings to right region** (x > 0.8)
- **0 mappings to left region** (x < 0.2)

This makes sense because:
- Starting near x = 0.99, the flow moves rightward with velocity 1.0
- In time τ = 1.42, trajectories can reach x = 1.0 and jump back to x = 0
- After jumping, they continue flowing rightward from x = 0

### 3. Y-Coordinate Patterns
- **Boxes near y = 0** have the most image boxes (up to 12)
- **Boxes at extreme y values** (near ±1) have fewer image boxes (2-4)
- This is due to the y-magnification factor of 1.75 in the reset map

### 4. Self-Mapping Boxes
- 40 boxes on the right boundary map to themselves
- These are primarily at the extreme y-values where trajectories may not reach the guard in time τ

### 5. Special Cases Near y = 0
- Boxes centered around y ≈ 0 have up to 12 destination boxes
- This occurs because:
  1. Small y values remain small after reset (1.75 × small ≈ small)
  2. Multiple jumps can occur within time τ
  3. Each jump spreads the trajectory across multiple boxes due to bloating

## Physical Interpretation

The unstable periodic system represents a "sliding and resetting" dynamics where:
- Particles slide rightward at constant speed
- Upon hitting the right boundary, they teleport to the left edge
- The y-coordinate is amplified by 1.75 at each reset
- This creates expanding oscillations in the y-direction

The box map correctly captures this complex behavior, showing how initial conditions near the right boundary can map to multiple regions due to the hybrid (continuous + discrete) dynamics.