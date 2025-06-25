# Corrected Explanation: Vertical Structures in Morse Sets

## The Reset Map: y_new = -y³ + 2y

I apologize for the earlier confusion. The reset map is NOT a simple amplification by 1.75, but rather a cubic polynomial.

### Key Properties of the Reset Map

1. **Fixed Points**: y = 0, ±1
2. **Critical Points**: y ≈ ±0.816 (where dy_new/dy = 0)
3. **Maximum values**: At y ≈ ±0.816, the map produces y_new ≈ ±1.089

### Why Vertical Structures Form

The vertical structures on the right boundary occur because:

1. **Boundary Overflow**:
   - For |y| > 0.816, the reset map produces |y_new| > 1
   - Example: y = 0.99 → y_new = 1.010 (outside domain!)
   - These values are clipped back to ±1

2. **Marginal Stability at Boundaries**:
   - The fixed points at y = ±1 have derivative = -1
   - This creates oscillatory behavior near the boundaries
   - Small perturbations lead to alternating behavior

3. **Trapping Mechanism**:
   - Boxes near y = ±1 on the right boundary experience:
     - Jump to left with y slightly outside domain
     - Clipping keeps them at y = ±1
     - Flow back to right boundary
     - Repeat cycle
   - This creates vertical strips of recurrent behavior

### Why You Expected Horizontal Strips

Your intuition was based on:
- The ODE: dx/dt = 1, dy/dt = 0 (pure horizontal flow)
- This suggests horizontal invariant manifolds

However, the cubic reset map creates complex behavior:
- Near y = 0: The map is expansive (derivative = 2)
- Near y = ±1: The map overshoots the domain
- This breaks the simple horizontal flow pattern

### The Morse Sets

The three Morse sets represent:
1. **Bottom strip** (y ≈ -1): Trapped by boundary effects
2. **Top strip** (y ≈ +1): Trapped by boundary effects  
3. **Middle region**: Complex mixing behavior

The vertical structures are genuine recurrent sets created by the interplay of:
- Horizontal continuous flow
- Cubic reset map that overshoots at boundaries
- Domain clipping that creates trapping regions

This is a beautiful example of how nonlinear reset maps in hybrid systems can create unexpected recurrent structures!