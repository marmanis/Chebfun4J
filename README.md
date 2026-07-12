# chebfun4j — Chebfun for Java

`chebfun4j` is a Java port of the numerical core of the MATLAB
[Chebfun](https://github.com/chebfun/chebfun) library: compute with smooth
real-valued functions on a real interval by representing them as truncated
Chebyshev (or Fourier, for periodic problems) polynomial expansions to
near machine precision, then doing arithmetic, calculus, rootfinding, and
ODE-solving on the coefficients directly.

The library is built on [jax4j](../jax4j) — the FFT/DCT primitives, the
LU-based dense solver, and the eigenvalue solver all live there in
`com.marmanis.jax4j.api.{Fft,Linalg}`, and are reusable outside chebfun4j.

## What's in iteration 5

Everything from earlier iterations plus:

- **Systems of ODEs.** Two new types cover the coupled multi-component case:
  - `LinearBlockChebop` — DSL for `k`-component linear block operators
    built up as sums of terms `coeff * d^order u_col / dx^order`.
    Solves `L u = f` where both `u` and `f` are vectors of Chebfuns.
    Same spectral-collocation solver path as `Chebop`, scaled up to a
    block matrix.
  - `NonlinearSystem` — vector-in vector-out pointwise residual
    `F(x, u[], u'[], u''[]) -> double[]`. Newton with block Jacobian
    (analytic partials via `Residual.jacobian` override, or centered
    FD default).
- **`SystemBC`.** Sealed type mirroring `BoundaryCondition` but with a
  `component` field, so a single BC picks out which unknown in the
  system it applies to.
- **`Quasimatrix.qr()` and `Quasimatrix.svd()`.** Continuous linear
  algebra of Chebfun columns:
  - `qr(Algorithm.MODIFIED_GRAM_SCHMIDT)` — column-wise MGS.
    Fast, mildly stable.
  - `qr(Algorithm.HOUSEHOLDER)` — chebfun-style Householder
    reflectors defined by orthonormal Chebfuns. Robust on
    ill-conditioned quasimatrices.
  - `svd()` — QR-then-SVD-of-R via jax4j's new `Linalg.svd`.
    Returns U (quasimatrix), sigma (descending), V^T (small matrix).
- **`jax4j Linalg.svd`.** General real SVD of an `m × n` matrix
  (`m >= n`) via the classical Golub-Kahan-Reinsch algorithm:
  Householder bidiagonalization on both sides, then an implicit-
  shifted QR sweep on the resulting bidiagonal (with deflation and
  splitting). Numerically stable across the whole spectrum — recovers
  singular values spanning 12 orders of magnitude to relative error
  ≤ 1e-13, where the earlier normal-equations path would have lost
  the small ones to `sqrt(eps)`.

## What was in iteration 4

Everything from earlier iterations plus:

- **Nonlinear ODE solver.** `NonlinearChebop` solves scalar BVPs of the
  form `F(x, u, u', u'') = 0` on `[a, b]` with any mix of Dirichlet,
  Neumann, and Robin BCs. Damped-Newton iteration on a Chebyshev spectral
  collocation grid, with an outer adaptive loop over grid sizes. Two
  ways to specify the residual:
  - **Scalar `Residual` interface** — `(x, u, u', u'') -> double`. The
    interface's default partials are centered finite differences on the
    scalar `F`; users can override with analytic derivatives.
  - **Autodiff DSL via jax4j** — `NonlinearChebop.autodiffResidual(fn)`
    takes a `(x, u, up, upp) -> NDArray` lambda where each argument is a
    FLOAT64 NDArray (scalar for pointwise calls, vector for batched).
    The three partials come out of jax4j reverse-mode AD (`JAX.grad`) at
    machine precision, no finite-difference noise, no analytic
    derivatives required. The batched view traces the residual and
    its gradient Jaxpr <em>once per grid size</em> (four inputs, no
    baked-in constants) and reuses them via
    `Grad.forwardInterpret` / `Grad.backwardInterpret` across every
    Newton iteration and Armijo backtrack — measured 3–9× faster than
    the naive per-grid-point autodiff and within noise of FD on the
    Bratu problem. See `com.marmanis.chebfun4j.examples.NonlinearBench`.
- **`NewtonOptions`.** Configuration record for the Newton loop
  (`maxIter`, `tol`, `initialDamping`, `initialGuess`), with a sensible
  `defaults()` factory.
- **`Chebop.eigs` for Neumann/Robin BCs.** The eigenvalue path now
  handles all three BC types. Implementation uses a 2×2 substitution to
  express the two boundary DOFs in terms of the interior, reducing to a
  standard eigenvalue problem — no BC-row mass-matrix trick needed.
- **`Chebfun2` pointwise product.** `Chebfun2.times(Chebfun2)` runs the
  ACA constructor on the product function, adaptively picking the
  minimal rank that resolves it.

## What's in iteration 3

Everything from earlier iterations (single-piece Chebfun; piecewise
Chebfun with splitting-on; colleague rootfinding; Trigfun; Chebop
Dirichlet BVPs) plus:

- **Neumann and Robin BCs** for `Chebop`. A new sealed
  `BoundaryCondition` type — `Dirichlet(value)`, `Neumann(value)`,
  `Robin(alpha, beta, value)` — replaces the old (double, double)
  argument pair. `Chebop.solve(rhs, bcA, bcB)` handles any mix. The
  legacy `(alpha, beta)` overload still works and defaults to Dirichlet.
- **`Chebop.eigs` — Sturm-Liouville eigenvalue problems.** Solve
  {@code L u = λ u} with Dirichlet BCs; returns a
  `record Eigs(double[] eigenvalues, Quasimatrix eigenfunctions)` sorted
  ascending, eigenfunctions L^2-normalized. Uses `jax4j`'s new
  `Linalg.eig(A, B)` for the reduced interior generalized problem.
- **`Quasimatrix` — column vector of chebfuns.** Sharing a domain,
  supporting inner products, per-column L^2-normalization, and
  column-wise arithmetic. Sets up the natural return type for
  eigenfunctions and future SVD / QR work.
- **`Chebfun2` — 2-D smooth functions on a rectangle.** Low-rank ACA
  constructor (finds separation rank adaptively — `sin(x) cos(y)` is
  rank 1, `1/(1+x²+y²)` is rank 8, both machine-precision-accurate).
  Supports `feval`, `sum2` (double integral), `sum(axis)` (marginal
  integral returning a 1-D `Chebfun`), `partialX` / `partialY`, and
  `plus` / `minus` / `times`.
- **`Rectangle` type.** 2-D counterpart of `Domain`, distinguishing 1-D
  and 2-D domains at the API level.

## What was in iteration 2

- **Colleague-matrix rootfinding.** `Chebfun.roots()` finds every
  polynomial root regardless of multiplicity — the sign-change bracketer
  used to miss even-multiplicity roots like `x²`'s zero.
- **Piecewise Chebfun** via a shared `Fun` interface (implemented by
  both `Chebtech` and `Trigtech`), with a splitting-on adaptive
  constructor that resolves `|x|`, `sign(x)`, and general piecewise-
  smooth functions by bisecting the domain until each piece converges.
- **`Trigfun`** — periodic Fourier-series representation adaptive on
  power-of-two grid sizes.
- **`Chebop`** — Chebyshev-spectral-collocation linear ODE BVP solver
  (Dirichlet-only in iteration 2; extended in iteration 3).

## Setup

- **JDK**: Java 25 or higher (jax4j's version).
- **Maven**: standard build.
- **jax4j**: install the sibling project once so chebfun4j can consume
  `com.marmanis:jax4j:1.0-SNAPSHOT`:
  ```bash
  cd ../jax4j && mvn install -DskipTests
  ```

Then:
```bash
mvn test
mvn exec:java -Dexec.mainClass=com.marmanis.chebfun4j.examples.ChebfunDemo
mvn exec:java -Dexec.mainClass=com.marmanis.chebfun4j.examples.PiecewiseDemo
mvn exec:java -Dexec.mainClass=com.marmanis.chebfun4j.examples.BvpDemo
```

## Usage

Smooth adaptive construction, evaluation, integration (unchanged since
iteration 1):

```java
Chebfun f = new Chebfun(x -> Math.exp(Math.sin(x)), new Domain(0.0, 2 * Math.PI));
double y = f.feval(1.0);      // 2.319776824715853
double integral = f.sum();    // 7.954926521012847  (matches 2 pi I_0(1))
```

Piecewise for functions with kinks or jumps:

```java
Chebfun f = new Chebfun(Math::abs, new Domain(-1.0, 1.0));
f.numPieces();        // 2
f.breakpoints();      // [-1.0, 0.0, 1.0]
f.feval(-0.5);        // 0.5
f.sum();              // 1.0     (integral)
f.diff().feval(0.5);  // +1.0    (piecewise sign)
```

Colleague-matrix roots, including even-multiplicity:

```java
Chebfun f = new Chebfun(x -> x * x, new Domain(-1.0, 1.0));
double[] r = f.roots();   // [0.0]  — the double root at zero
```

Periodic functions with `Trigfun`:

```java
Trigfun s = new Trigfun(Math::sin, new Domain(0.0, 2 * Math.PI));
Trigfun c = new Trigfun(Math::cos, new Domain(0.0, 2 * Math.PI));
Trigfun prod = s.times(c);              // sin(2x) / 2 by identity
double energy = s.times(s).sum();       // pi   ( = ||sin||_2^2 )
```

Linear ODE boundary-value solves with `Chebop`, now with any mix of
Dirichlet, Neumann, and Robin BCs:

```java
import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.BoundaryCondition.Neumann;
import com.marmanis.chebfun4j.BoundaryCondition.Robin;

Domain d = new Domain(0.0, 1.0);
Chebop L = Chebop.constantCoefficients(d, -1.0, 0.0, 1.0);  // u'' - u

// Dirichlet at both ends (the old signature still works).
Chebfun u1 = L.solve(x -> 0.0, 1.0, 1.0 / Math.E);          // e^{-x}

// Dirichlet at 0, Neumann at 1.
Chebfun u2 = L.solve(x -> 0.0, new Dirichlet(1.0), new Neumann(-1.0 / Math.E));

// Robin at 0, Dirichlet at 1.
Chebfun u3 = L.solve(x -> 0.0, new Robin(1.0, -1.0, 2.0), new Dirichlet(Math.E));
```

Sturm-Liouville eigenvalue problems:

```java
Domain d = new Domain(0.0, Math.PI);
Chebop L = Chebop.zero(d).plus(2, -1.0);                    // L u = -u''
Chebop.Eigs eigs = L.eigs(new Dirichlet(0.0), new Dirichlet(0.0), 6);
double[] lambdas = eigs.eigenvalues();                       // {1, 4, 9, 16, 25, 36}
Quasimatrix V = eigs.eigenfunctions();                       // L^2-normalized
Chebfun v1 = V.get(0);                                       // ~ sqrt(2/pi) sin(x)
```

2-D functions on a rectangle:

```java
Rectangle unit = Rectangle.unit();
Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) * Math.cos(y), unit);
Chebfun2 g = new Chebfun2((x, y) -> Math.exp(x + y), unit);
f.rank();                     // 1  (separable ⇒ low-rank)
f.feval(0.3, 0.4);            // sin(0.3) * cos(0.4)
f.sum2();                     // integral over the whole rectangle
Chebfun marginal = f.sum(0);  // integrate over x, get a chebfun of y
Chebfun2 fx = f.partialX();
Chebfun2 prod = f.times(g);   // pointwise product, ACA-adaptive rank
```

Nonlinear ODE boundary-value problems:

```java
// Bratu: u'' + lambda e^u = 0 on [0, 1], u(0) = u(1) = 0.
Domain d = new Domain(0.0, 1.0);
double lam = 1.0;
NonlinearChebop.Residual F = (x, u, up, upp) -> upp + lam * Math.exp(u);
NonlinearChebop N = new NonlinearChebop(d, F);
Chebfun u = N.solve(new Dirichlet(0.0), new Dirichlet(0.0));

// Same problem, with a warm start:
NewtonOptions opts = NewtonOptions.withInitialGuess(previousSolution);
Chebfun refined = N.solve(new Dirichlet(0.0), new Dirichlet(0.0), opts);

// Autodiff DSL: partials computed exactly by jax4j reverse-mode AD.
NonlinearChebop.Residual Fad = NonlinearChebop.autodiffResidual(
    (x, u, up, upp) -> upp.add(u.exp().mul(scalar(1.0))));      // u'' + e^u
```

## Package layout

- `com.marmanis.chebfun4j.Chebfun` — piecewise container of `Fun` pieces.
- `com.marmanis.chebfun4j.Fun` — shared interface for a smooth
  representation on `[-1, 1]`.
- `com.marmanis.chebfun4j.Chebtech` — Chebyshev-series `Fun`
  implementation (`chebtech2`).
- `com.marmanis.chebfun4j.Trigtech` — Fourier-series `Fun` for periodic
  functions.
- `com.marmanis.chebfun4j.Trigfun` — user-facing periodic-function type.
- `com.marmanis.chebfun4j.Chebop` — linear-ODE BVP + eigenvalue solver.
- `com.marmanis.chebfun4j.NonlinearChebop` — nonlinear scalar BVP solver
  by damped Newton on Chebyshev collocation.
- `com.marmanis.chebfun4j.LinearBlockChebop` — linear block-operator DSL
  for coupled k-component systems.
- `com.marmanis.chebfun4j.NonlinearSystem` — Newton solver for coupled
  k-component nonlinear BVPs.
- `com.marmanis.chebfun4j.NewtonOptions` — Newton-loop configuration.
- `com.marmanis.chebfun4j.BoundaryCondition` — sealed `Dirichlet` /
  `Neumann` / `Robin` variants (scalar BVPs).
- `com.marmanis.chebfun4j.SystemBC` — sealed BC variants with a
  component index (system BVPs).
- `com.marmanis.chebfun4j.Quasimatrix` — column vector of chebfuns
  with continuous `qr` and `svd`.
- `com.marmanis.chebfun4j.Chebfun2` — 2-D low-rank Chebyshev
  representation with ACA constructor.
- `com.marmanis.chebfun4j.Rectangle` — 2-D domain.
- `com.marmanis.chebfun4j.Domain` — 1-D closed real interval.
- `com.marmanis.chebfun4j.util.ChebyshevPoints` — grid generation.
- `com.marmanis.chebfun4j.util.ChebTransform` — values ↔ Chebyshev
  coefficients via jax4j's DCT-I.
- `com.marmanis.chebfun4j.util.Clenshaw` — Chebyshev-series evaluation.
- `com.marmanis.chebfun4j.util.RootFinder` — colleague-matrix rootfinder
  with recursive subdivision.
- `com.marmanis.chebfun4j.util.DifferentiationMatrix` — Chebyshev
  spectral D matrix at second-kind points.
- `com.marmanis.chebfun4j.examples` — `ChebfunDemo`, `PiecewiseDemo`,
  `BvpDemo`, `EigenvalueDemo`, `Chebfun2Demo`, `NonlinearBvpDemo`,
  `NonlinearBench`, `SystemsDemo`, `QrSvdDemo`.

## What's still out of scope

- **Chebfun3 (3-D)** — trivariate low-rank representations.
- **Chebfun2 splitting-on / piecewise** — currently one smooth
  low-rank piece per rectangle.
- **Nonlinear eigenvalue problems** — the linear `Chebop.eigs`,
  `NonlinearChebop`, and `NonlinearSystem` are all there, but their
  combination (nonlinear eigenvalue via Newton around eigs) is not.
- **Quasimatrix `mrdivide`** — least-squares operations built on QR
  are natural but not yet wired.
- **Singular-endpoint representations** (`unbndfun`, `singfun`).
- **Cross-basis piecewise arithmetic.** A `Chebfun` whose pieces are all
  `Chebtech` composes fine, and a `Trigfun` composes with itself; mixing
  `Chebtech` and `Trigtech` pieces in one `Chebfun` throws pending a
  common-basis coercion.

## Testing

```bash
mvn test
```

Iteration 5 ships 124 chebfun4j unit tests (plus 224 jax4j tests)
across every earlier feature and iteration 5's LinearBlockChebop /
NonlinearSystem / Quasimatrix QR (MGS + Householder) / Quasimatrix SVD.

## A note on GPU acceleration

chebfun4j runs entirely on the host CPU by design. jax4j has FLOAT64
TornadoVM kernels available (elementwise + matmul), but on the consumer
NVIDIA GPU tested (RTX 3050 Laptop), those kernels were measured 5–60×
slower than the host CPU across every op, at array sizes 10,000× larger
than what chebfun4j actually uses. Three reasons: consumer NVIDIA cards
run FP64 at 1/32–1/64 of FP32 rate; TornadoVM's `EVERY_EXECUTION` data
transfer mode adds a PCIe-round-trip floor per op; and chebfun4j's typical
arrays (20–500 doubles) are far too small to amortize either. See the
[jax4j README](../jax4j/README.md#measured-gpu-acceleration-use-with-eyes-open)
for the full measurement table. If you're running chebfun4j on a
datacenter GPU (A100/H100) with full-rate FP64, the story likely differs
— re-benchmark with `com.marmanis.jax4j.examples.Fp64GpuVerify` first.

## License

Apache License 2.0.
