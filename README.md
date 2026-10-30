# Koopman analysis of financial factor dynamics

This project studies whether Koopman-operator approximations can describe and
forecast dynamics in daily Fama–French factor returns, especially the nonlinear
dependence visible in squared returns and volatility.

It is organized as a research pipeline:

1. validate the historical factor panel
2. diagnose tails, serial dependence, volatility clustering, and cross-factor
   structure
3. establish rolling linear baselines
4. derive and verify Dynamic Mode Decomposition (DMD)
5. extend DMD to polynomial observables with EDMD
6. compare volatility forecasts against HAR, EWMA, and GARCH

We do not assume that a Koopman model will beat GARCH.

## Research questions

The notebooks address four related questions:

- Are daily factor returns approximately linearly predictable at a one-day
  horizon?
- Does a finite-dimensional DMD operator reveal meaningful spectral structure?
- Do nonlinear polynomial observables improve signed-return or squared-return
  forecasts relative to ridge VAR?
- Can volatility persistence be represented by a linear operator acting on
  nonlinear observables such as squared returns and multiscale features?

## Mathematical background

### Nonlinear dynamics and observables

Let a state evolve according to

$$
x_{t+1}=F(x_t), \qquad x_t\in\mathbb{R}^d.
$$

For an observable $g(x)$, the Koopman operator advances the observable:

$$
(\mathcal{K}g)(x)=g(F(x)).
$$

The state map $F$ may be nonlinear, while $\mathcal{K}$ is linear in the
observable $g$. The tradeoff is that $\mathcal{K}$ is generally
infinite-dimensional. DMD and EDMD construct finite-dimensional approximations
from data.

The observed state in this project is the six-factor return vector

$$
r_t =
\begin{bmatrix}
r_t^{MKT} & r_t^{SMB} & r_t^{HML} &
r_t^{RMW} & r_t^{CMA} & r_t^{MOM}
\end{bmatrix}^{\top}.
$$

The signed observable $r_t$ is different from the nonlinear observable

$$
q_t=r_t\odot r_t,
$$

whose coordinates are squared returns. Signed returns can have weak
autocorrelation while squared returns remain persistent. This motivates
studying Koopman dynamics on nonlinear observables.

### Koopman eigenfunctions and spectrum

An observable $\phi$ is a Koopman eigenfunction if

$$
\mathcal{K}\phi=\lambda\phi.
$$

Along a trajectory,

$$
\phi(x_t)=\lambda^t\phi(x_0).
$$

For $\lambda=re^{i\theta}$:

- $r<1$ represents decay
- $r>1$ represents growth
- $\theta$ is an oscillatory phase increment
- for $0<r<1$, an approximate half-life is
  $\log(\frac12)/\log(r)$;
- for $\theta\neq0$, an approximate period is $2\pi/|\theta|$ steps.

In noisy financial data these are properties of a finite-sample operator. The notebooks therefore inspect rank,
centering, window length, and forecast performance before interpreting a
spectrum.

### Dynamic Mode Decomposition
DMD is a finite-dimensional approximation of the Koopman Operator. It takes snapshot data from a system over time and attempts to find a linear matrix that best steps the data forward in time. However, standard DMD uses the linear state itself as its observables. If the underlying system is highly non-linear, standard DMD will struggle to accurately approximate the Koopman operator because it hasn't properly lifted the data into a richer functional space.


With consecutive column snapshots

$$
X=[x_1,\ldots,x_m],\qquad
Y=[x_2,\ldots,x_{m+1}],
$$

DMD solves

$$
A_*=\arg\min_A\|Y-AX\|_F^2=YX^\dagger.
$$

Using a truncated SVD,

$$
X\approx U_r\Sigma_rV_r^\top,
$$

the full and reduced DMD operators are

$$
A_r=YV_r\Sigma_r^{-1}U_r^\top,
\qquad
\widetilde A_r=U_r^\top YV_r\Sigma_r^{-1}.
$$

Notebook 04 verifies that full-rank DMD agrees with the corresponding
no-intercept VAR. It also verifies centered DMD:

$$
A_c=(Y-\bar y\mathbf{1}^{\top})
(X-\bar x\mathbf{1}^{\top})^\dagger,
\qquad
c=\bar y-A_c\bar x,
$$

which is equivalent to an intercept VAR at full rank.

### Extended Dynamic Mode Decomposition
Instead of just feeding the raw state data into the DMD algorithm, EDMD first transforms the data using a dictionary of non-linear observables (e.g., polynomials, trigonometric functions). By lifting the data into this higher-dimensional space before applying DMD, EDMD provides a much more accurate finite approximation of the true infinite-dimensional Koopman Operator.

EDMD chooses a finite dictionary of observables

$$
\psi(x)=
\begin{bmatrix}
\psi_1(x)&\cdots&\psi_p(x)
\end{bmatrix}^{\top}
$$

and fits a linear transition in lifted coordinates. With row-oriented snapshots,
the ridge problem used here is

$$
\widehat K_\tau
=\arg\min_K
\frac{1}{n}\|Y_c-X_cK\|_F^2+\tau\|K\|_F^2.
$$

The centered fit has an unpenalized intercept:

$$
\widehat\psi_{t+1}
=\bar Y+(\psi_t-\bar X)\widehat K_\tau.
$$

For six standardized factors $z_t$, the current dictionaries are

$$
\psi_{\mathrm{linear}}(z)=z,
$$

$$
\psi_{\mathrm{squares}}(z)=(z_1,\ldots,z_6,z_1^2,\ldots,z_6^2),
$$

and

$$
\psi_{\mathrm{full}}(z)
=
\left(
z_1,\ldots,z_6,\,
z_1^2,\ldots,z_6^2,\,
z_i z_j \;:\; 1\leq i<j\leq 6
\right).
$$

The three dictionaries have dimensions \(6\), \(12\), and \(27\):

$$
6+6+\binom{6}{2}
=
6+\frac{6(6+1)}{2}
=
27.
$$

For squared-return forecasts, the inverse transformation is

$$
\widehat{r^2}
=\mu^2+2\mu\sigma\widehat z+\sigma^2\widehat{z^2}.
$$

The unconstrained polynomial EDMD regression can produce negative
second-moment forecasts.

## Data and statistical setup

The processed panel contains 15,897 observations from 1963-07-01 through
2026-08-31:

| Column | Meaning |
|---|---|
| MKT | Market excess return |
| SMB | Small-minus-big return |
| HML | High-minus-low return |
| RMW | Robust-minus-weak profitability return |
| CMA | Conservative-minus-aggressive investment return |
| MOM | Momentum return |

The source archives report percentage returns, while the processed panel stores
decimal returns. The five-factor and momentum archives are joined on date, and
the risk-free rate is excluded from the six-factor modeling panel.

Notebook 02 studies:

- skewness, kurtosis, extreme observations, and empirical Gaussian tails
- autocorrelation of returns at several lags
- autocorrelation and Ljung–Box evidence for squared-return dependence
- same-day cross-factor correlations and their eigenvalue spectrum
- lag-one stability across broad historical subperiods

## Forecasting

At origin $t$, a model with window length $W$ sees only

$$
\mathcal{I}_t=\{r_{t-W+1},\ldots,r_t\}
$$

and predicts the next observation. Standardization, feature construction,
operator fitting, and volatility-state updates occur inside that window.

The current configuration uses:

| Component | Value |
|---|---|
| Fixed loss-scale estimation | through 2004-12-31 |
| Development selection | 2005-01-01 through 2014-12-31 |
| Frozen holdout | 2015-01-01 onward |
| Rolling window | 756 observations |
| Ridge grid | 12 log-spaced values from $10^{-2}$ to $10^3$ |
| Selection block length | 20 trading observations |
| Bootstrap blocks | 5, 20, and 60 observations |

For actual value $y_{t,j}$, forecast $\widehat y_{t,j}$, and fixed factor scale
$s_j^2$, the pooled daily loss is

$$
L_t=\frac{1}{6}\sum_{j=1}^6
\frac{(y_{t,j}-\widehat y_{t,j})^2}{s_j^2}.
$$

Pooled out-of-sample $R^2$ is

$$
R^2_{OOS}=1-
\frac{\sum_t L_t^{\mathrm{model}}}
{\sum_t L_t^{\mathrm{benchmark}}}.
$$

Loss comparisons use paired circular date-block bootstrap intervals. The same
resampled dates are applied to competing models, preserving dependence in
forecast-loss differences.

## Results and interpretation

### The data support a volatility study more strongly than a return-prediction study

Signed-return lag-one autocorrelation ranges from 0.015 for MKT to 0.157 for
MOM. Squared-return autocorrelation is much larger: it ranges from 0.201 to
0.312 at lag one and remains positive at lag 20, from 0.068 to 0.241. This is
the central empirical motivation for the project: the predictable structure is
more visible in volatility-related observables than in the return levels.

The cross-factor correlation matrix also has a nontrivial but not dominant
common component. Its first three eigenvalues explain approximately 30.2%,
23.1%, and 17.8% of standardized cross-sectional variation, respectively.

### DMD gives a useful baseline

The pre-2005 six-dimensional DMD operator has full numerical rank and a spectral
radius of approximately 0.298. Its complex-conjugate pairs imply characteristic
periods near 11.4 and 27.6 trading days, but their moduli are only about 0.207
and 0.242. Their implied half-lives are less than one trading day.

The appropriate interpretation is therefore rapid decay in the fitted linear
return dynamics. The centered DMD operator agrees with the intercept VAR, confirming that DMD is a spectral
representation of the linear baseline.

### Signed-return forecasts show only marginal gains

On the 2005–2014 development period, the rolling mean has normalized loss
1.6905. The best ridge VAR setting has loss 1.6876, corresponding to an OOS
$R^2$ of only 0.17% versus the mean. The one-SE choice is more regularized,
$\tau=15.20$, and has a similarly small improvement of roughly 0.10%.

| Model | Development loss | OOS $R^2$ vs mean | Selected setting |
|---|---:|---:|---:|
| Rolling mean | 1.6905 | 0.00% | — |
| Ridge VAR | 1.6876 | 0.17% | $\tau=5.34$ |
| Squares EDMD | 1.6881 | 0.14% | $\tau=5.34$ |
| Full EDMD | 1.6897 | 0.05% | $\tau=43.29$ |

The polynomial dictionaries therefore do not demonstrate a meaningful
development-period advantage over the linear ridge model. We conclude that adding nonlinear observables alone does not make daily factor
returns predictably nonlinear. The best signed-return EDMD forecasts are also
not automatically valid volatility forecasts. At their selected settings,
roughly 29% of Squares EDMD and 28% of Full EDMD factor-date forecasts of
squared returns are negative.

### Volatility forecasts are more successful

For squared-return forecasts, the development period selects EWMA with decay
$\lambda=0.90$ as the strongest simple model. GARCH is close behind. The
relative ranking is broadly preserved on the frozen 2015+ holdout, although
all models become less effective during the more difficult period.

| Model | Development loss | Development $R^2$ | Holdout loss | Holdout $R^2$ |
|---|---:|---:|---:|---:|
| Historical mean | 3.5309 | 0.0% | 4.9783 | 0.0% |
| Squared VAR | 3.1048 | 12.1% | 4.8003 | 3.6% |
| Symmetric HAR | 2.7568 | 21.9% | 4.5603 | 8.4% |
| Asymmetric HAR | 2.7681 | 21.6% | 4.5705 | 8.2% |
| EWMA | 2.6480 | 25.0% | 4.5110 | 9.4% |
| GARCH(1,1) | 2.6802 | 24.1% | 4.5221 | 9.2% |

These results support three conclusions:

1. Volatility clustering is real and forecastable in this panel.
2. Simple adaptive volatility filters are competitive with a parametric GARCH
   benchmark.
3. The negative-shock extension of HAR does not improve on symmetric HAR for
   this specification and sample.

The volatility results are stronger than the signed-return results, but they do
not yet establish a Koopman advantage. The current best performers are EWMA and
GARCH, while the existing EDMD volatility forecasts are slightly weaker than a
direct squared-return VAR. The honest conclusion is that the project validates
the operator framework and reveals where nonlinear dynamics live, but does not
yet show an advantage of polynomial EDMD.

## Current conclusion

> Daily factor returns have little stable linear predictability, while their
> volatility has substantial persistence. DMD provides a transparent spectral
> view of the short-memory return dynamics, and EDMD provides a way
> to test nonlinear observables. In the current sample, however, polynomial
> EDMD does not outperform regularized linear or classical volatility models.


## Notebook sequence

Run the notebooks in this order.

### 01 — Data acquisition and validation

notebooks/01_data_exploration.py

- verifies or downloads the official raw archives
- checks SHA-256 checksums before parsing
- joins the factor files and converts percentage returns to decimals
- validates dates, gaps, missing values, bounds, and column order
- writes data/processed/factors_daily.csv and a data summary

### 02 — Statistical diagnostics

notebooks/02_statistical_analysis.py

Diagnoses the distributions, dependence, volatility clustering, factor
correlations, PCA spectrum, and stability needed to motivate the model
observables and evaluation design.

### 03 — Linear signed-return baselines

notebooks/03_linear_models.py

Compares zero, rolling mean, intercept OLS/VAR(1), and ridge VAR(1). It performs
development-period penalty selection, reports factor-level MSE and $R^2$, applies
a paired one-SE rule, estimates a minimum detectable improvement, and saves
ridge tuning artifacts.

### 04 — Dynamic Mode Decomposition

notebooks/04_dmd.py

Derives and verifies the SVD solution, checks DMD against the pseudoinverse and
VAR, visualizes the fitted spectrum, studies rank sensitivity, and verifies
centered DMD/intercept VAR equivalence.

### 05 — Polynomial EDMD

notebooks/05_edmd.py

Compares ridge VAR with squares-only and full quadratic EDMD dictionaries for
signed-return and squared-return forecasts. It validates dictionary dimensions,
keeps the rolling information set explicit, studies window-length spectral
sensitivity, and reports negative second-moment forecasts as a diagnostic.

### 06 — Volatility forecasting

notebooks/06_volatility_forecasting.py

Forecasts next-day squared returns using historical mean, squared-return ridge
VAR, symmetric and asymmetric HAR, EWMA, and GARCH(1,1). It separates
development selection from the frozen 2015+ holdout, reports GARCH status and
negative-forecast diagnostics, computes paired bootstrap intervals, and includes
a multiscale Koopman diagnostic.

## Repository structure

~~~text
koopman-financial-dynamics/
├── config.yaml
├── data/processed/
├── notebooks/
├── scripts/fetch_data.py
├── src/
│   ├── data.py
│   ├── evaluation.py
│   └── models.py
└── tests/
~~~

- src/data.py owns acquisition, checksum verification, parsing, joining, and
  panel validation.
- src/models.py owns baselines, DMD, EDMD, HAR, EWMA, and GARCH.
- src/evaluation.py owns rolling forecasts, losses, OOS scores, configuration,
  and paired bootstrap inference.

## Setup

From the repository root:

~~~sh
uv sync --locked --no-install-project
~~~

Open a notebook interactively with:

~~~sh
./.venv/bin/python -m marimo edit notebooks/01_data_exploration.py
~~~

Replace the filename to open another notebook. The notebooks are designed to be
run from the repository root.

## Rebuilding the processed data

~~~sh
./.venv/bin/python -m scripts.fetch_data
~~~

If a checksum manifest exists, both required archives must be present and match
it. If neither archive is present, the script downloads both official archives,
verifies SHA-256, and then processes them. Partial snapshots and checksum
mismatches stop execution without replacing existing files.

Live downloads may differ from the checked-in processed snapshot. For
reproducible analysis, retain the processed CSV together with the matching raw
archives and checksum manifest.

## Verification

~~~sh
./.venv/bin/python -m pytest -q
~~~

The tests cover data parsing and checksums, rolling forecast boundaries, DMD/VAR
equivalence, EDMD dictionary dimensions, volatility feature timing, bootstrap
determinism, and notebook-level holdout safeguards.

## Research limitations

This is a finite-sample empirical study, not a proof that financial markets
possess a low-dimensional Koopman representation. The learned operator depends
on the observable dictionary, rolling window, regularization, and sampling
frequency. Daily factor returns also contain substantial noise and
nonstationarity.

Therefore:

- eigenvalues are diagnostics of the fitted approximation
- development gains are not treated as holdout evidence
- the holdout is evaluated only after settings are frozen
- negative or unstable second-moment forecasts are reported, not hidden
- a model is not considered successful merely because its dictionary is more
  sophisticated

## References

- Koopman, B. O. (1931), Hamiltonian Systems and Transformation in Hilbert
  Space.
- Tu, J. H. et al. (2014), [On Dynamic Mode Decomposition: Theory and
  Applications](https://doi.org/10.3934/jcd.2014.1.391).
- Williams, M. O., Kevrekidis, I., and Rowley, C. W. (2015), [A Data-Driven
  Approximation of the Koopman Operator: Extending Dynamic Mode
  Decomposition](https://doi.org/10.1007/s00332-015-9258-5).
- Arbabi, H. and Mezić, I. (2017), [Ergodic Theory, Dynamic Mode Decomposition,
  and Computation of Spectral Properties of the Koopman
  Operator](https://doi.org/10.1137/17M1125236).
