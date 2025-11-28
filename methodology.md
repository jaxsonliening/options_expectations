Mathematical Methodology

This document outlines the computational framework used to derive "real-world" probabilities from option prices. The process moves from noisy market data to a clean probability density function (PDF) through three distinct stages.

Stage 1: Volatility Smoothing (Shimko's Method)

Raw option prices are notoriously noisy due to wide bid-ask spreads and liquidity gaps. Differentiating raw prices directly (the standard approach) leads to unstable "spikes" in probability.

Instead, we utilize the Shimko (1993) technique, which fits the Implied Volatility (IV) smile rather than the prices themselves.

Filter: We discard illiquid options (volume=0) and data artifacts (IV < 1% or > 200%).

Fit: We fit a quadratic polynomial to the valid strike-IV pairs:


$$\sigma(K) = aK^2 + bK + c$$


This enforces a smooth "smile" shape, ensuring the resulting probability distribution is continuous and differentiable.

Reconstruction: We generate a dense grid of strikes ($K_{grid}$) and calculate smooth IVs ($\sigma_{smooth}$) using the polynomial.

Inversion: We feed $\sigma_{smooth}$ back into the Black-Scholes equation to derive "Clean Call Prices" ($C_{smooth}$).

Stage 2: Risk-Neutral Extraction ($\mathbb{Q}$-Measure)

We utilize the Breeden-Litzenberger (1978) result, which states that the Risk-Neutral PDF ($f_\mathbb{Q}$) is proportional to the second derivative of the Call price with respect to the Strike.

$$f_\mathbb{Q}(K) = e^{rT} \frac{\partial^2 C_{smooth}}{\partial K^2}$$

Since $C_{smooth}$ is derived from a quadratic volatility function, the second derivative is stable, eliminating the negative probabilities often found in cubic spline approaches.

Stage 3: The Real-World Transform ($\mathbb{P}$-Measure)

The Risk-Neutral distribution ($\mathbb{Q}$) reflects hedging costs, not actual forecasts. It overweights crash probabilities because investors pay a premium for insurance (Puts).

To recover the Real-World distribution ($\mathbb{P}$), we apply the Bliss & Panigirtzoglou (2004) Power Utility transformation. We assume a Representative Agent with Constant Relative Risk Aversion (CRRA).

The Transformation

$$f_\mathbb{P}(S_T) = \frac{f_\mathbb{Q}(S_T) \cdot S_T^{-\gamma}}{\int f_\mathbb{Q}(x) \cdot x^{-\gamma} dx}$$

Where:

$\gamma$ (Gamma) is the Coefficient of Risk Aversion.

$S_T^{-\gamma}$ is the marginal utility function.

Solving for Gamma ($\gamma$)

Since $\gamma$ is unobservable, we solve for it implicitly by anchoring the distribution to the Capital Asset Pricing Model (CAPM).

We numerically optimize $\gamma$ such that the mean of the resulting $\mathbb{P}$-distribution equals the CAPM expected return:

$$\text{Minimize: } \left( \mathbb{E}^\mathbb{P}[S_T] - S_0 e^{(r + ERP)T} \right)^2$$

If $\gamma > 0$: The market is Risk Averse (Real-world mean > Risk-neutral mean).

If $\gamma < 0$: The market is Risk Seeking (Real-world mean < Risk-neutral mean).

Assumptions & Limitations

Geometric Brownian Motion: The Shimko method assumes the underlying process can be modeled via Black-Scholes for the purpose of smoothing, though the resulting smile allows for non-lognormal distributions (skewness/kurtosis).

Constant Risk Aversion: We assume a single $\gamma$ applies across all strike prices. In reality, risk aversion likely changes in the tails (investors are more risk-averse regarding crashes than melt-ups).

CAPM Reliance: The accuracy of the Real-World probability is entirely dependent on the accuracy of the input ERP (Equity Risk Premium). If the user inputs a wrong required return, the probability distribution will shift to match it.