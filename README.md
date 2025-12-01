Market Expectations: Options-Implied Scenario Analysis

A quantitative tool for equity researchers to validate Price Targets and assign empirical probabilities to Bear/Base/Bull scenarios.

Traditional valuation often uses arbitrary weightings (e.g., 25% Bear / 50% Base / 25% Bull). This tool replaces those guesses with Real-World Probability Measures ($\mathbb{P}$) derived from the options market.

Core Methodology

Breeden-Litzenberger Framework: Extracts the Risk-Neutral Density (Q-Measure) from option prices by calculating the second derivative of the call price with respect to the strike price ($f_\mathbb{Q} \propto \frac{\partial^2 C}{\partial K^2}$).

Real-World Transform ($\mathbb{P}$-Measure): Converts the Risk-Neutral distribution (hedging costs) to the Real-World distribution (market forecasts) using a Power Utility transformation. It solves for the Risk Aversion coefficient ($\gamma$) that aligns the distribution's mean with the CAPM expected return.

Scenario Bucketing: Integrates the probability density function over specific price ranges to determine the exact likelihood of your fundamental price targets.

Prerequisites

Python 3.8+

Dependencies:

pip install numpy pandas yfinance scipy matplotlib seaborn



Usage

Run the script from the command line. The ticker argument is required.

1. Scenario Expected Value (Recommended)

This is the primary function for fundamental analysis. Input your Bear, Base, and Bull price targets. The tool calculates the market-implied probability of each bucket and the resulting Expected Value.

Syntax:
python market_expectations.py [TICKER] --ev [BEAR] [BASE] [BULL]

Example:
Bear case $35, Base $60, Bull $80.

python market_expectations.py GTLB --months 12 --ev 35 60 80



2. Visualization (Plotting)

To see the probability distribution curve (and check for data artifacts), add the --plot flag. This overlays the Risk-Neutral ($\mathbb{Q}$) and Real-World ($\mathbb{P}$) distributions.

python market_expectations.py GTLB --months 12 --plot



3. Advanced Filtering (The --bounds Argument)

If you see spikes at the edges of the plot (artifacts), use --bounds to constrain the analysis range as a multiple of the Spot Price.

Default: 0.6 2.5 (Analyzes strikes between 60% and 250% of Spot).

For Volatile/Distressed Stocks: Widen the net to capture downside risk.

# Analyzes strikes from 40% of spot to 250% of spot
python market_expectations.py GTLB --ev 35 60 80 --bounds 0.4 2.5



4. Single Target Probability

Check the specific probability of the stock exceeding a single price target.

American (One-Touch) vs. European (Terminal):

Default Behavior: Calculates American (One-Touch) probability. This estimates the chance of the stock touching the target price at any point before expiration (using the Reflection Principle approximation: $P_{touch} \approx 2 \times P_{terminal}$). This is ideal for comparing against prediction markets like Polymarket.

--european Flag: Calculates European (Terminal) probability. This is the chance the stock closes above the target at expiration.

Example 1 (Default - One Touch): "What are the odds GOOGL hits $355?"

python market_expectations.py GOOGL --months 12 --target 355


Example 2 (Strict Terminal): "What are the odds GOOGL finishes the year above $355?"

python market_expectations.py GOOGL --months 12 --target 355 --european


5. Specific Date Targeting

Instead of a generic month horizon, you can target a specific expiration date using the --date argument.

Syntax:
python market_expectations.py [TICKER] --date MM/DD/YYYY --target [PRICE]

Example:
Check the probability of NVDA hitting $150 by Dec 18, 2026.

python market_expectations.py NVDA --date 12/18/2026 --target 150


Configuration Arguments

--months: Investment horizon in months (default: 12).

--date: Specific expiration date (MM/DD/YYYY). Overrides --months.

--r: Risk-free rate (default: 0.042 i.e., 4.2%).

--erp: Equity Risk Premium for the Real-World transform (default: 0.055 i.e., 5.5%).

--save-plot [FILENAME]: Saves the chart to a file instead of displaying it.

Output Interpretation

The Scenario Table

MARKET PROB: The probability the market assigns to each bucket (Bear/Base/Bull).

CONTRIBUTION: The weighted value (Target $\times$ Prob).

TOTAL EV: The Market-Implied Expected Value.

Strategic Decision Rules (The Alpha Signal)

Comparing your fundamental conviction to the Market EV generates the trade thesis. Differentiation comes from diverging on Magnitude (Price) or Likelihood (Probability).

Market EV vs. Your Target

Implication

Market EV > Your Target

Market is Optimistic. The options market is pricing in a breakout. If you are short, this is a "Crowded Long" (opportunity).

Market EV < Your Target

Market is Defensive. The options market is pricing in downside protection. If you are long, this is a "Contrarian Long" (opportunity).

Pitching the "Probability Gap":

"The market assigns only a 30% probability to our Bull case (Defensive Skew). However, our research suggests the true probability is 60%. We are buying this mispricing of risk."

Warnings & Troubleshooting

"WARNING: NEGATIVE GAMMA DETECTED"

If the script outputs a warning that $\gamma < 0$, it means the market is implying Risk-Seeking Behavior.

Cause: Call options are overpriced relative to Puts (High Skew), often due to "Squeeze" speculation or bad data in deep-ITM options.

Fix: Run with --plot to check for artifacts. Try tightening the bounds (e.g., --bounds 0.7 2.0).

"Spike" at the Left Edge

If the plot shows a massive vertical line at the lowest price:

Cause: Illiquid deep-in-the-money options are distorting the spline.

Fix: Increase the lower bound (e.g., from 0.4 to 0.6 or 0.7).