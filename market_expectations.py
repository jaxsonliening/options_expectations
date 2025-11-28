#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
from typing import Dict, Optional

class MarketExpectations:
    def __init__(self, ticker: str, risk_free_rate: float = 0.04, dividend_yield: float = 0.0):
        self.ticker = ticker.upper()
        self.r = risk_free_rate
        self.q = dividend_yield
        self.stock = yf.Ticker(self.ticker)
        
        try:
            # Fetch minimal history to get spot
            hist = self.stock.history(period="5d")
            if hist.empty:
                raise ValueError(f"No price data found for {self.ticker}")
            self.spot_price = hist['Close'].iloc[-1]
            self.expirations = self.stock.options
            if not self.expirations:
                raise ValueError(f"No option chain found for {self.ticker}")
        except Exception as e:
            raise ValueError(f"Initialization failed: {e}")

    def _get_nearest_expiry(self, target_months: float) -> str:
        target_date = datetime.now() + timedelta(days=int(30.44 * target_months))
        edates = [datetime.strptime(e, "%Y-%m-%d") for e in self.expirations]
        diffs = [abs((e - target_date).days) for e in edates]
        return self.expirations[int(np.argmin(diffs))]

    def _black_scholes_call(self, S, K, T, r, q, sigma):
        """
        Black-Scholes-Merton Call Price (Vectorized).
        Adjusted for continuous dividend yield (q).
        """
        # T is scalar here. If T is near zero, return intrinsic.
        if T < 1e-6:
             return np.maximum(S - K, 0)

        # d1 numerator includes (r - q)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Call = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def _compute_risk_neutral_pdf(self, expiry: str, min_pct: float, max_pct: float) -> Optional[Dict]:
        """
        Derives PDF using Vol-Smoothing (Shimko-style).
        """
        try:
            opts = self.stock.option_chain(expiry)
            calls = opts.calls
        except:
            return None

        exp_date = datetime.strptime(expiry, "%Y-%m-%d")
        T = (exp_date - datetime.now()).days / 365.25
        if T < 0.001: return None 

        # 1. Data Hygiene (Strict Filtering)
        calls = calls[
            (calls['strike'] > self.spot_price * min_pct) & 
            (calls['strike'] < self.spot_price * max_pct) &
            (calls['impliedVolatility'] > 0.01) &
            (calls['impliedVolatility'] < 2.0) &
            ((calls['volume'] > 0) | (calls['openInterest'] > 0))
        ].copy()
        
        if len(calls) < 5: return None

        strikes = calls['strike'].values
        ivs = calls['impliedVolatility'].values

        # 2. Fit the Volatility Smile (Quadratic Poly)
        try:
            poly_coeffs = np.polyfit(strikes, ivs, 2)
            vol_curve = np.poly1d(poly_coeffs)
        except:
            return None

        # 3. Generate Dense Grid
        grid_min, grid_max = strikes.min(), strikes.max()
        x_grid = np.linspace(grid_min, grid_max, 1000)
        
        # Predict Smooth IVs for the grid
        smooth_ivs = vol_curve(x_grid)
        smooth_ivs = np.maximum(smooth_ivs, 0.01) 
        
        # 4. Convert Smooth IV -> Smooth Call Prices (using q)
        smooth_calls = self._black_scholes_call(self.spot_price, x_grid, T, self.r, self.q, smooth_ivs)
        
        # 5. Numerical Differentiation (Breeden-Litzenberger)
        # PDF = e^(rT) * d^2C/dK^2 
        # Note: Breeden-Litzenberger formula is invariant to q, 
        # but the inputs (C) are now adjusted for it.
        dk = x_grid[1] - x_grid[0]
        second_deriv = np.gradient(np.gradient(smooth_calls, dk), dk)
        
        pdf_values = np.exp(self.r * T) * second_deriv
        pdf_values = np.maximum(pdf_values, 0) 
        
        integral = trapezoid(pdf_values, x_grid)
        if integral <= 0: return None
            
        pdf_normalized = pdf_values / integral

        return {
            "expiry": expiry,
            "T": T,
            "strikes": x_grid,
            "pdf": pdf_normalized,
            "raw_iv_poly": poly_coeffs
        }

    def _convert_to_real_world(self, data: Dict, erp: float) -> Dict:
        """
        Bliss & Panigirtzoglou Utility Transformation.
        """
        x = data['strikes']
        pdf_q = data['pdf']
        T = data['T']
        
        # Target Mean Logic:
        # CAPM gives Expected Total Return = r + ERP.
        # Total Return = Price Appreciation + Dividend Yield.
        # Price Appreciation = (r + ERP) - q.
        # Target Price = Spot * e^( (r + ERP - q) * T )
        
        target_mean = self.spot_price * np.exp((self.r + erp - self.q) * T)
        
        def get_stats(gamma):
            # Power Utility Transform
            with np.errstate(over='ignore', invalid='ignore'):
                num = pdf_q * (x ** gamma)
            
            if np.any(~np.isfinite(num)): return None
                
            denom = trapezoid(num, x)
            if denom == 0 or not np.isfinite(denom): return None
            
            pdf_p = num / denom
            mean_p = trapezoid(x * pdf_p, x)
            return pdf_p, mean_p

        def objective(gamma):
            stats = get_stats(gamma)
            if stats is None: return 1e9
            _, mean_p = stats
            return (mean_p - target_mean) ** 2

        # Solve for Gamma (-10 to 30)
        res = minimize_scalar(objective, bounds=(-10, 30), method='bounded')
        best_gamma = res.x
        
        stats = get_stats(best_gamma)
        if stats:
            pdf_p, mean_p = stats
        else:
            pdf_p, mean_p = pdf_q, 0 
            best_gamma = 0

        return {
            "strikes": x,
            "pdf_q": pdf_q,
            "pdf_p": pdf_p,
            "gamma": best_gamma,
            "mean_p": mean_p,
            "T": T,
            "expiry": data['expiry']
        }

    def calculate_scenario_ev(self, bear: float, base: float, bull: float, months: float, erp: float, min_pct: float, max_pct: float):
        expiry = self._get_nearest_expiry(months)
        data_q = self._compute_risk_neutral_pdf(expiry, min_pct, max_pct)
        if not data_q: return {"error": "Insufficient data or bad fit"}
        
        data_p = self._convert_to_real_world(data_q, erp)
        x = data_p['strikes']
        pdf = data_p['pdf_p']
        
        # CDF calculation
        cdf = np.cumsum(pdf) * (x[1] - x[0])
        if cdf[-1] > 0: cdf = cdf / cdf[-1]
        
        cut_bear_base = (bear + base) / 2
        cut_base_bull = (base + bull) / 2
        
        p_bear = np.interp(cut_bear_base, x, cdf)
        p_bull = 1.0 - np.interp(cut_base_bull, x, cdf)
        p_base = max(0, 1.0 - p_bear - p_bull) 
        
        total_p = p_bear + p_base + p_bull
        if total_p > 0:
            p_bear /= total_p
            p_base /= total_p
            p_bull /= total_p
        
        ev = (p_bear * bear) + (p_base * base) + (p_bull * bull)
        
        return {
            "expiry": expiry,
            "spot": self.spot_price,
            "gamma": data_p['gamma'],
            "bear_case": {"pt": bear, "prob": p_bear},
            "base_case": {"pt": base, "prob": p_base},
            "bull_case": {"pt": bull, "prob": p_bull},
            "implied_ev": ev
        }
    
    def analyze_target(self, target: float, months: float, erp: float, min_pct: float, max_pct: float):
        expiry = self._get_nearest_expiry(months)
        data_q = self._compute_risk_neutral_pdf(expiry, min_pct, max_pct)
        if not data_q: return {"error": "Insufficient data"}
        
        data_p = self._convert_to_real_world(data_q, erp)
        x = data_p['strikes']
        
        cdf = np.cumsum(data_p['pdf_p']) * (x[1] - x[0])
        if cdf[-1] > 0: cdf = cdf / cdf[-1]
        
        prob_below = np.interp(target, x, cdf)
        
        return {
            "expiry": expiry,
            "spot": self.spot_price,
            "target": target,
            "prob_above": 1.0 - prob_below,
            "gamma": data_p['gamma']
        }

    def plot_dual_distribution(self, months, erp, min_pct, max_pct, save_path=None):
        expiry = self._get_nearest_expiry(months)
        data_q = self._compute_risk_neutral_pdf(expiry, min_pct, max_pct)
        if not data_q: 
            print("No data for plotting")
            return

        data_p = self._convert_to_real_world(data_q, erp)
        
        plt.figure(figsize=(12, 7))
        plt.plot(data_p['strikes'], data_p['pdf_q'], label='Risk-Neutral (Market Price)', linestyle='--', color='gray', alpha=0.7)
        plt.plot(data_p['strikes'], data_p['pdf_p'], label=f'Real-World (Forecast) | γ={data_p["gamma"]:.2f}', color='#007acc', linewidth=2.5)
        plt.axvline(self.spot_price, color='black', linestyle=':', linewidth=1.5, label=f'Spot: ${self.spot_price:.2f}')
        plt.fill_between(data_p['strikes'], data_p['pdf_p'], alpha=0.1, color='#007acc')
        plt.title(f"{self.ticker} Implied Probability Density ({months}m Horizon)\nMethod: Volatility Smoothing (Shimko)", fontsize=12)
        plt.xlabel("Price ($)")
        plt.ylabel("Probability Density")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlim(data_p['strikes'].min(), data_p['strikes'].max())
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Equity Analysis Tool: Options-Implied Valuation (Vol-Smoothing)")
    parser.add_argument("ticker", type=str)
    parser.add_argument("--months", type=float, default=12, help="Horizon (months)")
    parser.add_argument("--r", type=float, default=0.042, help="Risk-free rate")
    parser.add_argument("--div", type=float, default=0.0, help="Dividend Yield (decimal, e.g. 0.03 for 3%)")
    parser.add_argument("--erp", type=float, default=0.055, help="Equity Risk Premium")
    parser.add_argument("--bounds", nargs=2, type=float, default=[0.5, 2.5], metavar=('MIN_X', 'MAX_X'),
                        help="Filter range as multiple of spot. Default: 0.5 2.5")
    parser.add_argument("--ev", nargs=3, type=float, metavar=('BEAR', 'BASE', 'BULL'), 
                        help="Calculate EV using Bear/Base/Bull targets")
    parser.add_argument("--target", type=float, help="Calculate probability of exceeding specific target")
    parser.add_argument("--plot", action="store_true", help="Visualize Q vs P distributions")
    parser.add_argument("--save-plot", type=str, help="Save plot to file")

    args = parser.parse_args()

    try:
        me = MarketExpectations(args.ticker, args.r, args.div)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n--- Analysis for {args.ticker.upper()} ---")
    print(f"Spot: ${me.spot_price:.2f} | Risk-Free: {args.r:.1%} | ERP: {args.erp:.1%} | Div: {args.div:.1%}")

    min_p, max_p = args.bounds

    if args.ev:
        bear, base, bull = args.ev
        res = me.calculate_scenario_ev(bear, base, bull, args.months, args.erp, min_p, max_p)
        
        if "error" in res:
            print(f"Error: {res['error']}")
        else:
            gamma = res['gamma']
            print(f"\n[Real-World Scenario Weighting] ({res['expiry']})")
            print(f"Implied Risk Aversion (Gamma): {gamma:.2f}")
            print(f"\n{'CASE':<10} {'TARGET':>10} {'MARKET PROB':>15} {'CONTRIBUTION':>15}")
            print("-" * 55)
            row_fmt = "{:<10} ${:>9.2f} {:>14.1%} ${:>14.2f}"
            print(row_fmt.format("Bear", res['bear_case']['pt'], res['bear_case']['prob'], res['bear_case']['pt'] * res['bear_case']['prob']))
            print(row_fmt.format("Base", res['base_case']['pt'], res['base_case']['prob'], res['base_case']['pt'] * res['base_case']['prob']))
            print(row_fmt.format("Bull", res['bull_case']['pt'], res['bull_case']['prob'], res['bull_case']['pt'] * res['bull_case']['prob']))
            print("-" * 55)
            print(f"{'TOTAL EV':<10} {'':>10} {'100.0%':>15} ${res['implied_ev']:>14.2f}")

    if args.target:
        res = me.analyze_target(args.target, args.months, args.erp, min_p, max_p)
        if "error" in res:
            print(f"Error: {res['error']}")
        else:
            print(f"\n[Single Target Analysis]")
            print(f"Prob > ${args.target:.2f}: {res['prob_above']:.2%}")

    if args.plot or args.save_plot:
        print(f"\nGenerating distribution plot...")
        me.plot_dual_distribution(args.months, args.erp, min_p, max_p, args.save_plot)

if __name__ == "__main__":
    main()