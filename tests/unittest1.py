import unittest
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import trapezoid
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_expectations import MarketExpectations

class TestMarketExpectations(unittest.TestCase):
    
    def setUp(self):
        # Patch yfinance to prevent network calls during initialization
        with patch('yfinance.Ticker') as MockTicker:
            self.mock_ticker = MockTicker.return_value
            
            # Setup default mock data
            self.mock_ticker.history.return_value = pd.DataFrame(
                {'Close': [100.0, 100.0, 100.0, 100.0, 100.0]}
            )
            self.mock_ticker.options = ("2025-01-01", "2025-06-01")
            
            # Initialize with standard values
            self.me = MarketExpectations("TEST", risk_free_rate=0.05, dividend_yield=0.0)

    def test_black_scholes_vanilla(self):
        """
        Verify BS price against a known benchmark (No dividends).
        S=100, K=100, T=1, r=0.05, sigma=0.2
        Expected Price approx 10.4506
        """
        price = self.me._black_scholes_call(
            S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2
        )
        self.assertAlmostEqual(price, 10.4506, places=3)

    def test_black_scholes_with_dividends(self):
        """
        Verify Merton adjustment (With dividends).
        S=100, K=100, T=1, r=0.05, q=0.02, sigma=0.2
        Dividends reduce the forward price, so Call value must be lower than vanilla.
        """
        price_no_div = self.me._black_scholes_call(100, 100, 1.0, 0.05, 0.0, 0.2)
        price_with_div = self.me._black_scholes_call(100, 100, 1.0, 0.05, 0.02, 0.2)
        
        self.assertTrue(price_with_div < price_no_div, 
                        f"Dividend payer should have lower call price. Got {price_with_div} >= {price_no_div}")
        
        # Theoretical check: Put-Call parity or external calc check
        # For r=5%, q=2%, T=1, S=100, K=100, vol=20% -> ~9.22
        self.assertAlmostEqual(price_with_div, 9.22, places=1)

    def test_shimko_flat_volatility_recovery(self):
        """
        Math Check:
        If we feed a constant IV (Flat Smile) into the Shimko smoothing,
        the resulting pdf must be lognormal centered at the forward price.
        """
        # 1. Mock Option Chain with constant IV
        # Widen range to 10-300 to capture full tails (prevent truncation error)
        strikes = np.linspace(10, 300, 200)
        ivs = np.full_like(strikes, 0.20) # flat 20% vol
        
        mock_chain = MagicMock()
        mock_chain.calls = pd.DataFrame({
            'strike': strikes,
            'impliedVolatility': ivs,
            'volume': [100] * 200,
            'openInterest': [100] * 200
        })
        self.mock_ticker.option_chain.return_value = mock_chain

        # 2. Run computation
        with patch('market_expectations.datetime') as mock_date:
            from datetime import datetime
            # Fix "now" and "expiry" to 1 year difference
            mock_date.now.return_value = datetime(2024, 1, 1)
            mock_date.strptime.return_value = datetime(2025, 1, 1)
            
            res = self.me._compute_risk_neutral_pdf("2025-01-01", 0.1, 3.0) # Widen bounds for test
            
            self.assertIsNotNone(res, "Shimko method failed to fit flat volatility.")
            
            # 3. Verify Mean of pdf = forward price
            # F = S * e^{(r-q)T}
            T = res['T']
            expected_mean = 100 * np.exp((0.05 - 0.0) * T)
            
            # Calculate mean from pdf
            pdf_mean = trapezoid(res['strikes'] * res['pdf'], res['strikes'])
            
            # Allow 1% error margin for numerical integration/polyfit artifacts
            error_pct = abs(pdf_mean - expected_mean) / expected_mean
            self.assertLess(error_pct, 0.01, 
                            f"Flat Vol did not recover Forward Price. Mean: {pdf_mean:.2f}, Expected: {expected_mean:.2f}")

    def test_real_world_transform_neutral(self):
        """
        If expected return (ERP) matches the risk-neutral drift exactly, 
        Gamma should be 0 (risk neutral = real world).
        """
        # create a dummy risk-neutral dataset
        x = np.linspace(80, 120, 100)
        pdf = norm.pdf(x, 100, 10)
        pdf /= trapezoid(pdf, x) # normalize
        
        data_q = {
            'strikes': x,
            'pdf': pdf,
            'T': 1.0,
            'expiry': 'mock'
        }
        
        # Set ERP such that target mean matches current mean exactly
        # Current Mean is approx 100.
        # Target Mean = S * e^(r + erp - q) * T
        # We want 100 = 100 * e^(0.05 + erp - 0) * 1
        # => 0.05 + erp = 0 => erp = -0.05
        
        res = self.me._convert_to_real_world(data_q, erp=-0.05)
        
        self.assertAlmostEqual(res['gamma'], 0.0, delta=0.5, 
                               msg=f"Gamma should be near 0 when Target Mean = Neutral Mean. Got {res['gamma']}")

    def test_real_world_transform_risk_averse(self):
        """
        If CAPM Target > Risk-Neutral Mean, Gamma should be positive 
        (Shifting probability weight to higher strikes).
        """
        x = np.linspace(50, 150, 100)
        pdf = norm.pdf(x, 100, 10) # Mean 100
        pdf /= trapezoid(pdf, x)
        
        data_q = { 'strikes': x, 'pdf': pdf, 'T': 1.0, 'expiry': 'mock' }
        
        # Target: Higher return (Standard CAPM)
        # S=100, r=0.05, ERP=0.05 -> Target Mean = 100 * e^0.1 = 110.5
        
        res = self.me._convert_to_real_world(data_q, erp=0.05)
        
        self.assertGreater(res['mean_p'], 100.0)
        self.assertGreater(res['gamma'], 0.0)

    def test_scenario_probability_sum(self):
        """
        Ensure Bear + Base + Bull probabilities roughly sum to 100% 
        after normalization.
        """
        # Inject the mock again for a full run
        strikes = np.linspace(50, 150, 50)
        ivs = np.full_like(strikes, 0.20)
        mock_chain = MagicMock()
        mock_chain.calls = pd.DataFrame({
            'strike': strikes, 'impliedVolatility': ivs,
            'volume': [100]*50, 'openInterest': [100]*50
        })
        self.mock_ticker.option_chain.return_value = mock_chain

        with patch('market_expectations.datetime') as mock_date:
            from datetime import datetime
            mock_date.now.return_value = datetime(2024, 1, 1)
            mock_date.strptime.return_value = datetime(2025, 1, 1)
            
            res = self.me.calculate_scenario_ev(
                bear=80, base=110, bull=130, 
                months=12, erp=0.05, min_pct=0.5, max_pct=1.5
            )
            
            total_prob = (res['bear_case']['prob'] + 
                          res['base_case']['prob'] + 
                          res['bull_case']['prob'])
            
            self.assertAlmostEqual(total_prob, 1.0, places=4)
            self.assertGreater(res['implied_ev'], 0)

if __name__ == '__main__':
    unittest.main()