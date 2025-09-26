Here's a comprehensive README for your combined option pricing and hedging analysis project:

```markdown
# Option Pricing and Hedging Strategy Analysis

This project combines comprehensive option pricing models with advanced hedging strategy simulations for financial derivatives analysis.

## Overview

The project implements various option pricing methods and hedging strategies, including:
- Black-Scholes analytical pricing
- Binomial tree pricing
- Monte Carlo simulations using Euler method
- Exact Geometric Brownian Motion (GBM) pricing
- Asian option pricing
- Delta hedging strategy simulations

## Features

### Option Pricing Methods
- **Black-Scholes Formula**: Analytical solution for European call options
- **Binomial Tree Model**: Discrete-time lattice approach
- **Monte Carlo Simulation**: 
  - Euler method with different time steps
  - Exact GBM formula implementation
- **Asian Options**: Average price options using Monte Carlo

### Hedging Analysis
- **Delta Hedging**: Dynamic hedging strategy simulation
- **Multiple Frequencies**: Weekly, monthly, and daily rebalancing
- **Volatility Sensitivity**: Analysis with different market volatilities
- **P&L Distribution**: Statistical analysis of hedging performance

## Requirements

```python
numpy
pandas
matplotlib
scipy
math (built-in)
```

## Installation

```bash
pip install numpy pandas matplotlib scipy
```

## Usage

### Basic Execution
```python
python main1.py
```

This will run both the option pricing analysis and hedging strategy simulations.

### Individual Components

#### Option Pricing Analysis
```python
from main1 import run_original_analysis
run_original_analysis()
```

#### Hedging Strategy Analysis
```python
from main1 import run_hedging_analysis
run_hedging_analysis()
```

### Custom Option Pricing
```python
from main1 import black_scholes_call_price, binomial_tree_call_price

# Parameters
S0 = 4507.66  # Current stock price
K = 4600      # Strike price
T = 0.25      # Time to maturity (3 months)
r = 0.035     # Risk-free rate
sigma = 0.173 # Volatility

# Calculate prices
bs_price = black_scholes_call_price(S0, K, T, r, sigma)
binomial_price = binomial_tree_call_price(S0, K, T, r, sigma, 300)
```

## Project Structure

```
├── main1.py                    # Combined analysis script              
├── README.md                   # This file
└── plots/                    # Generated plots and results
```

## Code Organization

The code is structured into the following sections:

1. **Black-Scholes and Binomial Tree Functions**
   - `black_scholes_call_price()`: Analytical BS formula
   - `binomial_tree_call_price()`: Discrete lattice model
   - `black_scholes()`: BS with delta calculation

2. **Monte Carlo Simulation Methods**
   - `euler_method_call_price()`: 3-step Euler simulation
   - `euler_method()`: 63-step daily simulation
   - `exact_gb_call_price()`: Exact GBM formula
   - `asian_call_option_price()`: Asian option pricing

3. **Hedging Simulation Functions**
   - `simulate_hedging_strategy()`: Main hedging simulation
   - `simulate_F()`: Hedging with different pricing volatilities

4. **Utility Functions**
   - `call_price()`: Price calculation and display
   - `plot_histogram()`: P&L visualization

## Default Parameters

The analysis uses the following market parameters:

```python
S0 = 4507.66        # Initial stock price
K = 4600            # Strike price
T = 3/12            # 3 months to expiration
r = 0.035           # 3.5% risk-free rate
sigma = 0.173       # Annualized volatility (17.3%)
```


```

### Hedging Analysis
The program generates:
- P&L histograms for different hedging frequencies
- Statistical summaries of hedging performance
- Comparison of strategies under different market conditions

## Key Insights

1. **Convergence Analysis**: Monte Carlo prices converge to analytical solutions as simulation count increases
2. **Hedging Frequency**: Higher frequency hedging reduces P&L variance but increases transaction costs
3. **Volatility Impact**: Misestimated volatility significantly affects hedging performance
4. **Asian Options**: Show lower prices due to averaging effect reducing volatility

## Mathematical Models

### Black-Scholes Formula
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```
where:
- d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
- d₂ = d₁ - σ√T

### Euler Method
```
S_{i+1} = S_i + S_i(rΔt + σ√Δt·φ)
```
where φ ~ N(0,1)





## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
- Hull, J. C. Options, Futures, and Other Derivatives
- Shreve, S. E. Stochastic Calculus for Finance


