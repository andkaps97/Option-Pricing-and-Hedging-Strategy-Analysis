import numpy as np
import pandas as pd
import math
import scipy.stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt


# ============================================================================
# BLACK-SCHOLES AND BINOMIAL TREE PRICING FUNCTIONS
# ============================================================================

def black_scholes_call_price(S0, K, T, r, sigma):
    """Black-Scholes call option pricing formula"""
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def binomial_tree_call_price(S0, K, T, r, sigma, n):
    """Binomial tree call option pricing"""
    deltaT = T / n
    u = math.exp(sigma * math.sqrt(deltaT))
    d = 1 / u
    q = (math.exp(r * deltaT) - d) / (u - d)

    # Create stock price tree
    stock_tree = np.zeros([n + 1, n + 1])
    for j in range(n + 1):
        stock_tree[j, n] = S0 * (u ** (n - j)) * (d ** j)

    # Calculate option price at terminal nodes
    option_tree = np.zeros([n + 1, n + 1])
    for j in range(n + 1):
        option_tree[j, n] = max(0, stock_tree[j, n] - K)

    # Work backward
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = (q * option_tree[j, i + 1] + (1 - q) * option_tree[j + 1, i + 1]) * math.exp(
                -r * deltaT)

    return option_tree[0, 0]


def black_scholes(S, K, T, r, sigma):
    """Black-Scholes formula returning both price and delta"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return call_price, delta


# ============================================================================
# MONTE CARLO SIMULATION METHODS
# ============================================================================

def euler_method_call_price(M):
    """Euler method for option pricing with N=3 steps"""
    total_payoff = 0
    S0 = 4507.66
    K = 4600
    T = 3 / 12
    r = 0.035
    sigma = 0.0502 * np.sqrt(12)
    N = 3
    deltaT = T / N

    for _ in range(M):
        S = S0
        for _ in range(N):
            phi = np.random.normal(0, 1)
            S += S * (r * deltaT + sigma * math.sqrt(deltaT) * phi)

        total_payoff += max(S - K, 0)

    # Average payoff and discounting
    option_price = math.exp(-r * T) * total_payoff / M
    return option_price


def euler_method(M):
    """Euler method with N=63 steps (daily)"""
    total_payoff = 0
    S0 = 4507.66
    K = 4600
    T = 3 / 12
    r = 0.035
    sigma = 0.0502 * np.sqrt(12)
    N = 63
    deltaT = T / N

    for _ in range(M):
        S = S0
        for _ in range(N):
            phi = np.random.normal(0, 1)
            S += S * (r * deltaT + sigma * math.sqrt(deltaT) * phi)

        total_payoff += max(S - K, 0)

    # Average payoff and discounting
    option_price = math.exp(-r * T) * total_payoff / M
    return option_price


def exact_gb_call_price(M):
    """Exact geometric Brownian motion formula"""
    total_payoff = 0
    S0 = 4507.66
    K = 4600
    T = 3 / 12
    r = 0.035
    sigma = 0.0502 * np.sqrt(12)

    for _ in range(M):
        phi = np.random.normal(0, 1)
        ST = S0 * math.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * phi)
        total_payoff += max(ST - K, 0)

    # Average payoff and discounting
    option_price = math.exp(-r * T) * total_payoff / M
    return option_price


def asian_call_option_price():
    """Asian call option pricing using Monte Carlo"""
    total_payoff = 0
    S0 = 4507.66
    K = 4600
    T = 3 / 12
    r = 0.035
    sigma = 0.0502 * np.sqrt(12)
    N = 63
    deltaT = T / N
    M = 10000
    n_last_month = 21

    for _ in range(M):
        S = [S0]
        for _ in range(N):
            phi = np.random.normal(0, 1)
            next_S = S[-1] + S[-1] * (r * deltaT + sigma * math.sqrt(deltaT) * phi)
            S.append(next_S)

        average_last_month = sum(S[-n_last_month:]) / n_last_month
        total_payoff += max(average_last_month - K, 0)

    # Average payoff and discounting
    option_price = math.exp(-r * T) * total_payoff / M
    return option_price


# ============================================================================
# HEDGING SIMULATION FUNCTIONS
# ============================================================================

def simulate_hedging_strategy(S0, K, T, r, sigma, mu, n_simulations, n_steps, hedging_frequency, n_options=1000):
    """Simulate hedging strategy with different frequencies"""
    dt = T / n_steps
    hedge_interval = int(n_steps / hedging_frequency)
    n_simulations = int(n_simulations)

    # Store the P&L for each simulation
    PnL = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Simulate a price path
        path = S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt)
                                     * np.random.randn(n_steps), axis=0))
        path = np.insert(path, 0, S0)  # adding the initial asset price at t=0

        hedge_portfolio = 0
        cash = black_scholes(S0, K, T, r, sigma)[0] * n_options  # initial cash from option premium
        stock_quantity = 0

        for t in range(0, n_steps + 1, hedge_interval):
            time_to_expiry = T - t * dt

            # Calculate option price and delta for the current time step
            if time_to_expiry > 0:
                option_price, delta = black_scholes(path[t], K, time_to_expiry, r, sigma)
            else:
                option_price = max(path[t] - K, 0)  # option payoff at maturity
                delta = 1 if path[t] > K else 0  # delta at maturity

            # Rebalance portfolio
            new_stock_quantity = delta * n_options
            stock_transaction = new_stock_quantity - stock_quantity
            stock_quantity = new_stock_quantity
            cash -= stock_transaction * path[t]  # cash spent/acquired after rebalancing stocks
            cash *= np.exp(r * dt * hedge_interval)  # interest on cash

            hedge_portfolio = stock_quantity * path[t] + cash

        # P&L for this simulation run
        PnL[i] = hedge_portfolio - option_price * n_options

    return PnL


def simulate_F(S0, K, T, r, sigma, mu, option_pricing_vol, n_simulations, n_steps, hedging_frequency, n_options=1000):
    """Simulate hedging with different volatility for option pricing"""
    dt = T / n_steps
    hedge_interval = int(n_steps / hedging_frequency)

    # Store the P&L for each simulation
    PnL = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Simulate a price path
        path = S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt)
                                     * np.random.randn(n_steps), axis=0))
        path = np.insert(path, 0, S0)  # adding the initial asset price at t=0

        hedge_portfolio = 0
        cash = black_scholes(S0, K, T, r, sigma)[0] * n_options  # initial cash from option premium
        stock_quantity = 0

        for t in range(0, n_steps + 1, hedge_interval):
            time_to_expiry = T - t * dt

            # Calculate option price and delta for the current time step
            if time_to_expiry > 0:
                option_price, delta = black_scholes(path[t], K, time_to_expiry, r, option_pricing_vol)
            else:
                option_price = max(path[t] - K, 0)  # option payoff at maturity
                delta = 1 if path[t] > K else 0  # delta at maturity

            # Rebalance portfolio
            new_stock_quantity = delta * n_options
            stock_transaction = new_stock_quantity - stock_quantity
            stock_quantity = new_stock_quantity
            cash -= stock_transaction * path[t]  # cash spent/acquired after rebalancing stocks
            cash *= np.exp(r * dt * hedge_interval)  # interest on cash

            hedge_portfolio = stock_quantity * path[t] + cash

        # P&L for this simulation run
        PnL[i] = hedge_portfolio - option_price * n_options

    return PnL


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def call_price(S0, X, T, r, sigma):
    """Calculate call option price and total amount for trader"""
    d1 = (math.log(S0 / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Calculate the price of the call option
    C = S0 * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)

    # Total amount the trader receives
    total_amount = C * 100 * 10

    print(f"The price of one call option is: ${C:.2f}")
    print(f"The total amount received by the trader is: ${total_amount:.2f}")


def plot_histogram(data, title, xlabel, ylabel):
    """Plot histogram for P&L analysis"""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_original_analysis():
    """Run the original option pricing analysis from main.py"""
    S0 = 4507.66
    K = 4600
    T = 3 / 12
    r = 0.035
    sigma = 0.0502 * np.sqrt(12)
    n = 300

    binomial_price = binomial_tree_call_price(S0, K, T, r, sigma, n)
    print(f"Binomial Tree Call Price: {binomial_price:.2f}")

    bs_price = black_scholes_call_price(S0, K, T, r, sigma)
    print(f"Black-Scholes Call Price: {bs_price:.2f}")

    M_values = [100, 500, 1000, 5000, 10000]

    for M in M_values:
        price = euler_method_call_price(M)
        print(f"European Call Option Price using Euler Method for M={M}: {price:.2f}")

    print("Option Prices for Δt equal to 1 trading day:")
    for M in M_values:
        price = euler_method(M)
        print(f"For M={M}: {price:.2f}")

    print("Option Prices using exact GBM formula:")
    for M in M_values:
        price = exact_gb_call_price(M)
        print(f"For {M} draws: {price:.2f}")

    asian_price = asian_call_option_price()
    print(f"Price of the Asian Call Option: {asian_price:.2f}")


def run_hedging_analysis():
    """Run the hedging strategy analysis from ttthtttttttttttt.py"""
    S0 = 4600
    X = 4600
    T = 0.25
    r = 0.035
    sigma = 0.0502 * np.sqrt(12)
    mu = 0.04
    n_simulations = 5000

    call_price(S0, X, T, r, sigma)

    n_steps_weekly = 13
    n_steps_daily = 63
    hedging_frequencies = {
        'weekly': n_steps_weekly,
        'monthly': 3,
        'daily': n_steps_daily
    }

    # Simulations
    results = {}
    for freq, steps in hedging_frequencies.items():
        results[freq] = simulate_hedging_strategy(S0, X, T, r, sigma, mu, n_simulations, steps, hedging_frequency=steps)

    # With a different mu
    results['daily_higher_mu'] = simulate_hedging_strategy(S0, X, T, r, sigma, 0.08, n_simulations, n_steps_daily,
                                                           hedging_frequency=n_steps_daily)

    # Plotting the results
    for freq, pnl in results.items():
        plot_histogram(pnl, f'P&L Histogram with {freq} hedging', 'Profit and Loss (P&L)', 'Frequency')

    hedging_frequency = n_steps_daily
    sigma_market = sigma + 0.05  # 5%-points higher

    # Simulating the strategy with a higher market volatility for option pricing
    PnL_high_vol = simulate_F(S0, X, T, r, sigma, 0.08, sigma_market, n_simulations, n_steps_daily,
                              hedging_frequency)

    # Plotting the results
    plot_histogram(PnL_high_vol, 'P&L Histogram with higher market volatility', 'Profit and Loss (P&L)', 'Frequency')


def main():
    """Main function to run both analyses"""
    print("=" * 80)
    print("ORIGINAL OPTION PRICING ANALYSIS")
    print("=" * 80)
    run_original_analysis()

    print("\n" + "=" * 80)
    print("HEDGING STRATEGY ANALYSIS")
    print("=" * 80)
    run_hedging_analysis()


if __name__ == '__main__':
    main()
