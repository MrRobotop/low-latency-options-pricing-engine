#include "options/BlackScholes.hpp"
#include "options/Greeks.hpp"
#include "options/ImpliedVolatility.hpp"
#include "utils/Timer.hpp"
#include <iostream>
#include <iomanip>

using namespace options;
using namespace options::utils;

int main() {
    std::cout << "=== Basic Options Pricing Example ===\n\n";
    
    // Define option contracts
    OptionSpec call_option(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "AAPL");
    OptionSpec put_option(OptionType::PUT, ExerciseStyle::EUROPEAN, 100.0, 0.25, "AAPL");
    
    // Market data: S=$105, σ=20%, r=5%, q=2%
    MarketData market(105.0, 0.20, 0.05, 0.02);
    
    std::cout << "Market Data:\n";
    std::cout << "  Spot Price: $" << market.spot_price << "\n";
    std::cout << "  Volatility: " << market.volatility * 100 << "%\n";
    std::cout << "  Risk-free Rate: " << market.risk_free_rate * 100 << "%\n";
    std::cout << "  Dividend Yield: " << market.dividend_yield * 100 << "%\n";
    std::cout << "  Strike Price: $" << call_option.strike << "\n";
    std::cout << "  Time to Expiry: " << call_option.time_to_expiry << " years\n\n";
    
    // Price call option
    std::cout << "=== Call Option Pricing ===\n";
    HighResolutionTimer call_timer;
    auto call_result = BlackScholesPricer::price_european_option(call_option, market);
    auto call_time = call_timer.elapsed();
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Call Option Price: $" << call_result.option_price << "\n";
    std::cout << "Greeks:\n";
    std::cout << "  Delta: " << call_result.greeks.delta << "\n";
    std::cout << "  Gamma: " << call_result.greeks.gamma << "\n";
    std::cout << "  Theta: " << call_result.greeks.theta << " (per day)\n";
    std::cout << "  Vega:  " << call_result.greeks.vega << " (per 1% vol)\n";
    std::cout << "  Rho:   " << call_result.greeks.rho << " (per 1% rate)\n";
    std::cout << "Computation Time: " << call_time.count() << " nanoseconds\n\n";
    
    // Price put option
    std::cout << "=== Put Option Pricing ===\n";
    HighResolutionTimer put_timer;
    auto put_result = BlackScholesPricer::price_european_option(put_option, market);
    auto put_time = put_timer.elapsed();
    
    std::cout << "Put Option Price: $" << put_result.option_price << "\n";
    std::cout << "Greeks:\n";
    std::cout << "  Delta: " << put_result.greeks.delta << "\n";
    std::cout << "  Gamma: " << put_result.greeks.gamma << "\n";
    std::cout << "  Theta: " << put_result.greeks.theta << " (per day)\n";
    std::cout << "  Vega:  " << put_result.greeks.vega << " (per 1% vol)\n";
    std::cout << "  Rho:   " << put_result.greeks.rho << " (per 1% rate)\n";
    std::cout << "Computation Time: " << put_time.count() << " nanoseconds\n\n";
    
    // Verify put-call parity
    const double forward = market.spot_price * std::exp(-market.dividend_yield * call_option.time_to_expiry);
    const double pv_strike = call_option.strike * std::exp(-market.risk_free_rate * call_option.time_to_expiry);
    const double put_call_parity = call_result.option_price - put_result.option_price - (forward - pv_strike);
    
    std::cout << "=== Put-Call Parity Verification ===\n";
    std::cout << "C - P - (F - PV(K)) = " << put_call_parity << "\n";
    std::cout << "Error: " << std::abs(put_call_parity) << "\n";
    std::cout << (std::abs(put_call_parity) < 1e-10 ? "✓ PASSED" : "✗ FAILED") << "\n\n";
    
    // Implied volatility calculation
    std::cout << "=== Implied Volatility Calculation ===\n";
    const double market_price = call_result.option_price * 1.05; // 5% premium
    
    ImpliedVolatilitySolver iv_solver;
    HighResolutionTimer iv_timer;
    const double implied_vol = iv_solver.solve_newton_raphson(call_option, market, market_price);
    auto iv_time = iv_timer.elapsed();
    
    std::cout << "Market Price: $" << market_price << "\n";
    std::cout << "Theoretical Price: $" << call_result.option_price << "\n";
    std::cout << "Market Volatility: " << market.volatility * 100 << "%\n";
    std::cout << "Implied Volatility: " << implied_vol * 100 << "%\n";
    std::cout << "Volatility Difference: " << (implied_vol - market.volatility) * 100 << " bps\n";
    std::cout << "IV Computation Time: " << iv_time.count() << " nanoseconds\n\n";
    
    // Sensitivity analysis
    std::cout << "=== Sensitivity Analysis ===\n";
    std::cout << "Price sensitivity to 1% spot move: $" << call_result.greeks.delta * market.spot_price * 0.01 << "\n";
    std::cout << "Price sensitivity to 1% vol move: $" << call_result.greeks.vega << "\n";
    std::cout << "Price decay per day: $" << call_result.greeks.theta << "\n";
    std::cout << "Gamma P&L for 1% spot move: $" << 
                 0.5 * call_result.greeks.gamma * std::pow(market.spot_price * 0.01, 2) << "\n\n";
    
    // Performance summary
    std::cout << "=== Performance Summary ===\n";
    std::cout << "Call pricing: " << call_time.count() << " ns (" << 
                 call_time.count() / 1000.0 << " μs)\n";
    std::cout << "Put pricing: " << put_time.count() << " ns (" << 
                 put_time.count() / 1000.0 << " μs)\n";
    std::cout << "IV calculation: " << iv_time.count() << " ns (" << 
                 iv_time.count() / 1000.0 << " μs)\n";
    std::cout << "Total execution: " << (call_time + put_time + iv_time).count() << " ns\n";
    
    const double pricing_throughput = 2.0 * 1e9 / (call_time + put_time).count();
    std::cout << "Pricing throughput: " << std::fixed << std::setprecision(0) << 
                 pricing_throughput << " options/second\n";
    
    return 0;
}