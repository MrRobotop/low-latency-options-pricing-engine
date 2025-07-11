#include "options/PricingEngine.hpp"
#include "options/BlackScholes.hpp"
#include "options/MonteCarlo.hpp"
#include "options/AmericanOptions.hpp"
#include "options/ImpliedVolatility.hpp"
#include "utils/Timer.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace options;
using namespace options::utils;

void demonstrate_european_pricing() {
    std::cout << "\n=== European Options Pricing Demo ===\n";
    
    OptionSpec call_option(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "AAPL");
    OptionSpec put_option(OptionType::PUT, ExerciseStyle::EUROPEAN, 100.0, 0.25, "AAPL");
    
    MarketData market(105.0, 0.20, 0.05, 0.02);
    
    auto call_result = BlackScholesPricer::price_european_option(call_option, market);
    auto put_result = BlackScholesPricer::price_european_option(put_option, market);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Call Option:\n";
    std::cout << "  Price: $" << call_result.option_price << "\n";
    std::cout << "  Delta: " << call_result.greeks.delta << "\n";
    std::cout << "  Gamma: " << call_result.greeks.gamma << "\n";
    std::cout << "  Theta: " << call_result.greeks.theta << "\n";
    std::cout << "  Vega:  " << call_result.greeks.vega << "\n";
    std::cout << "  Rho:   " << call_result.greeks.rho << "\n";
    std::cout << "  Computation Time: " << call_result.computation_time.count() << " ns\n";
    
    std::cout << "\nPut Option:\n";
    std::cout << "  Price: $" << put_result.option_price << "\n";
    std::cout << "  Delta: " << put_result.greeks.delta << "\n";
    std::cout << "  Gamma: " << put_result.greeks.gamma << "\n";
    std::cout << "  Theta: " << put_result.greeks.theta << "\n";
    std::cout << "  Vega:  " << put_result.greeks.vega << "\n";
    std::cout << "  Rho:   " << put_result.greeks.rho << "\n";
    std::cout << "  Computation Time: " << put_result.computation_time.count() << " ns\n";
}

void demonstrate_american_pricing() {
    std::cout << "\n=== American Options Pricing Demo ===\n";
    
    OptionSpec american_put(OptionType::PUT, ExerciseStyle::AMERICAN, 100.0, 0.25, "AAPL");
    MarketData market(95.0, 0.25, 0.05, 0.03);
    
    BinomialTreePricer binomial_pricer;
    auto american_result = binomial_pricer.price_american_option(american_put, market);
    
    OptionSpec european_put(OptionType::PUT, ExerciseStyle::EUROPEAN, 100.0, 0.25, "AAPL");
    auto european_result = BlackScholesPricer::price_european_option(european_put, market);
    
    const double early_exercise_premium = american_result.option_price - european_result.option_price;
    
    std::cout << "American Put Option:\n";
    std::cout << "  Price: $" << american_result.option_price << "\n";
    std::cout << "  European Price: $" << european_result.option_price << "\n";
    std::cout << "  Early Exercise Premium: $" << early_exercise_premium << "\n";
    std::cout << "  Tree Depth: " << american_result.tree_depth << "\n";
    std::cout << "  Computation Time: " << american_result.computation_time.count() << " ns\n";
}

void demonstrate_monte_carlo() {
    std::cout << "\n=== Monte Carlo Simulation Demo ===\n";
    
    OptionSpec asian_call(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "AAPL");
    MarketData market(105.0, 0.20, 0.05, 0.02);
    
    MonteCarloEngine::Configuration mc_config;
    mc_config.num_paths = 1000000;
    mc_config.variance_reduction = VarianceReductionTechnique::ANTITHETIC_VARIATES;
    mc_config.enable_vectorization = true;
    
    MonteCarloEngine mc_engine(mc_config);
    
    auto european_mc = mc_engine.price_european_option(asian_call, market);
    auto asian_mc = mc_engine.price_asian_option(asian_call, market, 252);
    
    std::cout << "European Option (Monte Carlo):\n";
    std::cout << "  Price: $" << european_mc.option_price << "\n";
    std::cout << "  Standard Error: $" << european_mc.standard_error << "\n";
    std::cout << "  Paths Used: " << european_mc.paths_used << "\n";
    std::cout << "  Computation Time: " << european_mc.computation_time.count() << " ns\n";
    
    std::cout << "\nAsian Option (Monte Carlo):\n";
    std::cout << "  Price: $" << asian_mc.option_price << "\n";
    std::cout << "  Standard Error: $" << asian_mc.standard_error << "\n";
    std::cout << "  Paths Used: " << asian_mc.paths_used << "\n";
    std::cout << "  Computation Time: " << asian_mc.computation_time.count() << " ns\n";
}

void demonstrate_implied_volatility() {
    std::cout << "\n=== Implied Volatility Demo ===\n";
    
    OptionSpec option(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "AAPL");
    MarketData market(105.0, 0.20, 0.05, 0.02);
    
    auto theoretical_result = BlackScholesPricer::price_european_option(option, market);
    const double market_price = theoretical_result.option_price * 1.05;
    
    ImpliedVolatilitySolver iv_solver;
    
    HighResolutionTimer timer;
    const double implied_vol = iv_solver.solve_newton_raphson(option, market, market_price);
    const auto computation_time = timer.elapsed();
    
    std::cout << "Implied Volatility Calculation:\n";
    std::cout << "  Market Price: $" << market_price << "\n";
    std::cout << "  Theoretical Price: $" << theoretical_result.option_price << "\n";
    std::cout << "  Market Volatility: " << market.volatility * 100 << "%\n";
    std::cout << "  Implied Volatility: " << implied_vol * 100 << "%\n";
    std::cout << "  Computation Time: " << computation_time.count() << " ns\n";
}

void demonstrate_portfolio_pricing() {
    std::cout << "\n=== Portfolio Pricing Demo ===\n";
    
    PricingEngine::Configuration config;
    config.enable_caching = true;
    config.enable_vectorization = true;
    config.enable_multithreading = true;
    
    PricingEngine engine(config);
    
    std::vector<OptionSpec> options;
    std::vector<MarketData> market_data;
    
    for (int i = 0; i < 1000; ++i) {
        const double strike = 90.0 + i * 0.2;
        const double expiry = 0.1 + (i % 10) * 0.05;
        const OptionType type = (i % 2 == 0) ? OptionType::CALL : OptionType::PUT;
        
        options.emplace_back(type, ExerciseStyle::EUROPEAN, strike, expiry, "PORTFOLIO");
        market_data.emplace_back(100.0 + (i % 20) * 0.5, 0.15 + (i % 5) * 0.01, 0.05, 0.02);
    }
    
    HighResolutionTimer timer;
    auto results = engine.price_portfolio(options, market_data);
    const auto total_time = timer.elapsed();
    
    double total_portfolio_value = 0.0;
    for (const auto& result : results) {
        total_portfolio_value += result.option_price;
    }
    
    const auto performance_metrics = engine.get_performance_metrics();
    
    std::cout << "Portfolio Pricing Results:\n";
    std::cout << "  Options Priced: " << results.size() << "\n";
    std::cout << "  Total Portfolio Value: $" << std::fixed << std::setprecision(2) 
              << total_portfolio_value << "\n";
    std::cout << "  Total Computation Time: " << total_time.count() / 1e6 << " ms\n";
    std::cout << "  Average Time per Option: " << total_time.count() / results.size() << " ns\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0) 
              << (results.size() * 1e9 / total_time.count()) << " options/second\n";
}

void performance_benchmark() {
    std::cout << "\n=== Performance Benchmark ===\n";
    
    OptionSpec option(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "BENCHMARK");
    MarketData market(105.0, 0.20, 0.05, 0.02);
    
    constexpr std::size_t num_iterations = 100000;
    LatencyMeasurement latency_stats;
    
    std::cout << "Running " << num_iterations << " pricing iterations...\n";
    
    for (std::size_t i = 0; i < num_iterations; ++i) {
        HighResolutionTimer timer;
        auto result = BlackScholesPricer::price_european_option(option, market);
        latency_stats.add_sample(timer.elapsed());
    }
    
    std::cout << "Latency Statistics (nanoseconds):\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(0) << latency_stats.get_average() << "\n";
    std::cout << "  Minimum: " << latency_stats.get_min() << "\n";
    std::cout << "  Maximum: " << latency_stats.get_max() << "\n";
    std::cout << "  50th Percentile: " << latency_stats.get_percentile(0.50) << "\n";
    std::cout << "  95th Percentile: " << latency_stats.get_percentile(0.95) << "\n";
    std::cout << "  99th Percentile: " << latency_stats.get_percentile(0.99) << "\n";
    std::cout << "  99.9th Percentile: " << latency_stats.get_percentile(0.999) << "\n";
}

int main() {
    std::cout << "=========================================\n";
    std::cout << "Low-Latency Options Pricing Engine Demo\n";
    std::cout << "=========================================\n";
    
    try {
        demonstrate_european_pricing();
        demonstrate_american_pricing();
        demonstrate_monte_carlo();
        demonstrate_implied_volatility();
        demonstrate_portfolio_pricing();
        performance_benchmark();
        
        std::cout << "\n=== Demo Complete ===\n";
        std::cout << "All pricing engines executed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}