#pragma once

#include <chrono>
#include <array>

namespace options {

struct Greeks {
    double delta = 0.0;     // Price sensitivity to underlying
    double gamma = 0.0;     // Delta sensitivity to underlying  
    double theta = 0.0;     // Price sensitivity to time
    double vega = 0.0;      // Price sensitivity to volatility
    double rho = 0.0;       // Price sensitivity to interest rate
    double epsilon = 0.0;   // Price sensitivity to dividend yield
    
    Greeks() = default;
    Greeks(double d, double g, double t, double v, double r, double e = 0.0)
        : delta(d), gamma(g), theta(t), vega(v), rho(r), epsilon(e) {}
};

struct PricingResult {
    double option_price = 0.0;
    Greeks greeks;
    double implied_volatility = 0.0;
    std::chrono::nanoseconds computation_time{0};
    std::size_t iterations_used = 0;
    bool converged = false;
    double numerical_error = 0.0;
    
    PricingResult() = default;
    
    PricingResult(double price, const Greeks& g)
        : option_price(price), greeks(g), converged(true) {}
};

struct MonteCarloResult {
    double option_price = 0.0;
    double standard_error = 0.0;
    double confidence_interval_lower = 0.0;
    double confidence_interval_upper = 0.0;
    std::size_t paths_used = 0;
    std::chrono::nanoseconds computation_time{0};
    double convergence_rate = 0.0;
    
    MonteCarloResult() = default;
    
    void calculate_confidence_interval(double confidence_level = 0.95);
};

struct BinomialResult {
    double option_price = 0.0;
    double early_exercise_premium = 0.0;
    std::size_t optimal_exercise_node = 0;
    std::size_t tree_depth = 0;
    std::chrono::nanoseconds computation_time{0};
    
    BinomialResult() = default;
    
    BinomialResult(double price, std::size_t depth)
        : option_price(price), tree_depth(depth) {}
};

template<typename T>
struct alignas(64) BatchResults {
    static constexpr std::size_t BATCH_SIZE = 64;
    
    T option_price[BATCH_SIZE];
    T delta[BATCH_SIZE];
    T gamma[BATCH_SIZE];
    T theta[BATCH_SIZE];
    T vega[BATCH_SIZE];
    T rho[BATCH_SIZE];
    bool converged[BATCH_SIZE];
    std::size_t count = 0;
    
    void clear() { count = 0; }
    bool is_full() const { return count >= BATCH_SIZE; }
    bool empty() const { return count == 0; }
};

struct PerformanceMetrics {
    std::chrono::nanoseconds total_time{0};
    std::chrono::nanoseconds avg_pricing_time{0};
    std::chrono::nanoseconds min_pricing_time{std::chrono::nanoseconds::max()};
    std::chrono::nanoseconds max_pricing_time{0};
    std::size_t total_options_priced = 0;
    double throughput_per_second = 0.0;
    
    void update(std::chrono::nanoseconds pricing_time) {
        total_time += pricing_time;
        ++total_options_priced;
        
        if (pricing_time < min_pricing_time) {
            min_pricing_time = pricing_time;
        }
        if (pricing_time > max_pricing_time) {
            max_pricing_time = pricing_time;
        }
        
        avg_pricing_time = total_time / total_options_priced;
        throughput_per_second = 1e9 / avg_pricing_time.count();
    }
    
    void reset() {
        total_time = std::chrono::nanoseconds{0};
        avg_pricing_time = std::chrono::nanoseconds{0};
        min_pricing_time = std::chrono::nanoseconds::max();
        max_pricing_time = std::chrono::nanoseconds{0};
        total_options_priced = 0;
        throughput_per_second = 0.0;
    }
};

}