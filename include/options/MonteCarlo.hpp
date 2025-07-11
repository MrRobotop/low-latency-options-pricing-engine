#pragma once

#include "../types/Option.hpp"
#include "../types/Market.hpp"
#include "../types/Results.hpp"
#include "../math/Statistics.hpp"
#include "../utils/Timer.hpp"
#include <vector>
#include <random>
#include <thread>
#include <future>
#include <algorithm>
#include <immintrin.h>

namespace options {

enum class VarianceReductionTechnique {
    NONE,
    ANTITHETIC_VARIATES,
    CONTROL_VARIATES,
    STRATIFIED_SAMPLING,
    IMPORTANCE_SAMPLING
};

class MonteCarloEngine {
public:
    struct Configuration {
        std::size_t num_paths = 1000000;
        std::size_t num_threads = std::thread::hardware_concurrency();
        VarianceReductionTechnique variance_reduction = VarianceReductionTechnique::ANTITHETIC_VARIATES;
        bool enable_vectorization = true;
        std::uint64_t random_seed = std::random_device{}();
        double confidence_level = 0.95;
    };

    explicit MonteCarloEngine(const Configuration& config = Configuration{})
        : config_(config) {}

    MonteCarloResult price_european_option(
        const OptionSpec& option,
        const MarketData& market) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const double S0 = market.spot_price;
        const double K = option.strike;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double vol = market.volatility;
        const double q = market.dividend_yield;
        
        const double drift = (r - q - 0.5 * vol * vol) * T;
        const double vol_sqrt_T = vol * std::sqrt(T);
        const double discount = std::exp(-r * T);
        
        std::vector<double> payoffs;
        payoffs.reserve(config_.num_paths);
        
        if (config_.enable_vectorization) {
            payoffs = simulate_paths_vectorized(S0, K, drift, vol_sqrt_T, option.type);
        } else {
            payoffs = simulate_paths_standard(S0, K, drift, vol_sqrt_T, option.type);
        }
        
        if (config_.variance_reduction == VarianceReductionTechnique::CONTROL_VARIATES) {
            apply_control_variates(payoffs, option, market);
        }
        
        const double mean_payoff = math::FastStatistics<double>::mean(payoffs);
        const double option_price = discount * mean_payoff;
        const double std_dev = math::FastStatistics<double>::standard_deviation(payoffs);
        const double standard_error = std_dev / std::sqrt(static_cast<double>(payoffs.size()));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto computation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        MonteCarloResult result;
        result.option_price = option_price;
        result.standard_error = discount * standard_error;
        result.paths_used = payoffs.size();
        result.computation_time = computation_time;
        result.calculate_confidence_interval(config_.confidence_level);
        
        return result;
    }

    MonteCarloResult price_asian_option(
        const OptionSpec& option,
        const MarketData& market,
        std::size_t monitoring_points = 252) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const double S0 = market.spot_price;
        const double K = option.strike;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double vol = market.volatility;
        const double q = market.dividend_yield;
        
        const double dt = T / monitoring_points;
        const double drift = (r - q - 0.5 * vol * vol) * dt;
        const double vol_sqrt_dt = vol * std::sqrt(dt);
        const double discount = std::exp(-r * T);
        
        std::vector<double> payoffs = simulate_asian_paths(
            S0, K, drift, vol_sqrt_dt, monitoring_points, option.type);
        
        const double mean_payoff = math::FastStatistics<double>::mean(payoffs);
        const double option_price = discount * mean_payoff;
        const double std_dev = math::FastStatistics<double>::standard_deviation(payoffs);
        const double standard_error = std_dev / std::sqrt(static_cast<double>(payoffs.size()));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto computation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        MonteCarloResult result;
        result.option_price = option_price;
        result.standard_error = discount * standard_error;
        result.paths_used = payoffs.size();
        result.computation_time = computation_time;
        result.calculate_confidence_interval(config_.confidence_level);
        
        return result;
    }

    MonteCarloResult price_barrier_option(
        const OptionSpec& option,
        const MarketData& market,
        std::size_t monitoring_points = 252) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const double S0 = market.spot_price;
        const double K = option.strike;
        const double B = option.barrier_level;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double vol = market.volatility;
        const double q = market.dividend_yield;
        
        const double dt = T / monitoring_points;
        const double drift = (r - q - 0.5 * vol * vol) * dt;
        const double vol_sqrt_dt = vol * std::sqrt(dt);
        const double discount = std::exp(-r * T);
        
        std::vector<double> payoffs = simulate_barrier_paths(
            S0, K, B, drift, vol_sqrt_dt, monitoring_points, option.type, option.barrier_type);
        
        const double mean_payoff = math::FastStatistics<double>::mean(payoffs);
        const double option_price = discount * mean_payoff;
        const double std_dev = math::FastStatistics<double>::standard_deviation(payoffs);
        const double standard_error = std_dev / std::sqrt(static_cast<double>(payoffs.size()));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto computation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        MonteCarloResult result;
        result.option_price = option_price;
        result.standard_error = discount * standard_error;
        result.paths_used = payoffs.size();
        result.computation_time = computation_time;
        result.calculate_confidence_interval(config_.confidence_level);
        
        return result;
    }

    void set_random_seed(std::uint64_t seed) {
        config_.random_seed = seed;
    }

private:
    Configuration config_;

    std::vector<double> simulate_paths_standard(
        double S0, double K, double drift, double vol_sqrt_T, OptionType type) {
        
        std::vector<double> payoffs;
        payoffs.reserve(config_.num_paths);
        
        if (config_.num_threads > 1) {
            return simulate_paths_parallel(S0, K, drift, vol_sqrt_T, type);
        }
        
        math::RandomNumberGenerator rng(config_.random_seed);
        
        const std::size_t actual_paths = config_.variance_reduction == VarianceReductionTechnique::ANTITHETIC_VARIATES ?
                                        config_.num_paths / 2 : config_.num_paths;
        
        for (std::size_t i = 0; i < actual_paths; ++i) {
            const double z = rng.normal();
            const double S_T = S0 * std::exp(drift + vol_sqrt_T * z);
            
            double payoff;
            if (type == OptionType::CALL) {
                payoff = std::max(S_T - K, 0.0);
            } else {
                payoff = std::max(K - S_T, 0.0);
            }
            payoffs.push_back(payoff);
            
            if (config_.variance_reduction == VarianceReductionTechnique::ANTITHETIC_VARIATES) {
                const double S_T_anti = S0 * std::exp(drift - vol_sqrt_T * z);
                double payoff_anti;
                if (type == OptionType::CALL) {
                    payoff_anti = std::max(S_T_anti - K, 0.0);
                } else {
                    payoff_anti = std::max(K - S_T_anti, 0.0);
                }
                payoffs.push_back(payoff_anti);
            }
        }
        
        return payoffs;
    }

#ifdef __AVX2__
    std::vector<double> simulate_paths_vectorized(
        double S0, double K, double drift, double vol_sqrt_T, OptionType type) {
        
        std::vector<double> payoffs;
        payoffs.reserve(config_.num_paths);
        
        const std::size_t simd_width = 4;
        const std::size_t vectorized_paths = (config_.num_paths / simd_width) * simd_width;
        
        math::RandomNumberGenerator rng(config_.random_seed);
        
        __m256d S0_vec = _mm256_set1_pd(S0);
        __m256d K_vec = _mm256_set1_pd(K);
        __m256d drift_vec = _mm256_set1_pd(drift);
        __m256d vol_sqrt_T_vec = _mm256_set1_pd(vol_sqrt_T);
        __m256d zero_vec = _mm256_setzero_pd();
        
        for (std::size_t i = 0; i < vectorized_paths; i += simd_width) {
            alignas(32) double z_vals[4];
            for (int j = 0; j < 4; ++j) {
                z_vals[j] = rng.normal();
            }
            
            __m256d z_vec = _mm256_load_pd(z_vals);
            __m256d exponent = _mm256_add_pd(drift_vec, _mm256_mul_pd(vol_sqrt_T_vec, z_vec));
            __m256d exp_result = _mm256_exp_pd(exponent);
            __m256d S_T = _mm256_mul_pd(S0_vec, exp_result);
            
            __m256d payoff_vec;
            if (type == OptionType::CALL) {
                __m256d diff = _mm256_sub_pd(S_T, K_vec);
                payoff_vec = _mm256_max_pd(diff, zero_vec);
            } else {
                __m256d diff = _mm256_sub_pd(K_vec, S_T);
                payoff_vec = _mm256_max_pd(diff, zero_vec);
            }
            
            alignas(32) double payoff_vals[4];
            _mm256_store_pd(payoff_vals, payoff_vec);
            
            for (int j = 0; j < 4; ++j) {
                payoffs.push_back(payoff_vals[j]);
            }
        }
        
        for (std::size_t i = vectorized_paths; i < config_.num_paths; ++i) {
            const double z = rng.normal();
            const double S_T = S0 * std::exp(drift + vol_sqrt_T * z);
            
            double payoff;
            if (type == OptionType::CALL) {
                payoff = std::max(S_T - K, 0.0);
            } else {
                payoff = std::max(K - S_T, 0.0);
            }
            payoffs.push_back(payoff);
        }
        
        return payoffs;
    }
#else
    std::vector<double> simulate_paths_vectorized(
        double S0, double K, double drift, double vol_sqrt_T, OptionType type) {
        
        return simulate_paths_standard(S0, K, drift, vol_sqrt_T, type);
    }
#endif

    std::vector<double> simulate_paths_parallel(
        double S0, double K, double drift, double vol_sqrt_T, OptionType type) {
        
        const std::size_t paths_per_thread = config_.num_paths / config_.num_threads;
        std::vector<std::future<std::vector<double>>> futures;
        
        for (std::size_t t = 0; t < config_.num_threads; ++t) {
            const std::size_t start_path = t * paths_per_thread;
            const std::size_t end_path = (t == config_.num_threads - 1) ? 
                                        config_.num_paths : (t + 1) * paths_per_thread;
            const std::size_t thread_paths = end_path - start_path;
            
            futures.push_back(std::async(std::launch::async, [=]() {
                math::RandomNumberGenerator thread_rng(config_.random_seed + t);
                std::vector<double> thread_payoffs;
                thread_payoffs.reserve(thread_paths);
                
                for (std::size_t i = 0; i < thread_paths; ++i) {
                    const double z = thread_rng.normal();
                    const double S_T = S0 * std::exp(drift + vol_sqrt_T * z);
                    
                    double payoff;
                    if (type == OptionType::CALL) {
                        payoff = std::max(S_T - K, 0.0);
                    } else {
                        payoff = std::max(K - S_T, 0.0);
                    }
                    thread_payoffs.push_back(payoff);
                }
                
                return thread_payoffs;
            }));
        }
        
        std::vector<double> all_payoffs;
        all_payoffs.reserve(config_.num_paths);
        
        for (auto& future : futures) {
            auto thread_payoffs = future.get();
            all_payoffs.insert(all_payoffs.end(), thread_payoffs.begin(), thread_payoffs.end());
        }
        
        return all_payoffs;
    }

    std::vector<double> simulate_asian_paths(
        double S0, double K, double drift, double vol_sqrt_dt, 
        std::size_t monitoring_points, OptionType type) {
        
        std::vector<double> payoffs;
        payoffs.reserve(config_.num_paths);
        
        math::RandomNumberGenerator rng(config_.random_seed);
        
        for (std::size_t path = 0; path < config_.num_paths; ++path) {
            double S = S0;
            double sum_S = 0.0;
            
            for (std::size_t step = 0; step < monitoring_points; ++step) {
                const double z = rng.normal();
                S *= std::exp(drift + vol_sqrt_dt * z);
                sum_S += S;
            }
            
            const double average_S = sum_S / monitoring_points;
            
            double payoff;
            if (type == OptionType::CALL) {
                payoff = std::max(average_S - K, 0.0);
            } else {
                payoff = std::max(K - average_S, 0.0);
            }
            payoffs.push_back(payoff);
        }
        
        return payoffs;
    }

    std::vector<double> simulate_barrier_paths(
        double S0, double K, double B, double drift, double vol_sqrt_dt,
        std::size_t monitoring_points, OptionType option_type, BarrierType barrier_type) {
        
        std::vector<double> payoffs;
        payoffs.reserve(config_.num_paths);
        
        math::RandomNumberGenerator rng(config_.random_seed);
        
        for (std::size_t path = 0; path < config_.num_paths; ++path) {
            double S = S0;
            bool barrier_hit = false;
            
            for (std::size_t step = 0; step < monitoring_points; ++step) {
                const double z = rng.normal();
                S *= std::exp(drift + vol_sqrt_dt * z);
                
                switch (barrier_type) {
                    case BarrierType::UP_AND_OUT:
                    case BarrierType::UP_AND_IN:
                        if (S >= B) barrier_hit = true;
                        break;
                    case BarrierType::DOWN_AND_OUT:
                    case BarrierType::DOWN_AND_IN:
                        if (S <= B) barrier_hit = true;
                        break;
                    default:
                        break;
                }
            }
            
            double payoff = 0.0;
            
            if (option_type == OptionType::CALL) {
                payoff = std::max(S - K, 0.0);
            } else {
                payoff = std::max(K - S, 0.0);
            }
            
            switch (barrier_type) {
                case BarrierType::UP_AND_OUT:
                case BarrierType::DOWN_AND_OUT:
                    if (barrier_hit) payoff = 0.0;
                    break;
                case BarrierType::UP_AND_IN:
                case BarrierType::DOWN_AND_IN:
                    if (!barrier_hit) payoff = 0.0;
                    break;
                default:
                    break;
            }
            
            payoffs.push_back(payoff);
        }
        
        return payoffs;
    }

    void apply_control_variates(
        std::vector<double>& payoffs,
        const OptionSpec& option,
        const MarketData& market) {
        
        auto bs_result = BlackScholesPricer::price_european_option(option, market);
        const double analytical_price = bs_result.option_price;
        
        const double sample_mean = math::FastStatistics<double>::mean(payoffs);
        const double control_adjustment = analytical_price - sample_mean;
        
        for (auto& payoff : payoffs) {
            payoff += control_adjustment;
        }
    }
};

}