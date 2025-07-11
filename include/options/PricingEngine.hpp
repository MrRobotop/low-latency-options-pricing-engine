#pragma once

#include "../types/Option.hpp"
#include "../types/Market.hpp"
#include "../types/Results.hpp"
#include "../utils/Timer.hpp"
#include "../utils/MemoryPool.hpp"
#include "BlackScholes.hpp"
#include "Greeks.hpp"
#include <memory>
#include <atomic>
#include <thread>
#include <vector>

namespace options {

class PricingEngine {
public:
    struct Configuration {
        bool enable_caching = true;
        bool enable_vectorization = true;
        bool enable_multithreading = true;
        std::size_t thread_pool_size = std::thread::hardware_concurrency();
        std::size_t cache_size = 10000;
        double cache_tolerance = 1e-6;
    };

    explicit PricingEngine(const Configuration& config = Configuration{})
        : config_(config), memory_pool_(std::make_unique<utils::MemoryPool<PricingResult>>()) {
        
        if (config_.enable_multithreading && config_.thread_pool_size > 0) {
            initialize_thread_pool();
        }
        
        if (config_.enable_caching) {
            initialize_cache();
        }
    }

    PricingResult price_option(const OptionSpec& option, const MarketData& market) {
        PROFILE_FUNCTION();
        
        if (config_.enable_caching) {
            if (auto cached_result = get_cached_result(option, market)) {
                return *cached_result;
            }
        }
        
        PricingResult result;
        
        switch (option.exercise_style) {
            case ExerciseStyle::EUROPEAN:
                if (option.barrier_type == BarrierType::NONE) {
                    result = BlackScholesPricer::price_european_option(option, market);
                } else {
                    result = BlackScholesBarrierPricer::price_barrier_option(option, market);
                }
                break;
                
            case ExerciseStyle::AMERICAN:
                result = price_american_option(option, market);
                break;
                
            case ExerciseStyle::BERMUDAN:
                result = price_bermudan_option(option, market);
                break;
        }
        
        if (config_.enable_caching && result.converged) {
            cache_result(option, market, result);
        }
        
        update_performance_metrics(result.computation_time);
        
        return result;
    }

    std::vector<PricingResult> price_portfolio(
        const std::vector<OptionSpec>& options,
        const std::vector<MarketData>& market_data) {
        
        PROFILE_FUNCTION();
        
        if (options.size() != market_data.size()) {
            throw std::invalid_argument("Options and market data vectors must have the same size");
        }
        
        std::vector<PricingResult> results(options.size());
        
        if (config_.enable_multithreading && options.size() > 100) {
            price_portfolio_parallel(options, market_data, results);
        } else if (config_.enable_vectorization && options.size() > 10) {
            price_portfolio_vectorized(options, market_data, results);
        } else {
            price_portfolio_sequential(options, market_data, results);
        }
        
        return results;
    }

    double calculate_implied_volatility(
        const OptionSpec& option,
        const MarketData& market,
        double market_price,
        double tolerance = 1e-6) {
        
        PROFILE_FUNCTION();
        
        return BlackScholesPricer::calculate_implied_volatility(
            option, market, market_price, tolerance);
    }

    Greeks calculate_greeks(const OptionSpec& option, const MarketData& market) {
        PROFILE_FUNCTION();
        
        if (option.exercise_style == ExerciseStyle::EUROPEAN && option.barrier_type == BarrierType::NONE) {
            return GreeksCalculator::calculate_analytical_greeks(option, market);
        } else {
            auto pricing_func = [this](const OptionSpec& opt, const MarketData& mkt) -> double {
                return price_option(opt, mkt).option_price;
            };
            return GreeksCalculator::calculate_numerical_greeks(option, market, pricing_func);
        }
    }

    PerformanceMetrics get_performance_metrics() const {
        return performance_metrics_;
    }

    void reset_performance_metrics() {
        performance_metrics_.reset();
    }

    std::size_t get_cache_size() const {
        return cache_size_.load(std::memory_order_relaxed);
    }

    void clear_cache() {
        if (config_.enable_caching) {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            pricing_cache_.clear();
            cache_size_.store(0, std::memory_order_relaxed);
        }
    }

private:
    struct CacheKey {
        OptionSpec option;
        MarketData market;
        
        bool operator==(const CacheKey& other) const {
            return std::abs(option.strike - other.option.strike) < config_.cache_tolerance &&
                   std::abs(option.time_to_expiry - other.option.time_to_expiry) < config_.cache_tolerance &&
                   std::abs(market.spot_price - other.market.spot_price) < config_.cache_tolerance &&
                   std::abs(market.volatility - other.market.volatility) < config_.cache_tolerance &&
                   std::abs(market.risk_free_rate - other.market.risk_free_rate) < config_.cache_tolerance &&
                   option.type == other.option.type &&
                   option.exercise_style == other.option.exercise_style;
        }
    };

    struct CacheKeyHash {
        std::size_t operator()(const CacheKey& key) const {
            std::size_t h1 = std::hash<double>{}(key.option.strike);
            std::size_t h2 = std::hash<double>{}(key.option.time_to_expiry);
            std::size_t h3 = std::hash<double>{}(key.market.spot_price);
            std::size_t h4 = std::hash<double>{}(key.market.volatility);
            std::size_t h5 = static_cast<std::size_t>(key.option.type);
            
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
        }
    };

    Configuration config_;
    std::unique_ptr<utils::MemoryPool<PricingResult>> memory_pool_;
    
    mutable std::mutex cache_mutex_;
    std::unordered_map<CacheKey, PricingResult, CacheKeyHash> pricing_cache_;
    std::atomic<std::size_t> cache_size_{0};
    
    mutable std::mutex performance_mutex_;
    PerformanceMetrics performance_metrics_;
    
    std::vector<std::thread> thread_pool_;
    std::atomic<bool> shutdown_requested_{false};

    void initialize_thread_pool() {
        // Implementation would create worker threads for parallel processing
    }

    void initialize_cache() {
        pricing_cache_.reserve(config_.cache_size);
    }

    std::optional<PricingResult> get_cached_result(const OptionSpec& option, const MarketData& market) const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        CacheKey key{option, market};
        auto it = pricing_cache_.find(key);
        
        if (it != pricing_cache_.end()) {
            return it->second;
        }
        
        return std::nullopt;
    }

    void cache_result(const OptionSpec& option, const MarketData& market, const PricingResult& result) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        if (cache_size_.load(std::memory_order_relaxed) >= config_.cache_size) {
            auto oldest_it = pricing_cache_.begin();
            pricing_cache_.erase(oldest_it);
            cache_size_.fetch_sub(1, std::memory_order_relaxed);
        }
        
        CacheKey key{option, market};
        pricing_cache_[key] = result;
        cache_size_.fetch_add(1, std::memory_order_relaxed);
    }

    void update_performance_metrics(std::chrono::nanoseconds computation_time) {
        std::lock_guard<std::mutex> lock(performance_mutex_);
        performance_metrics_.update(computation_time);
    }

    PricingResult price_american_option(const OptionSpec& option, const MarketData& market) {
        PROFILE_SCOPE("american_pricing");
        
        const std::size_t time_steps = 1000;
        const double dt = option.time_to_expiry / time_steps;
        const double u = std::exp(market.volatility * std::sqrt(dt));
        const double d = 1.0 / u;
        const double p = (std::exp(market.risk_free_rate * dt) - d) / (u - d);
        const double discount = std::exp(-market.risk_free_rate * dt);
        
        std::vector<double> option_values(time_steps + 1);
        
        for (std::size_t i = 0; i <= time_steps; ++i) {
            const double S_T = market.spot_price * std::pow(u, static_cast<double>(i)) * 
                              std::pow(d, static_cast<double>(time_steps - i));
            
            if (option.type == OptionType::CALL) {
                option_values[i] = std::max(S_T - option.strike, 0.0);
            } else {
                option_values[i] = std::max(option.strike - S_T, 0.0);
            }
        }
        
        for (int step = static_cast<int>(time_steps) - 1; step >= 0; --step) {
            for (int i = 0; i <= step; ++i) {
                const double S = market.spot_price * std::pow(u, static_cast<double>(i)) * 
                                std::pow(d, static_cast<double>(step - i));
                
                const double continuation_value = discount * (p * option_values[i + 1] + (1.0 - p) * option_values[i]);
                
                double exercise_value;
                if (option.type == OptionType::CALL) {
                    exercise_value = std::max(S - option.strike, 0.0);
                } else {
                    exercise_value = std::max(option.strike - S, 0.0);
                }
                
                option_values[i] = std::max(continuation_value, exercise_value);
            }
        }
        
        PricingResult result;
        result.option_price = option_values[0];
        result.converged = true;
        result.iterations_used = time_steps;
        
        return result;
    }

    PricingResult price_bermudan_option(const OptionSpec& option, const MarketData& market) {
        return price_american_option(option, market);
    }

    void price_portfolio_sequential(
        const std::vector<OptionSpec>& options,
        const std::vector<MarketData>& market_data,
        std::vector<PricingResult>& results) {
        
        for (std::size_t i = 0; i < options.size(); ++i) {
            results[i] = price_option(options[i], market_data[i]);
        }
    }

    void price_portfolio_vectorized(
        const std::vector<OptionSpec>& options,
        const std::vector<MarketData>& market_data,
        std::vector<PricingResult>& results) {
        
#ifdef __AVX2__
        constexpr std::size_t SIMD_WIDTH = 4;
        const std::size_t vectorized_count = (options.size() / SIMD_WIDTH) * SIMD_WIDTH;
        
        OptionBatch<double> option_batch;
        MarketDataBatch<double> market_batch;
        BatchResults<double> batch_results;
        
        for (std::size_t i = 0; i < vectorized_count; i += SIMD_WIDTH) {
            option_batch.clear();
            market_batch.clear();
            
            for (std::size_t j = 0; j < SIMD_WIDTH && (i + j) < options.size(); ++j) {
                const auto& opt = options[i + j];
                const auto& mkt = market_data[i + j];
                
                option_batch.strike[j] = opt.strike;
                option_batch.time_to_expiry[j] = opt.time_to_expiry;
                option_batch.option_type[j] = opt.type;
                
                market_batch.spot_price[j] = mkt.spot_price;
                market_batch.volatility[j] = mkt.volatility;
                market_batch.risk_free_rate[j] = mkt.risk_free_rate;
                market_batch.dividend_yield[j] = mkt.dividend_yield;
                
                option_batch.count++;
                market_batch.count++;
            }
            
            BlackScholesPricer::vectorized_pricing(option_batch, market_batch, batch_results);
            
            for (std::size_t j = 0; j < batch_results.count; ++j) {
                results[i + j].option_price = batch_results.option_price[j];
                results[i + j].greeks.delta = batch_results.delta[j];
                results[i + j].greeks.gamma = batch_results.gamma[j];
                results[i + j].greeks.theta = batch_results.theta[j];
                results[i + j].greeks.vega = batch_results.vega[j];
                results[i + j].greeks.rho = batch_results.rho[j];
                results[i + j].converged = batch_results.converged[j];
            }
        }
        
        for (std::size_t i = vectorized_count; i < options.size(); ++i) {
            results[i] = price_option(options[i], market_data[i]);
        }
#else
        price_portfolio_sequential(options, market_data, results);
#endif
    }

    void price_portfolio_parallel(
        const std::vector<OptionSpec>& options,
        const std::vector<MarketData>& market_data,
        std::vector<PricingResult>& results) {
        
        const std::size_t num_threads = std::min(config_.thread_pool_size, options.size());
        const std::size_t chunk_size = options.size() / num_threads;
        
        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        
        for (std::size_t t = 0; t < num_threads; ++t) {
            const std::size_t start_idx = t * chunk_size;
            const std::size_t end_idx = (t == num_threads - 1) ? options.size() : (t + 1) * chunk_size;
            
            workers.emplace_back([this, &options, &market_data, &results, start_idx, end_idx]() {
                for (std::size_t i = start_idx; i < end_idx; ++i) {
                    results[i] = price_option(options[i], market_data[i]);
                }
            });
        }
        
        for (auto& worker : workers) {
            worker.join();
        }
    }
};

}