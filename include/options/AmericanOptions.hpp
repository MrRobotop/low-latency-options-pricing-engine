#pragma once

#include "../types/Option.hpp"
#include "../types/Market.hpp"
#include "../types/Results.hpp"
#include "../utils/Timer.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace options {

class BinomialTreePricer {
public:
    struct TreeParameters {
        std::size_t time_steps = 1000;
        bool use_caching = true;
        bool optimize_memory = true;
        double convergence_tolerance = 1e-6;
    };

    explicit BinomialTreePricer(const TreeParameters& params = TreeParameters{})
        : params_(params) {}

    BinomialResult price_american_option(
        const OptionSpec& option,
        const MarketData& market) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const double S0 = market.spot_price;
        const double K = option.strike;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double vol = market.volatility;
        const double q = market.dividend_yield;
        
        if (T <= 0.0 || vol <= 0.0 || S0 <= 0.0) {
            return BinomialResult{};
        }
        
        const double dt = T / params_.time_steps;
        const double u = std::exp(vol * std::sqrt(dt));
        const double d = 1.0 / u;
        const double p = (std::exp((r - q) * dt) - d) / (u - d);
        const double discount = std::exp(-r * dt);
        
        BinomialResult result;
        
        if (params_.optimize_memory) {
            result = price_with_memory_optimization(S0, K, u, d, p, discount, option.type);
        } else {
            result = price_with_full_tree(S0, K, u, d, p, discount, option.type);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.computation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        result.tree_depth = params_.time_steps;
        
        return result;
    }

    BinomialResult price_american_option_adaptive(
        const OptionSpec& option,
        const MarketData& market,
        double target_accuracy = 1e-4) {
        
        std::size_t min_steps = 100;
        std::size_t max_steps = 5000;
        std::size_t current_steps = 500;
        
        double previous_price = 0.0;
        BinomialResult result;
        
        while (current_steps <= max_steps) {
            TreeParameters temp_params = params_;
            temp_params.time_steps = current_steps;
            
            BinomialTreePricer temp_pricer(temp_params);
            result = temp_pricer.price_american_option(option, market);
            
            if (current_steps > min_steps) {
                double price_diff = std::abs(result.option_price - previous_price);
                if (price_diff < target_accuracy) {
                    break;
                }
            }
            
            previous_price = result.option_price;
            current_steps = static_cast<std::size_t>(current_steps * 1.5);
        }
        
        return result;
    }

    double calculate_early_exercise_premium(
        const OptionSpec& option,
        const MarketData& market) {
        
        auto american_result = price_american_option(option, market);
        
        OptionSpec european_option = option;
        european_option.exercise_style = ExerciseStyle::EUROPEAN;
        auto european_result = price_american_option(european_option, market);
        
        return american_result.option_price - european_result.option_price;
    }

    std::vector<double> calculate_early_exercise_boundary(
        const OptionSpec& option,
        const MarketData& market,
        std::size_t boundary_points = 100) {
        
        const double T = option.time_to_expiry;
        std::vector<double> boundary(boundary_points);
        
        for (std::size_t i = 0; i < boundary_points; ++i) {
            const double t = (static_cast<double>(i) / (boundary_points - 1)) * T;
            boundary[i] = find_exercise_boundary_at_time(option, market, t);
        }
        
        return boundary;
    }

private:
    TreeParameters params_;

    BinomialResult price_with_memory_optimization(
        double S0, double K, double u, double d, double p, double discount, OptionType type) {
        
        std::vector<double> option_values(params_.time_steps + 1);
        std::vector<double> stock_prices(params_.time_steps + 1);
        
        for (std::size_t i = 0; i <= params_.time_steps; ++i) {
            stock_prices[i] = S0 * std::pow(u, static_cast<double>(i)) * 
                             std::pow(d, static_cast<double>(params_.time_steps - i));
            
            if (type == OptionType::CALL) {
                option_values[i] = std::max(stock_prices[i] - K, 0.0);
            } else {
                option_values[i] = std::max(K - stock_prices[i], 0.0);
            }
        }
        
        std::size_t optimal_exercise_node = 0;
        double max_early_exercise_value = 0.0;
        
        for (int step = static_cast<int>(params_.time_steps) - 1; step >= 0; --step) {
            for (int i = 0; i <= step; ++i) {
                const double S = S0 * std::pow(u, static_cast<double>(i)) * 
                                std::pow(d, static_cast<double>(step - i));
                
                const double continuation_value = discount * (p * option_values[i + 1] + (1.0 - p) * option_values[i]);
                
                double exercise_value;
                if (type == OptionType::CALL) {
                    exercise_value = std::max(S - K, 0.0);
                } else {
                    exercise_value = std::max(K - S, 0.0);
                }
                
                option_values[i] = std::max(continuation_value, exercise_value);
                
                if (exercise_value > continuation_value && exercise_value > max_early_exercise_value) {
                    max_early_exercise_value = exercise_value;
                    optimal_exercise_node = static_cast<std::size_t>(step * params_.time_steps + i);
                }
            }
        }
        
        BinomialResult result;
        result.option_price = option_values[0];
        result.early_exercise_premium = max_early_exercise_value;
        result.optimal_exercise_node = optimal_exercise_node;
        
        return result;
    }

    BinomialResult price_with_full_tree(
        double S0, double K, double u, double d, double p, double discount, OptionType type) {
        
        const std::size_t tree_size = (params_.time_steps + 1) * (params_.time_steps + 2) / 2;
        std::vector<double> option_tree(tree_size);
        std::vector<double> stock_tree(tree_size);
        
        auto get_index = [this](std::size_t step, std::size_t node) -> std::size_t {
            return step * (step + 1) / 2 + node;
        };
        
        for (std::size_t i = 0; i <= params_.time_steps; ++i) {
            const std::size_t idx = get_index(params_.time_steps, i);
            stock_tree[idx] = S0 * std::pow(u, static_cast<double>(i)) * 
                             std::pow(d, static_cast<double>(params_.time_steps - i));
            
            if (type == OptionType::CALL) {
                option_tree[idx] = std::max(stock_tree[idx] - K, 0.0);
            } else {
                option_tree[idx] = std::max(K - stock_tree[idx], 0.0);
            }
        }
        
        std::size_t optimal_exercise_node = 0;
        double max_early_exercise_value = 0.0;
        
        for (int step = static_cast<int>(params_.time_steps) - 1; step >= 0; --step) {
            for (int i = 0; i <= step; ++i) {
                const std::size_t current_idx = get_index(step, i);
                const std::size_t up_idx = get_index(step + 1, i + 1);
                const std::size_t down_idx = get_index(step + 1, i);
                
                stock_tree[current_idx] = S0 * std::pow(u, static_cast<double>(i)) * 
                                         std::pow(d, static_cast<double>(step - i));
                
                const double continuation_value = discount * (p * option_tree[up_idx] + (1.0 - p) * option_tree[down_idx]);
                
                double exercise_value;
                if (type == OptionType::CALL) {
                    exercise_value = std::max(stock_tree[current_idx] - K, 0.0);
                } else {
                    exercise_value = std::max(K - stock_tree[current_idx], 0.0);
                }
                
                option_tree[current_idx] = std::max(continuation_value, exercise_value);
                
                if (exercise_value > continuation_value && exercise_value > max_early_exercise_value) {
                    max_early_exercise_value = exercise_value;
                    optimal_exercise_node = current_idx;
                }
            }
        }
        
        BinomialResult result;
        result.option_price = option_tree[0];
        result.early_exercise_premium = max_early_exercise_value;
        result.optimal_exercise_node = optimal_exercise_node;
        
        return result;
    }

    double find_exercise_boundary_at_time(
        const OptionSpec& option,
        const MarketData& market,
        double time_to_exercise) {
        
        if (time_to_exercise <= 0.0) {
            return option.strike;
        }
        
        const double tolerance = 1e-6;
        const std::size_t max_iterations = 100;
        
        double S_low = option.strike * 0.5;
        double S_high = option.strike * 2.0;
        
        for (std::size_t iter = 0; iter < max_iterations; ++iter) {
            const double S_mid = (S_low + S_high) / 2.0;
            
            MarketData temp_market = market;
            temp_market.spot_price = S_mid;
            
            OptionSpec temp_option = option;
            temp_option.time_to_expiry = time_to_exercise;
            
            auto result = price_american_option(temp_option, temp_market);
            
            double exercise_value;
            if (option.type == OptionType::CALL) {
                exercise_value = std::max(S_mid - option.strike, 0.0);
            } else {
                exercise_value = std::max(option.strike - S_mid, 0.0);
            }
            
            const double diff = result.option_price - exercise_value;
            
            if (std::abs(diff) < tolerance) {
                return S_mid;
            }
            
            if (diff > 0) {
                if (option.type == OptionType::CALL) {
                    S_low = S_mid;
                } else {
                    S_high = S_mid;
                }
            } else {
                if (option.type == OptionType::CALL) {
                    S_high = S_mid;
                } else {
                    S_low = S_mid;
                }
            }
        }
        
        return (S_low + S_high) / 2.0;
    }
};

class TrinomialTreePricer {
public:
    struct TrinomialParameters {
        std::size_t time_steps = 500;
        double lambda = 1.5;
        bool adaptive_spacing = true;
    };

    explicit TrinomialTreePricer(const TrinomialParameters& params = TrinomialParameters{})
        : params_(params) {}

    BinomialResult price_american_option(
        const OptionSpec& option,
        const MarketData& market) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const double S0 = market.spot_price;
        const double K = option.strike;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double vol = market.volatility;
        const double q = market.dividend_yield;
        
        const double dt = T / params_.time_steps;
        const double dx = vol * std::sqrt(params_.lambda * dt);
        
        const double nu = r - q - 0.5 * vol * vol;
        const double edx = std::exp(dx);
        const double pu = 0.5 * ((vol * vol * dt + nu * nu * dt * dt) / (dx * dx) + nu * dt / dx);
        const double pd = 0.5 * ((vol * vol * dt + nu * nu * dt * dt) / (dx * dx) - nu * dt / dx);
        const double pm = 1.0 - pu - pd;
        const double discount = std::exp(-r * dt);
        
        const std::size_t num_nodes = 2 * params_.time_steps + 1;
        std::vector<double> option_values(num_nodes);
        
        for (std::size_t i = 0; i < num_nodes; ++i) {
            const int j = static_cast<int>(i) - static_cast<int>(params_.time_steps);
            const double S = S0 * std::exp(j * dx);
            
            if (option.type == OptionType::CALL) {
                option_values[i] = std::max(S - K, 0.0);
            } else {
                option_values[i] = std::max(K - S, 0.0);
            }
        }
        
        for (int step = static_cast<int>(params_.time_steps) - 1; step >= 0; --step) {
            const std::size_t nodes_at_step = 2 * step + 1;
            
            for (std::size_t i = 0; i < nodes_at_step; ++i) {
                const int j = static_cast<int>(i) - step;
                const double S = S0 * std::exp(j * dx);
                
                const double continuation_value = discount * (
                    pu * option_values[i + 2] + 
                    pm * option_values[i + 1] + 
                    pd * option_values[i]
                );
                
                double exercise_value;
                if (option.type == OptionType::CALL) {
                    exercise_value = std::max(S - K, 0.0);
                } else {
                    exercise_value = std::max(K - S, 0.0);
                }
                
                option_values[i] = std::max(continuation_value, exercise_value);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        BinomialResult result;
        result.option_price = option_values[0];
        result.tree_depth = params_.time_steps;
        result.computation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        return result;
    }

private:
    TrinomialParameters params_;
};

}