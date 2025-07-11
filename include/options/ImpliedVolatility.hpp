#pragma once

#include "../types/Option.hpp"
#include "../types/Market.hpp"
#include "../math/Optimization.hpp"
#include "BlackScholes.hpp"
#include <functional>
#include <cmath>
#include <limits>

namespace options {

class ImpliedVolatilitySolver {
public:
    struct SolverConfiguration {
        double tolerance = 1e-8;
        std::size_t max_iterations = 100;
        double min_volatility = 1e-6;
        double max_volatility = 5.0;
        double initial_guess = 0.2;
        bool use_vega_scaling = true;
        bool use_numerical_stability = true;
    };

    explicit ImpliedVolatilitySolver(const SolverConfiguration& config = SolverConfiguration{})
        : config_(config) {}

    double solve_newton_raphson(
        const OptionSpec& option,
        const MarketData& market,
        double market_price) const {
        
        if (market_price <= 0.0 || option.time_to_expiry <= 0.0) {
            return 0.0;
        }
        
        const double intrinsic_value = calculate_intrinsic_value(option, market);
        if (market_price <= intrinsic_value + 1e-10) {
            return config_.min_volatility;
        }
        
        double vol_guess = get_initial_guess(option, market, market_price);
        
        for (std::size_t iter = 0; iter < config_.max_iterations; ++iter) {
            MarketData temp_market = market;
            temp_market.volatility = vol_guess;
            
            const auto result = BlackScholesPricer::price_european_option(option, temp_market);
            const double price_diff = result.option_price - market_price;
            const double vega = result.greeks.vega * 100.0;
            
            if (std::abs(price_diff) < config_.tolerance) {
                return vol_guess;
            }
            
            if (std::abs(vega) < std::numeric_limits<double>::epsilon()) {
                break;
            }
            
            double step = price_diff / vega;
            
            if (config_.use_numerical_stability) {
                step = apply_numerical_stability(vol_guess, step, iter);
            }
            
            const double new_vol = vol_guess - step;
            const double clamped_vol = std::clamp(new_vol, config_.min_volatility, config_.max_volatility);
            
            if (std::abs(clamped_vol - vol_guess) < config_.tolerance) {
                return clamped_vol;
            }
            
            vol_guess = clamped_vol;
        }
        
        return vol_guess;
    }

    double solve_brent_method(
        const OptionSpec& option,
        const MarketData& market,
        double market_price) const {
        
        if (market_price <= 0.0 || option.time_to_expiry <= 0.0) {
            return 0.0;
        }
        
        auto objective_function = [&](double vol) -> double {
            MarketData temp_market = market;
            temp_market.volatility = vol;
            const auto result = BlackScholesPricer::price_european_option(option, temp_market);
            return result.option_price - market_price;
        };
        
        double vol_low = config_.min_volatility;
        double vol_high = config_.max_volatility;
        
        const double f_low = objective_function(vol_low);
        const double f_high = objective_function(vol_high);
        
        if (f_low * f_high >= 0.0) {
            return solve_newton_raphson(option, market, market_price);
        }
        
        math::BrentSolver::Parameters params;
        params.tolerance = config_.tolerance;
        params.max_iterations = config_.max_iterations;
        
        const auto result = math::BrentSolver::solve(objective_function, vol_low, vol_high, params);
        
        if (result.converged) {
            return result.value;
        }
        
        return solve_newton_raphson(option, market, market_price);
    }

    double solve_bisection_method(
        const OptionSpec& option,
        const MarketData& market,
        double market_price) const {
        
        if (market_price <= 0.0 || option.time_to_expiry <= 0.0) {
            return 0.0;
        }
        
        auto objective_function = [&](double vol) -> double {
            MarketData temp_market = market;
            temp_market.volatility = vol;
            const auto result = BlackScholesPricer::price_european_option(option, temp_market);
            return result.option_price - market_price;
        };
        
        double vol_low = config_.min_volatility;
        double vol_high = config_.max_volatility;
        
        const double f_low = objective_function(vol_low);
        const double f_high = objective_function(vol_high);
        
        if (f_low * f_high >= 0.0) {
            return config_.initial_guess;
        }
        
        math::BisectionSolver::Parameters params;
        params.tolerance = config_.tolerance;
        params.max_iterations = config_.max_iterations;
        
        const auto result = math::BisectionSolver::solve(objective_function, vol_low, vol_high, params);
        
        return result.converged ? result.value : config_.initial_guess;
    }

    double solve_rational_approximation(
        const OptionSpec& option,
        const MarketData& market,
        double market_price) const {
        
        if (market_price <= 0.0 || option.time_to_expiry <= 0.0) {
            return 0.0;
        }
        
        const double S = market.spot_price;
        const double K = option.strike;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double q = market.dividend_yield;
        
        const double F = S * std::exp((r - q) * T);
        const double discount = std::exp(-r * T);
        
        double normalized_price;
        if (option.type == OptionType::CALL) {
            normalized_price = market_price / discount;
        } else {
            normalized_price = (market_price + F - K) / discount;
        }
        
        const double x = std::log(F / K);
        const double alpha = 2.0 * normalized_price / (F + K);
        
        if (alpha <= 0.0 || alpha >= 1.0) {
            return solve_newton_raphson(option, market, market_price);
        }
        
        const double beta = std::log((F + K) / (2.0 * std::sqrt(F * K)));
        
        double vol_approx;
        if (std::abs(x) < 0.1) {
            vol_approx = std::sqrt(2.0 * std::log(1.0 / alpha)) / std::sqrt(T);
        } else {
            const double eta = alpha - 0.5;
            const double zeta = (1.0 / std::sqrt(T)) * std::sqrt(2.0 * std::log((F + K) / (2.0 * F * alpha)));
            vol_approx = zeta + eta * zeta * zeta * zeta / 6.0;
        }
        
        vol_approx = std::clamp(vol_approx, config_.min_volatility, config_.max_volatility);
        
        return refine_approximation(option, market, market_price, vol_approx);
    }

    double solve_adaptive_method(
        const OptionSpec& option,
        const MarketData& market,
        double market_price) const {
        
        const double intrinsic_value = calculate_intrinsic_value(option, market);
        const double time_value = market_price - intrinsic_value;
        
        if (time_value <= config_.tolerance) {
            return config_.min_volatility;
        }
        
        if (option.time_to_expiry > 0.5 && std::abs(std::log(market.spot_price / option.strike)) < 0.1) {
            return solve_rational_approximation(option, market, market_price);
        } else if (option.time_to_expiry < 0.05) {
            return solve_bisection_method(option, market, market_price);
        } else {
            return solve_newton_raphson(option, market, market_price);
        }
    }

    std::vector<double> solve_volatility_surface(
        const std::vector<OptionSpec>& options,
        const std::vector<MarketData>& market_data,
        const std::vector<double>& market_prices) const {
        
        if (options.size() != market_data.size() || options.size() != market_prices.size()) {
            throw std::invalid_argument("Input vectors must have the same size");
        }
        
        std::vector<double> implied_vols(options.size());
        
        for (std::size_t i = 0; i < options.size(); ++i) {
            implied_vols[i] = solve_adaptive_method(options[i], market_data[i], market_prices[i]);
        }
        
        return implied_vols;
    }

    double calculate_vega_weighted_volatility(
        const std::vector<OptionSpec>& options,
        const std::vector<MarketData>& market_data,
        const std::vector<double>& market_prices) const {
        
        auto implied_vols = solve_volatility_surface(options, market_data, market_prices);
        
        double total_vega = 0.0;
        double weighted_vol = 0.0;
        
        for (std::size_t i = 0; i < options.size(); ++i) {
            MarketData temp_market = market_data[i];
            temp_market.volatility = implied_vols[i];
            
            const auto result = BlackScholesPricer::price_european_option(options[i], temp_market);
            const double vega = result.greeks.vega * 100.0;
            
            total_vega += vega;
            weighted_vol += vega * implied_vols[i];
        }
        
        return total_vega > 0.0 ? weighted_vol / total_vega : 0.0;
    }

private:
    SolverConfiguration config_;

    double calculate_intrinsic_value(const OptionSpec& option, const MarketData& market) const {
        if (option.type == OptionType::CALL) {
            return std::max(market.spot_price - option.strike, 0.0);
        } else {
            return std::max(option.strike - market.spot_price, 0.0);
        }
    }

    double get_initial_guess(
        const OptionSpec& option,
        const MarketData& market,
        double market_price) const {
        
        if (config_.initial_guess > 0.0) {
            return config_.initial_guess;
        }
        
        const double S = market.spot_price;
        const double K = option.strike;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        
        const double moneyness = S / K;
        const double forward_moneyness = moneyness * std::exp(r * T);
        
        double base_vol = 0.2;
        
        if (std::abs(std::log(forward_moneyness)) > 0.1) {
            base_vol += 0.1 * std::abs(std::log(forward_moneyness));
        }
        
        if (T < 0.1) {
            base_vol *= 1.5;
        }
        
        return std::clamp(base_vol, config_.min_volatility, config_.max_volatility);
    }

    double apply_numerical_stability(double current_vol, double step, std::size_t iteration) const {
        const double max_step = 0.5;
        const double damping_factor = 1.0 / (1.0 + iteration * 0.1);
        
        double adjusted_step = step * damping_factor;
        
        if (std::abs(adjusted_step) > max_step) {
            adjusted_step = std::copysign(max_step, adjusted_step);
        }
        
        const double new_vol = current_vol - adjusted_step;
        if (new_vol <= config_.min_volatility || new_vol >= config_.max_volatility) {
            adjusted_step *= 0.5;
        }
        
        return adjusted_step;
    }

    double refine_approximation(
        const OptionSpec& option,
        const MarketData& market,
        double market_price,
        double initial_vol) const {
        
        SolverConfiguration refined_config = config_;
        refined_config.max_iterations = 10;
        refined_config.initial_guess = initial_vol;
        
        ImpliedVolatilitySolver refined_solver(refined_config);
        return refined_solver.solve_newton_raphson(option, market, market_price);
    }
};

class VolatilitySurfaceCalibrator {
public:
    struct CalibrationParameters {
        double smoothing_factor = 0.01;
        std::size_t max_iterations = 1000;
        double convergence_tolerance = 1e-6;
        bool enforce_arbitrage_bounds = true;
    };

    explicit VolatilitySurfaceCalibrator(const CalibrationParameters& params = CalibrationParameters{})
        : params_(params) {}

    VolatilitySurface calibrate_surface(
        const std::vector<OptionSpec>& options,
        const std::vector<MarketData>& market_data,
        const std::vector<double>& market_prices) const {
        
        ImpliedVolatilitySolver solver;
        auto implied_vols = solver.solve_volatility_surface(options, market_data, market_prices);
        
        VolatilitySurface surface;
        
        std::vector<double> unique_strikes;
        std::vector<double> unique_expiries;
        
        for (const auto& option : options) {
            if (std::find(unique_strikes.begin(), unique_strikes.end(), option.strike) == unique_strikes.end()) {
                unique_strikes.push_back(option.strike);
            }
            if (std::find(unique_expiries.begin(), unique_expiries.end(), option.time_to_expiry) == unique_expiries.end()) {
                unique_expiries.push_back(option.time_to_expiry);
            }
        }
        
        std::sort(unique_strikes.begin(), unique_strikes.end());
        std::sort(unique_expiries.begin(), unique_expiries.end());
        
        surface.num_strikes = std::min(unique_strikes.size(), surface.MAX_STRIKES);
        surface.num_expiries = std::min(unique_expiries.size(), surface.MAX_EXPIRIES);
        
        for (std::size_t i = 0; i < surface.num_strikes; ++i) {
            surface.strikes[i] = unique_strikes[i];
        }
        
        for (std::size_t i = 0; i < surface.num_expiries; ++i) {
            surface.expiries[i] = unique_expiries[i];
        }
        
        for (std::size_t i = 0; i < options.size(); ++i) {
            surface.add_point(options[i].strike, options[i].time_to_expiry, implied_vols[i]);
        }
        
        if (params_.enforce_arbitrage_bounds) {
            enforce_arbitrage_constraints(surface);
        }
        
        return surface;
    }

private:
    CalibrationParameters params_;

    void enforce_arbitrage_constraints(VolatilitySurface& surface) const {
        for (std::size_t t = 0; t < surface.num_expiries; ++t) {
            for (std::size_t k = 1; k < surface.num_strikes - 1; ++k) {
                const double vol_left = surface.volatilities[t][k - 1];
                const double vol_center = surface.volatilities[t][k];
                const double vol_right = surface.volatilities[t][k + 1];
                
                const double smoothed_vol = (vol_left + 2.0 * vol_center + vol_right) / 4.0;
                surface.volatilities[t][k] = smoothed_vol;
            }
        }
    }
};

}