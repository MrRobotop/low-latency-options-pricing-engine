#pragma once

#include "../types/Option.hpp"
#include "../types/Market.hpp"
#include "../types/Results.hpp"
#include "../math/NormalDistribution.hpp"
#include <cmath>
#include <immintrin.h>

namespace options {

class BlackScholesPricer {
public:
    static PricingResult price_european_option(
        const OptionSpec& option,
        const MarketData& market) noexcept {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const double S = market.spot_price;
        const double K = option.strike;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double vol = market.volatility;
        const double q = market.dividend_yield;
        
        if (T <= 0.0 || vol <= 0.0 || S <= 0.0) {
            return PricingResult{};
        }
        
        const double d1 = math::NormalDistribution::d1(S, K, T, r, vol, q);
        const double d2 = math::NormalDistribution::d2(S, K, T, r, vol, q);
        
        const double Nd1 = math::NormalDistribution::cdf(d1);
        const double Nd2 = math::NormalDistribution::cdf(d2);
        const double N_minus_d1 = math::NormalDistribution::cdf(-d1);
        const double N_minus_d2 = math::NormalDistribution::cdf(-d2);
        
        double option_price;
        Greeks greeks;
        
        const double discount_factor = std::exp(-r * T);
        const double dividend_discount = std::exp(-q * T);
        const double sqrt_T = std::sqrt(T);
        const double vol_sqrt_T = vol * sqrt_T;
        
        if (option.type == OptionType::CALL) {
            option_price = S * dividend_discount * Nd1 - K * discount_factor * Nd2;
            greeks.delta = dividend_discount * Nd1;
        } else {
            option_price = K * discount_factor * N_minus_d2 - S * dividend_discount * N_minus_d1;
            greeks.delta = -dividend_discount * N_minus_d1;
        }
        
        const double pdf_d1 = math::NormalDistribution::pdf(d1);
        greeks.gamma = dividend_discount * pdf_d1 / (S * vol_sqrt_T);
        greeks.theta = calculate_theta(S, K, T, r, vol, q, option.type, pdf_d1, Nd1, Nd2);
        greeks.vega = S * dividend_discount * pdf_d1 * sqrt_T / 100.0;
        greeks.rho = calculate_rho(K, T, discount_factor, Nd2, N_minus_d2, option.type) / 100.0;
        greeks.epsilon = -S * T * dividend_discount * 
                        (option.type == OptionType::CALL ? Nd1 : -N_minus_d1) / 100.0;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto computation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        PricingResult result(option_price, greeks);
        result.computation_time = computation_time;
        result.converged = true;
        result.numerical_error = 0.0;
        
        return result;
    }

    static double calculate_implied_volatility(
        const OptionSpec& option,
        const MarketData& market,
        double market_price,
        double tolerance = 1e-6,
        std::size_t max_iterations = 100) noexcept {
        
        if (market_price <= 0.0 || option.time_to_expiry <= 0.0) {
            return 0.0;
        }
        
        double vol_guess = 0.2;
        const double vol_min = 1e-6;
        const double vol_max = 5.0;
        
        for (std::size_t i = 0; i < max_iterations; ++i) {
            MarketData temp_market = market;
            temp_market.volatility = vol_guess;
            
            const auto result = price_european_option(option, temp_market);
            const double price_diff = result.option_price - market_price;
            
            if (std::abs(price_diff) < tolerance) {
                return vol_guess;
            }
            
            const double vega = result.greeks.vega * 100.0;
            
            if (std::abs(vega) < 1e-10) {
                break;
            }
            
            double new_vol = vol_guess - price_diff / vega;
            new_vol = std::max(vol_min, std::min(vol_max, new_vol));
            
            if (std::abs(new_vol - vol_guess) < tolerance) {
                return new_vol;
            }
            
            vol_guess = new_vol;
        }
        
        return vol_guess;
    }

#ifdef __AVX2__
    static void vectorized_pricing(
        const OptionBatch<double>& options,
        const MarketDataBatch<double>& market,
        BatchResults<double>& results) noexcept {
        
        const std::size_t batch_size = std::min(options.count, market.count);
        const std::size_t simd_count = (batch_size / 4) * 4;
        
        for (std::size_t i = 0; i < simd_count; i += 4) {
            __m256d S = _mm256_load_pd(&market.spot_price[i]);
            __m256d K = _mm256_load_pd(&options.strike[i]);
            __m256d T = _mm256_load_pd(&options.time_to_expiry[i]);
            __m256d r = _mm256_load_pd(&market.risk_free_rate[i]);
            __m256d vol = _mm256_load_pd(&market.volatility[i]);
            __m256d q = _mm256_load_pd(&market.dividend_yield[i]);
            
            __m256d sqrt_T = _mm256_sqrt_pd(T);
            __m256d vol_sqrt_T = _mm256_mul_pd(vol, sqrt_T);
            
            __m256d log_S_K = _mm256_log_pd(_mm256_div_pd(S, K));
            __m256d vol_squared_half = _mm256_mul_pd(vol, _mm256_mul_pd(vol, _mm256_set1_pd(0.5)));
            __m256d drift = _mm256_add_pd(_mm256_sub_pd(r, q), vol_squared_half);
            
            __m256d d1 = _mm256_div_pd(_mm256_add_pd(log_S_K, _mm256_mul_pd(drift, T)), vol_sqrt_T);
            __m256d d2 = _mm256_sub_pd(d1, vol_sqrt_T);
            
            __m256d Nd1, Nd2;
            math::NormalDistribution::vectorized_cdf(d1, &Nd1);
            math::NormalDistribution::vectorized_cdf(d2, &Nd2);
            
            __m256d discount = _mm256_exp_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(r, T)));
            __m256d div_discount = _mm256_exp_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(q, T)));
            
            __m256d call_price = _mm256_sub_pd(
                _mm256_mul_pd(S, _mm256_mul_pd(div_discount, Nd1)),
                _mm256_mul_pd(K, _mm256_mul_pd(discount, Nd2))
            );
            
            _mm256_store_pd(&results.option_price[i], call_price);
            _mm256_store_pd(&results.delta[i], _mm256_mul_pd(div_discount, Nd1));
        }
        
        for (std::size_t i = simd_count; i < batch_size; ++i) {
            OptionSpec spec(OptionType::CALL, ExerciseStyle::EUROPEAN, 
                          options.strike[i], options.time_to_expiry[i]);
            MarketData mkt(market.spot_price[i], market.volatility[i], 
                          market.risk_free_rate[i], market.dividend_yield[i]);
            
            auto result = price_european_option(spec, mkt);
            results.option_price[i] = result.option_price;
            results.delta[i] = result.greeks.delta;
            results.gamma[i] = result.greeks.gamma;
            results.theta[i] = result.greeks.theta;
            results.vega[i] = result.greeks.vega;
            results.rho[i] = result.greeks.rho;
            results.converged[i] = result.converged;
        }
        
        results.count = batch_size;
    }
#endif

private:
    static double calculate_theta(
        double S, double K, double T, double r, double vol, double q,
        OptionType type, double pdf_d1, double Nd1, double Nd2) noexcept {
        
        const double sqrt_T = std::sqrt(T);
        const double discount_factor = std::exp(-r * T);
        const double dividend_discount = std::exp(-q * T);
        
        const double theta_common = -S * dividend_discount * pdf_d1 * vol / (2.0 * sqrt_T);
        
        if (type == OptionType::CALL) {
            return (theta_common - r * K * discount_factor * Nd2 + q * S * dividend_discount * Nd1) / 365.0;
        } else {
            const double N_minus_d1 = math::NormalDistribution::cdf(-math::NormalDistribution::d1(S, K, T, r, vol, q));
            const double N_minus_d2 = math::NormalDistribution::cdf(-math::NormalDistribution::d2(S, K, T, r, vol, q));
            return (theta_common + r * K * discount_factor * N_minus_d2 - q * S * dividend_discount * N_minus_d1) / 365.0;
        }
    }

    static double calculate_rho(
        double K, double T, double discount_factor,
        double Nd2, double N_minus_d2, OptionType type) noexcept {
        
        if (type == OptionType::CALL) {
            return K * T * discount_factor * Nd2;
        } else {
            return -K * T * discount_factor * N_minus_d2;
        }
    }
};

class BlackScholesBarrierPricer {
public:
    static PricingResult price_barrier_option(
        const OptionSpec& option,
        const MarketData& market) noexcept {
        
        if (option.barrier_type == BarrierType::NONE) {
            return BlackScholesPricer::price_european_option(option, market);
        }
        
        const double S = market.spot_price;
        const double K = option.strike;
        const double B = option.barrier_level;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double vol = market.volatility;
        const double q = market.dividend_yield;
        
        const double mu = (r - q - 0.5 * vol * vol) / (vol * vol);
        const double lambda = std::sqrt(mu * mu + 2.0 * r / (vol * vol));
        
        double option_price = 0.0;
        
        switch (option.barrier_type) {
            case BarrierType::DOWN_AND_OUT:
                option_price = price_down_and_out(S, K, B, T, r, vol, q, option.type, mu, lambda);
                break;
            case BarrierType::UP_AND_OUT:
                option_price = price_up_and_out(S, K, B, T, r, vol, q, option.type, mu, lambda);
                break;
            case BarrierType::DOWN_AND_IN:
                option_price = price_down_and_in(S, K, B, T, r, vol, q, option.type, mu, lambda);
                break;
            case BarrierType::UP_AND_IN:
                option_price = price_up_and_in(S, K, B, T, r, vol, q, option.type, mu, lambda);
                break;
            default:
                return PricingResult{};
        }
        
        PricingResult result;
        result.option_price = option_price;
        result.converged = true;
        
        return result;
    }

private:
    static double price_down_and_out(double S, double K, double B, double T, double r, double vol, double q, OptionType type, double mu, double lambda) {
        if (S <= B) return 0.0;
        
        const double d1 = math::NormalDistribution::d1(S, K, T, r, vol, q);
        const double d2 = d1 - vol * std::sqrt(T);
        const double d3 = math::NormalDistribution::d1(S, B, T, r, vol, q);
        const double d4 = d3 - vol * std::sqrt(T);
        
        const double y1 = (std::log(B * B / (S * K)) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        const double y2 = y1 - vol * std::sqrt(T);
        
        const double pow_term = std::pow(B / S, 2.0 * mu);
        
        if (type == OptionType::CALL) {
            if (K >= B) {
                return S * std::exp(-q * T) * math::NormalDistribution::cdf(d1) - K * std::exp(-r * T) * math::NormalDistribution::cdf(d2) -
                       S * std::exp(-q * T) * pow_term * math::NormalDistribution::cdf(y1) + K * std::exp(-r * T) * pow_term * math::NormalDistribution::cdf(y2);
            } else {
                return S * std::exp(-q * T) * (math::NormalDistribution::cdf(d3) - pow_term * math::NormalDistribution::cdf(y1)) -
                       K * std::exp(-r * T) * (math::NormalDistribution::cdf(d4) - pow_term * math::NormalDistribution::cdf(y2));
            }
        } else {
            if (K >= B) {
                return -S * std::exp(-q * T) * pow_term * math::NormalDistribution::cdf(-y1) + K * std::exp(-r * T) * pow_term * math::NormalDistribution::cdf(-y2);
            } else {
                return K * std::exp(-r * T) * math::NormalDistribution::cdf(-d2) - S * std::exp(-q * T) * math::NormalDistribution::cdf(-d1) +
                       S * std::exp(-q * T) * pow_term * math::NormalDistribution::cdf(-y1) - K * std::exp(-r * T) * pow_term * math::NormalDistribution::cdf(-y2);
            }
        }
    }

    static double price_up_and_out(double S, double K, double B, double T, double r, double vol, double q, OptionType type, double mu, double lambda) {
        if (S >= B) return 0.0;
        
        const double d1 = math::NormalDistribution::d1(S, K, T, r, vol, q);
        const double d2 = d1 - vol * std::sqrt(T);
        const double f1 = math::NormalDistribution::d1(S, B, T, r, vol, q);
        const double f2 = f1 - vol * std::sqrt(T);
        
        const double e1 = (std::log(B * B / (S * K)) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        const double e2 = e1 - vol * std::sqrt(T);
        
        const double pow_term = std::pow(B / S, 2.0 * mu);
        
        if (type == OptionType::CALL) {
            if (K <= B) {
                return S * std::exp(-q * T) * (math::NormalDistribution::cdf(f1) - pow_term * math::NormalDistribution::cdf(e1)) -
                       K * std::exp(-r * T) * (math::NormalDistribution::cdf(f2) - pow_term * math::NormalDistribution::cdf(e2));
            } else {
                return S * std::exp(-q * T) * math::NormalDistribution::cdf(d1) - K * std::exp(-r * T) * math::NormalDistribution::cdf(d2) -
                       S * std::exp(-q * T) * (math::NormalDistribution::cdf(f1) - pow_term * math::NormalDistribution::cdf(e1)) +
                       K * std::exp(-r * T) * (math::NormalDistribution::cdf(f2) - pow_term * math::NormalDistribution::cdf(e2));
            }
        } else {
            if (K <= B) {
                return K * std::exp(-r * T) * math::NormalDistribution::cdf(-d2) - S * std::exp(-q * T) * math::NormalDistribution::cdf(-d1) +
                       S * std::exp(-q * T) * (math::NormalDistribution::cdf(-f1) - pow_term * math::NormalDistribution::cdf(-e1)) -
                       K * std::exp(-r * T) * (math::NormalDistribution::cdf(-f2) - pow_term * math::NormalDistribution::cdf(-e2));
            } else {
                return -S * std::exp(-q * T) * pow_term * math::NormalDistribution::cdf(-e1) +
                       K * std::exp(-r * T) * pow_term * math::NormalDistribution::cdf(-e2);
            }
        }
    }

    static double price_down_and_in(double S, double K, double B, double T, double r, double vol, double q, OptionType type, double mu, double lambda) {
        auto vanilla_result = BlackScholesPricer::price_european_option(
            OptionSpec(type, ExerciseStyle::EUROPEAN, K, T), 
            MarketData(S, vol, r, q));
        double down_and_out = price_down_and_out(S, K, B, T, r, vol, q, type, mu, lambda);
        return vanilla_result.option_price - down_and_out;
    }

    static double price_up_and_in(double S, double K, double B, double T, double r, double vol, double q, OptionType type, double mu, double lambda) {
        auto vanilla_result = BlackScholesPricer::price_european_option(
            OptionSpec(type, ExerciseStyle::EUROPEAN, K, T),
            MarketData(S, vol, r, q));
        double up_and_out = price_up_and_out(S, K, B, T, r, vol, q, type, mu, lambda);
        return vanilla_result.option_price - up_and_out;
    }
};

}