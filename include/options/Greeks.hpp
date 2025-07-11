#pragma once

#include "../types/Option.hpp"
#include "../types/Market.hpp"
#include "../types/Results.hpp"
#include "../math/NormalDistribution.hpp"
#include "../math/Optimization.hpp"
#include <functional>
#include <array>

namespace options {

template<typename T>
class AutomaticDifferentiation {
public:
    struct DualNumber {
        T value;
        T derivative;
        
        DualNumber(T v = T{0}, T d = T{0}) : value(v), derivative(d) {}
        
        DualNumber operator+(const DualNumber& other) const {
            return DualNumber(value + other.value, derivative + other.derivative);
        }
        
        DualNumber operator-(const DualNumber& other) const {
            return DualNumber(value - other.value, derivative - other.derivative);
        }
        
        DualNumber operator*(const DualNumber& other) const {
            return DualNumber(
                value * other.value,
                derivative * other.value + value * other.derivative
            );
        }
        
        DualNumber operator/(const DualNumber& other) const {
            return DualNumber(
                value / other.value,
                (derivative * other.value - value * other.derivative) / (other.value * other.value)
            );
        }
        
        friend DualNumber exp(const DualNumber& x) {
            T exp_val = std::exp(x.value);
            return DualNumber(exp_val, x.derivative * exp_val);
        }
        
        friend DualNumber log(const DualNumber& x) {
            return DualNumber(std::log(x.value), x.derivative / x.value);
        }
        
        friend DualNumber sqrt(const DualNumber& x) {
            T sqrt_val = std::sqrt(x.value);
            return DualNumber(sqrt_val, x.derivative / (T{2} * sqrt_val));
        }
        
        friend DualNumber pow(const DualNumber& base, T exponent) {
            T pow_val = std::pow(base.value, exponent);
            return DualNumber(
                pow_val,
                exponent * std::pow(base.value, exponent - T{1}) * base.derivative
            );
        }
    };
    
    static T derivative(std::function<T(T)> f, T x, T h = T{1e-8}) {
        return math::FiniteDifference<T>::central_difference(f, x, h);
    }
    
    static T second_derivative(std::function<T(T)> f, T x, T h = T{1e-6}) {
        return math::FiniteDifference<T>::second_derivative(f, x, h);
    }
};

class GreeksCalculator {
public:
    static Greeks calculate_analytical_greeks(
        const OptionSpec& option,
        const MarketData& market) noexcept {
        
        const double S = market.spot_price;
        const double K = option.strike;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double vol = market.volatility;
        const double q = market.dividend_yield;
        
        if (T <= 0.0 || vol <= 0.0 || S <= 0.0) {
            return Greeks{};
        }
        
        const double d1 = math::NormalDistribution::d1(S, K, T, r, vol, q);
        const double d2 = math::NormalDistribution::d2(S, K, T, r, vol, q);
        
        const double Nd1 = math::NormalDistribution::cdf(d1);
        const double Nd2 = math::NormalDistribution::cdf(d2);
        const double N_minus_d1 = math::NormalDistribution::cdf(-d1);
        const double N_minus_d2 = math::NormalDistribution::cdf(-d2);
        const double pdf_d1 = math::NormalDistribution::pdf(d1);
        
        const double sqrt_T = std::sqrt(T);
        const double vol_sqrt_T = vol * sqrt_T;
        const double discount_factor = std::exp(-r * T);
        const double dividend_discount = std::exp(-q * T);
        
        Greeks greeks;
        
        if (option.type == OptionType::CALL) {
            greeks.delta = dividend_discount * Nd1;
        } else {
            greeks.delta = -dividend_discount * N_minus_d1;
        }
        
        greeks.gamma = dividend_discount * pdf_d1 / (S * vol_sqrt_T);
        
        const double theta_common = -S * dividend_discount * pdf_d1 * vol / (2.0 * sqrt_T);
        if (option.type == OptionType::CALL) {
            greeks.theta = (theta_common - r * K * discount_factor * Nd2 + 
                           q * S * dividend_discount * Nd1) / 365.0;
            greeks.rho = K * T * discount_factor * Nd2 / 100.0;
        } else {
            greeks.theta = (theta_common + r * K * discount_factor * N_minus_d2 - 
                           q * S * dividend_discount * N_minus_d1) / 365.0;
            greeks.rho = -K * T * discount_factor * N_minus_d2 / 100.0;
        }
        
        greeks.vega = S * dividend_discount * pdf_d1 * sqrt_T / 100.0;
        
        greeks.epsilon = -S * T * dividend_discount * 
                        (option.type == OptionType::CALL ? Nd1 : -N_minus_d1) / 100.0;
        
        return greeks;
    }

    static Greeks calculate_numerical_greeks(
        const OptionSpec& option,
        const MarketData& market,
        std::function<double(const OptionSpec&, const MarketData&)> pricing_function,
        double bump_size = 1e-4) noexcept {
        
        Greeks greeks;
        const double base_price = pricing_function(option, market);
        
        {
            MarketData bumped_market = market;
            bumped_market.spot_price += bump_size;
            const double bumped_price = pricing_function(option, bumped_market);
            greeks.delta = (bumped_price - base_price) / bump_size;
        }
        
        {
            MarketData up_market = market;
            MarketData down_market = market;
            up_market.spot_price += bump_size;
            down_market.spot_price -= bump_size;
            const double up_price = pricing_function(option, up_market);
            const double down_price = pricing_function(option, down_market);
            greeks.gamma = (up_price - 2.0 * base_price + down_price) / (bump_size * bump_size);
        }
        
        {
            OptionSpec bumped_option = option;
            bumped_option.time_to_expiry -= 1.0 / 365.0;
            if (bumped_option.time_to_expiry > 0.0) {
                const double bumped_price = pricing_function(bumped_option, market);
                greeks.theta = bumped_price - base_price;
            }
        }
        
        {
            MarketData bumped_market = market;
            bumped_market.volatility += bump_size;
            const double bumped_price = pricing_function(option, bumped_market);
            greeks.vega = (bumped_price - base_price) / bump_size;
        }
        
        {
            MarketData bumped_market = market;
            bumped_market.risk_free_rate += bump_size;
            const double bumped_price = pricing_function(option, bumped_market);
            greeks.rho = (bumped_price - base_price) / bump_size;
        }
        
        {
            MarketData bumped_market = market;
            bumped_market.dividend_yield += bump_size;
            const double bumped_price = pricing_function(option, bumped_market);
            greeks.epsilon = (bumped_price - base_price) / bump_size;
        }
        
        return greeks;
    }

    static std::array<double, 5> calculate_higher_order_greeks(
        const OptionSpec& option,
        const MarketData& market) noexcept {
        
        const double S = market.spot_price;
        const double K = option.strike;
        const double T = option.time_to_expiry;
        const double r = market.risk_free_rate;
        const double vol = market.volatility;
        const double q = market.dividend_yield;
        
        if (T <= 0.0 || vol <= 0.0 || S <= 0.0) {
            return {0.0, 0.0, 0.0, 0.0, 0.0};
        }
        
        const double d1 = math::NormalDistribution::d1(S, K, T, r, vol, q);
        const double sqrt_T = std::sqrt(T);
        const double vol_sqrt_T = vol * sqrt_T;
        const double dividend_discount = std::exp(-q * T);
        const double pdf_d1 = math::NormalDistribution::pdf(d1);
        
        const double speed = -dividend_discount * pdf_d1 * 
                            (d1 / (vol_sqrt_T) + 1.0) / (S * S * vol_sqrt_T);
        
        const double color = -dividend_discount * pdf_d1 / (2.0 * S * T * vol_sqrt_T) *
                            (2.0 * q * T + 1.0 + 
                             (2.0 * (r - q) * T - d1 * vol_sqrt_T) / (vol_sqrt_T) * d1);
        
        const double volga = S * dividend_discount * pdf_d1 * sqrt_T * d1 * 
                            math::NormalDistribution::d2(S, K, T, r, vol, q) / vol;
        
        const double vanna = -dividend_discount * pdf_d1 * 
                            math::NormalDistribution::d2(S, K, T, r, vol, q) / vol;
        
        const double charm = dividend_discount * pdf_d1 * 
                            (2.0 * (r - q) * T - d1 * vol_sqrt_T) / (2.0 * T * vol_sqrt_T);
        
        return {speed, color, volga, vanna, charm};
    }

    template<typename T>
    static Greeks calculate_autodiff_greeks(
        const OptionSpec& option,
        const MarketData& market) noexcept {
        
        using DualNumber = typename AutomaticDifferentiation<T>::DualNumber;
        
        const auto black_scholes_dual = [&](DualNumber S, DualNumber vol, DualNumber T_dual, 
                                           DualNumber r, DualNumber q) -> DualNumber {
            DualNumber K_dual(static_cast<T>(option.strike), T{0});
            
            DualNumber sqrt_T = sqrt(T_dual);
            DualNumber vol_sqrt_T = vol * sqrt_T;
            
            DualNumber log_S_K = log(S / K_dual);
            DualNumber vol_squared_half = vol * vol * DualNumber(T{0.5}, T{0});
            DualNumber drift = r - q + vol_squared_half;
            
            DualNumber d1 = (log_S_K + drift * T_dual) / vol_sqrt_T;
            DualNumber d2 = d1 - vol_sqrt_T;
            
            T Nd1_val = math::NormalDistribution::cdf(d1.value);
            T Nd2_val = math::NormalDistribution::cdf(d2.value);
            T pdf_d1 = math::NormalDistribution::pdf(d1.value);
            
            DualNumber Nd1(Nd1_val, pdf_d1 * d1.derivative);
            DualNumber Nd2(Nd2_val, pdf_d1 * d2.derivative);
            
            DualNumber discount = exp(DualNumber(T{0}, T{0}) - r * T_dual);
            DualNumber div_discount = exp(DualNumber(T{0}, T{0}) - q * T_dual);
            
            if (option.type == OptionType::CALL) {
                return S * div_discount * Nd1 - K_dual * discount * Nd2;
            } else {
                DualNumber one(T{1}, T{0});
                return K_dual * discount * (one - Nd2) - S * div_discount * (one - Nd1);
            }
        };
        
        Greeks greeks;
        
        {
            DualNumber S(static_cast<T>(market.spot_price), T{1});
            DualNumber vol(static_cast<T>(market.volatility), T{0});
            DualNumber T_dual(static_cast<T>(option.time_to_expiry), T{0});
            DualNumber r(static_cast<T>(market.risk_free_rate), T{0});
            DualNumber q(static_cast<T>(market.dividend_yield), T{0});
            
            auto result = black_scholes_dual(S, vol, T_dual, r, q);
            greeks.delta = static_cast<double>(result.derivative);
        }
        
        {
            DualNumber S(static_cast<T>(market.spot_price), T{0});
            DualNumber vol(static_cast<T>(market.volatility), T{1});
            DualNumber T_dual(static_cast<T>(option.time_to_expiry), T{0});
            DualNumber r(static_cast<T>(market.risk_free_rate), T{0});
            DualNumber q(static_cast<T>(market.dividend_yield), T{0});
            
            auto result = black_scholes_dual(S, vol, T_dual, r, q);
            greeks.vega = static_cast<double>(result.derivative) / 100.0;
        }
        
        {
            DualNumber S(static_cast<T>(market.spot_price), T{0});
            DualNumber vol(static_cast<T>(market.volatility), T{0});
            DualNumber T_dual(static_cast<T>(option.time_to_expiry), T{1});
            DualNumber r(static_cast<T>(market.risk_free_rate), T{0});
            DualNumber q(static_cast<T>(market.dividend_yield), T{0});
            
            auto result = black_scholes_dual(S, vol, T_dual, r, q);
            greeks.theta = -static_cast<double>(result.derivative) / 365.0;
        }
        
        {
            DualNumber S(static_cast<T>(market.spot_price), T{0});
            DualNumber vol(static_cast<T>(market.volatility), T{0});
            DualNumber T_dual(static_cast<T>(option.time_to_expiry), T{0});
            DualNumber r(static_cast<T>(market.risk_free_rate), T{1});
            DualNumber q(static_cast<T>(market.dividend_yield), T{0});
            
            auto result = black_scholes_dual(S, vol, T_dual, r, q);
            greeks.rho = static_cast<double>(result.derivative) / 100.0;
        }
        
        auto gamma_func = [&](double S_val) -> double {
            MarketData temp_market = market;
            temp_market.spot_price = S_val;
            auto temp_greeks = calculate_analytical_greeks(option, temp_market);
            return temp_greeks.delta;
        };
        
        greeks.gamma = AutomaticDifferentiation<double>::derivative(gamma_func, market.spot_price);
        
        return greeks;
    }

    static double calculate_effective_delta(
        const OptionSpec& option,
        const MarketData& market,
        double portfolio_delta_target = 0.0) noexcept {
        
        const auto greeks = calculate_analytical_greeks(option, market);
        return portfolio_delta_target - greeks.delta;
    }

    static double calculate_gamma_scalping_pnl(
        const OptionSpec& option,
        const MarketData& current_market,
        const MarketData& previous_market,
        double position_size = 1.0) noexcept {
        
        const auto greeks = calculate_analytical_greeks(option, previous_market);
        const double price_change = current_market.spot_price - previous_market.spot_price;
        
        return 0.5 * greeks.gamma * position_size * price_change * price_change;
    }
};

}