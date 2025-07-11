#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "options/BlackScholes.hpp"
#include "options/PricingEngine.hpp"
#include "options/Greeks.hpp"
#include "options/ImpliedVolatility.hpp"
#include "math/NormalDistribution.hpp"
#include <cmath>

using namespace options;
using namespace options::math;

class BlackScholesTest : public ::testing::Test {
protected:
    void SetUp() override {
        option_call_ = OptionSpec(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "TEST");
        option_put_ = OptionSpec(OptionType::PUT, ExerciseStyle::EUROPEAN, 100.0, 0.25, "TEST");
        market_ = MarketData(100.0, 0.20, 0.05, 0.02);
        tolerance_ = 1e-6;
    }

    OptionSpec option_call_;
    OptionSpec option_put_;
    MarketData market_;
    double tolerance_;
};

TEST_F(BlackScholesTest, CallPutParity) {
    auto call_result = BlackScholesPricer::price_european_option(option_call_, market_);
    auto put_result = BlackScholesPricer::price_european_option(option_put_, market_);
    
    const double S = market_.spot_price;
    const double K = option_call_.strike;
    const double T = option_call_.time_to_expiry;
    const double r = market_.risk_free_rate;
    const double q = market_.dividend_yield;
    
    const double forward = S * std::exp(-q * T);
    const double present_value_strike = K * std::exp(-r * T);
    const double put_call_parity_diff = call_result.option_price - put_result.option_price - (forward - present_value_strike);
    
    EXPECT_NEAR(put_call_parity_diff, 0.0, tolerance_);
}

TEST_F(BlackScholesTest, ATMCallPrice) {
    market_.spot_price = 100.0;
    option_call_.strike = 100.0;
    
    auto result = BlackScholesPricer::price_european_option(option_call_, market_);
    
    const double expected_price = 4.1304;
    EXPECT_NEAR(result.option_price, expected_price, 0.01);
}

TEST_F(BlackScholesTest, DeltaRange) {
    auto call_result = BlackScholesPricer::price_european_option(option_call_, market_);
    auto put_result = BlackScholesPricer::price_european_option(option_put_, market_);
    
    EXPECT_GE(call_result.greeks.delta, 0.0);
    EXPECT_LE(call_result.greeks.delta, 1.0);
    EXPECT_GE(put_result.greeks.delta, -1.0);
    EXPECT_LE(put_result.greeks.delta, 0.0);
}

TEST_F(BlackScholesTest, GammaPositive) {
    auto call_result = BlackScholesPricer::price_european_option(option_call_, market_);
    auto put_result = BlackScholesPricer::price_european_option(option_put_, market_);
    
    EXPECT_GT(call_result.greeks.gamma, 0.0);
    EXPECT_GT(put_result.greeks.gamma, 0.0);
    EXPECT_NEAR(call_result.greeks.gamma, put_result.greeks.gamma, tolerance_);
}

TEST_F(BlackScholesTest, VegaPositive) {
    auto call_result = BlackScholesPricer::price_european_option(option_call_, market_);
    auto put_result = BlackScholesPricer::price_european_option(option_put_, market_);
    
    EXPECT_GT(call_result.greeks.vega, 0.0);
    EXPECT_GT(put_result.greeks.vega, 0.0);
    EXPECT_NEAR(call_result.greeks.vega, put_result.greeks.vega, tolerance_);
}

TEST_F(BlackScholesTest, CallThetaNegative) {
    auto call_result = BlackScholesPricer::price_european_option(option_call_, market_);
    EXPECT_LT(call_result.greeks.theta, 0.0);
}

class NormalDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        tolerance_ = 1e-10;
    }
    
    double tolerance_;
};

TEST_F(NormalDistributionTest, StandardNormalCDF) {
    EXPECT_NEAR(NormalDistribution::cdf(0.0), 0.5, tolerance_);
    EXPECT_NEAR(NormalDistribution::cdf(1.96), 0.975, 1e-3);
    EXPECT_NEAR(NormalDistribution::cdf(-1.96), 0.025, 1e-3);
    EXPECT_NEAR(NormalDistribution::cdf(3.0), 0.9987, 1e-3);
}

TEST_F(NormalDistributionTest, StandardNormalPDF) {
    EXPECT_NEAR(NormalDistribution::pdf(0.0), NormalDistribution::INV_SQRT_2_PI, tolerance_);
    EXPECT_GT(NormalDistribution::pdf(0.0), NormalDistribution::pdf(1.0));
    EXPECT_GT(NormalDistribution::pdf(0.0), NormalDistribution::pdf(-1.0));
}

TEST_F(NormalDistributionTest, D1D2Calculation) {
    const double S = 100.0, K = 105.0, T = 0.25, r = 0.05, vol = 0.20, q = 0.02;
    
    const double d1 = NormalDistribution::d1(S, K, T, r, vol, q);
    const double d2 = NormalDistribution::d2(S, K, T, r, vol, q);
    
    EXPECT_NEAR(d2, d1 - vol * std::sqrt(T), tolerance_);
}

class GreeksTest : public ::testing::Test {
protected:
    void SetUp() override {
        option_ = OptionSpec(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "GREEKS");
        market_ = MarketData(105.0, 0.20, 0.05, 0.02);
        tolerance_ = 1e-4;
    }

    OptionSpec option_;
    MarketData market_;
    double tolerance_;
};

TEST_F(GreeksTest, AnalyticalVsNumerical) {
    auto analytical_greeks = GreeksCalculator::calculate_analytical_greeks(option_, market_);
    
    auto pricing_func = [](const OptionSpec& opt, const MarketData& mkt) -> double {
        return BlackScholesPricer::price_european_option(opt, mkt).option_price;
    };
    
    auto numerical_greeks = GreeksCalculator::calculate_numerical_greeks(option_, market_, pricing_func);
    
    EXPECT_NEAR(analytical_greeks.delta, numerical_greeks.delta, tolerance_);
    EXPECT_NEAR(analytical_greeks.gamma, numerical_greeks.gamma, tolerance_);
    EXPECT_NEAR(analytical_greeks.vega, numerical_greeks.vega, tolerance_);
    EXPECT_NEAR(analytical_greeks.theta, numerical_greeks.theta, tolerance_);
    EXPECT_NEAR(analytical_greeks.rho, numerical_greeks.rho, tolerance_);
}

TEST_F(GreeksTest, DeltaGammaRelationship) {
    const double bump = 1.0;
    
    MarketData market_up = market_;
    MarketData market_down = market_;
    market_up.spot_price += bump;
    market_down.spot_price -= bump;
    
    auto greeks_up = GreeksCalculator::calculate_analytical_greeks(option_, market_up);
    auto greeks_down = GreeksCalculator::calculate_analytical_greeks(option_, market_down);
    auto greeks_base = GreeksCalculator::calculate_analytical_greeks(option_, market_);
    
    const double finite_diff_gamma = (greeks_up.delta - greeks_down.delta) / (2.0 * bump);
    
    EXPECT_NEAR(greeks_base.gamma, finite_diff_gamma, 1e-3);
}

class ImpliedVolatilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        option_ = OptionSpec(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "IV");
        market_ = MarketData(105.0, 0.25, 0.05, 0.02);
        tolerance_ = 1e-6;
    }

    OptionSpec option_;
    MarketData market_;
    double tolerance_;
};

TEST_F(ImpliedVolatilityTest, RecoverKnownVolatility) {
    const double known_vol = 0.25;
    market_.volatility = known_vol;
    
    auto theoretical_result = BlackScholesPricer::price_european_option(option_, market_);
    const double market_price = theoretical_result.option_price;
    
    ImpliedVolatilitySolver solver;
    const double implied_vol = solver.solve_newton_raphson(option_, market_, market_price);
    
    EXPECT_NEAR(implied_vol, known_vol, tolerance_);
}

TEST_F(ImpliedVolatilityTest, ConsistencyAcrossMethods) {
    auto theoretical_result = BlackScholesPricer::price_european_option(option_, market_);
    const double market_price = theoretical_result.option_price * 1.02;
    
    ImpliedVolatilitySolver solver;
    
    const double iv_newton = solver.solve_newton_raphson(option_, market_, market_price);
    const double iv_brent = solver.solve_brent_method(option_, market_, market_price);
    const double iv_bisection = solver.solve_bisection_method(option_, market_, market_price);
    
    EXPECT_NEAR(iv_newton, iv_brent, 1e-4);
    EXPECT_NEAR(iv_newton, iv_bisection, 1e-3);
}

class PricingEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.enable_caching = true;
        config_.enable_vectorization = true;
        config_.enable_multithreading = true;
        engine_ = std::make_unique<PricingEngine>(config_);
        
        option_ = OptionSpec(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "ENGINE");
        market_ = MarketData(105.0, 0.20, 0.05, 0.02);
    }

    PricingEngine::Configuration config_;
    std::unique_ptr<PricingEngine> engine_;
    OptionSpec option_;
    MarketData market_;
};

TEST_F(PricingEngineTest, SingleOptionPricing) {
    auto result = engine_->price_option(option_, market_);
    
    EXPECT_GT(result.option_price, 0.0);
    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.computation_time.count(), 0);
    EXPECT_GT(result.greeks.delta, 0.0);
    EXPECT_GT(result.greeks.gamma, 0.0);
    EXPECT_GT(result.greeks.vega, 0.0);
}

TEST_F(PricingEngineTest, PortfolioPricingConsistency) {
    constexpr std::size_t portfolio_size = 100;
    
    std::vector<OptionSpec> options;
    std::vector<MarketData> market_data;
    
    for (std::size_t i = 0; i < portfolio_size; ++i) {
        const double strike = 95.0 + i * 0.1;
        const OptionType type = (i % 2 == 0) ? OptionType::CALL : OptionType::PUT;
        
        options.emplace_back(type, ExerciseStyle::EUROPEAN, strike, 0.25, "PORT");
        market_data.emplace_back(100.0, 0.20, 0.05, 0.02);
    }
    
    auto portfolio_results = engine_->price_portfolio(options, market_data);
    
    EXPECT_EQ(portfolio_results.size(), portfolio_size);
    
    for (std::size_t i = 0; i < portfolio_size; ++i) {
        auto single_result = engine_->price_option(options[i], market_data[i]);
        EXPECT_NEAR(portfolio_results[i].option_price, single_result.option_price, 1e-10);
    }
}

TEST_F(PricingEngineTest, CachingFunctionality) {
    engine_->clear_cache();
    EXPECT_EQ(engine_->get_cache_size(), 0);
    
    auto result1 = engine_->price_option(option_, market_);
    EXPECT_EQ(engine_->get_cache_size(), 1);
    
    auto result2 = engine_->price_option(option_, market_);
    EXPECT_EQ(engine_->get_cache_size(), 1);
    
    EXPECT_NEAR(result1.option_price, result2.option_price, 1e-15);
    EXPECT_LT(result2.computation_time.count(), result1.computation_time.count());
}

TEST_F(PricingEngineTest, PerformanceMetrics) {
    engine_->reset_performance_metrics();
    
    for (int i = 0; i < 10; ++i) {
        engine_->price_option(option_, market_);
    }
    
    auto metrics = engine_->get_performance_metrics();
    
    EXPECT_EQ(metrics.total_options_priced, 10);
    EXPECT_GT(metrics.avg_pricing_time.count(), 0);
    EXPECT_GT(metrics.throughput_per_second, 0.0);
    EXPECT_LE(metrics.min_pricing_time, metrics.avg_pricing_time);
    EXPECT_GE(metrics.max_pricing_time, metrics.avg_pricing_time);
}

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        option_ = OptionSpec(OptionType::CALL, ExerciseStyle::EUROPEAN, 100.0, 0.25, "PERF");
        market_ = MarketData(105.0, 0.20, 0.05, 0.02);
    }

    OptionSpec option_;
    MarketData market_;
};

TEST_F(PerformanceTest, SubMicrosecondPricing) {
    constexpr std::size_t num_iterations = 1000;
    constexpr auto max_allowed_time = std::chrono::nanoseconds(1000);
    
    std::vector<std::chrono::nanoseconds> times;
    times.reserve(num_iterations);
    
    for (std::size_t i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = BlackScholesPricer::price_european_option(option_, market_);
        auto end = std::chrono::high_resolution_clock::now();
        
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start));
        EXPECT_GT(result.option_price, 0.0);
    }
    
    const auto avg_time = std::accumulate(times.begin(), times.end(), std::chrono::nanoseconds(0)) / times.size();
    const auto min_time = *std::min_element(times.begin(), times.end());
    
    EXPECT_LT(min_time, max_allowed_time) << "Minimum pricing time exceeds 1 microsecond";
    EXPECT_LT(avg_time, max_allowed_time * 2) << "Average pricing time exceeds 2 microseconds";
}

TEST_F(PerformanceTest, HighThroughputPortfolio) {
    constexpr std::size_t portfolio_size = 10000;
    constexpr auto max_allowed_total_time = std::chrono::milliseconds(100);
    
    std::vector<OptionSpec> options;
    std::vector<MarketData> market_data;
    
    options.reserve(portfolio_size);
    market_data.reserve(portfolio_size);
    
    for (std::size_t i = 0; i < portfolio_size; ++i) {
        const double strike = 90.0 + (i % 100) * 0.2;
        const OptionType type = (i % 2 == 0) ? OptionType::CALL : OptionType::PUT;
        
        options.emplace_back(type, ExerciseStyle::EUROPEAN, strike, 0.25, "PERF_PORT");
        market_data.emplace_back(100.0, 0.20, 0.05, 0.02);
    }
    
    PricingEngine::Configuration config;
    config.enable_vectorization = true;
    config.enable_multithreading = true;
    
    PricingEngine engine(config);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = engine.price_portfolio(options, market_data);
    auto end = std::chrono::high_resolution_clock::now();
    
    const auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const auto avg_time_per_option = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) / portfolio_size;
    
    EXPECT_EQ(results.size(), portfolio_size);
    EXPECT_LT(total_time, max_allowed_total_time) << "Portfolio pricing too slow";
    EXPECT_LT(avg_time_per_option.count(), 10000) << "Average per-option time exceeds 10 microseconds";
    
    const double throughput = static_cast<double>(portfolio_size) * 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    EXPECT_GT(throughput, 100000.0) << "Throughput below 100K options/second";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}