#pragma once

#include <chrono>
#include <string>
#include <atomic>
#include <array>

namespace options {

struct alignas(64) MarketData {
    double spot_price;
    double volatility;
    double risk_free_rate;
    double dividend_yield;
    std::chrono::high_resolution_clock::time_point timestamp;
    std::atomic<std::uint64_t> sequence_number{0};
    
    MarketData() = default;
    
    MarketData(double S, double vol, double r, double q = 0.0)
        : spot_price(S), volatility(vol), risk_free_rate(r), dividend_yield(q),
          timestamp(std::chrono::high_resolution_clock::now()) {}
          
    void update(double S, double vol, double r, double q = 0.0) {
        spot_price = S;
        volatility = vol;
        risk_free_rate = r;
        dividend_yield = q;
        timestamp = std::chrono::high_resolution_clock::now();
        sequence_number.fetch_add(1, std::memory_order_relaxed);
    }
    
    bool is_valid() const {
        return spot_price > 0.0 && volatility > 0.0 && risk_free_rate >= 0.0;
    }
};

struct VolatilitySurface {
    static constexpr std::size_t MAX_STRIKES = 50;
    static constexpr std::size_t MAX_EXPIRIES = 20;
    
    std::array<double, MAX_STRIKES> strikes;
    std::array<double, MAX_EXPIRIES> expiries;
    std::array<std::array<double, MAX_STRIKES>, MAX_EXPIRIES> volatilities;
    
    std::size_t num_strikes = 0;
    std::size_t num_expiries = 0;
    
    double interpolate(double strike, double expiry) const;
    void add_point(double strike, double expiry, double vol);
};

struct RealTimeMarketData {
    MarketData current;
    MarketData previous;
    double price_change;
    double vol_change;
    std::chrono::nanoseconds update_latency;
    
    void update_market(const MarketData& new_data) {
        previous = current;
        current = new_data;
        price_change = current.spot_price - previous.spot_price;
        vol_change = current.volatility - previous.volatility;
        update_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(
            current.timestamp - previous.timestamp);
    }
};

template<typename T>
struct alignas(64) MarketDataBatch {
    static constexpr std::size_t BATCH_SIZE = 64;
    
    T spot_price[BATCH_SIZE];
    T volatility[BATCH_SIZE];
    T risk_free_rate[BATCH_SIZE];
    T dividend_yield[BATCH_SIZE];
    std::size_t count = 0;
    
    void clear() { count = 0; }
    bool is_full() const { return count >= BATCH_SIZE; }
    bool empty() const { return count == 0; }
};

}