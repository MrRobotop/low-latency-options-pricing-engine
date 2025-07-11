#pragma once

#include <chrono>
#include <string>
#include <memory>

namespace options {

enum class OptionType {
    CALL,
    PUT
};

enum class ExerciseStyle {
    EUROPEAN,
    AMERICAN,
    BERMUDAN
};

enum class BarrierType {
    NONE,
    UP_AND_OUT,
    UP_AND_IN,
    DOWN_AND_OUT,
    DOWN_AND_IN
};

struct OptionSpec {
    OptionType type;
    ExerciseStyle exercise_style;
    double strike;
    double time_to_expiry;
    BarrierType barrier_type = BarrierType::NONE;
    double barrier_level = 0.0;
    std::string underlying_symbol;
    
    OptionSpec(OptionType t, ExerciseStyle style, double K, double T, 
               const std::string& symbol = "")
        : type(t), exercise_style(style), strike(K), time_to_expiry(T), 
          underlying_symbol(symbol) {}
};

struct OptionContract {
    OptionSpec spec;
    std::size_t contract_id;
    std::chrono::high_resolution_clock::time_point creation_time;
    
    OptionContract(const OptionSpec& s, std::size_t id)
        : spec(s), contract_id(id), 
          creation_time(std::chrono::high_resolution_clock::now()) {}
};

template<typename T>
struct alignas(64) OptionBatch {
    static constexpr std::size_t BATCH_SIZE = 64;
    
    T strike[BATCH_SIZE];
    T time_to_expiry[BATCH_SIZE];
    T spot_price[BATCH_SIZE];
    T volatility[BATCH_SIZE];
    T risk_free_rate[BATCH_SIZE];
    T dividend_yield[BATCH_SIZE];
    OptionType option_type[BATCH_SIZE];
    std::size_t count = 0;
    
    void clear() { count = 0; }
    bool is_full() const { return count >= BATCH_SIZE; }
    bool empty() const { return count == 0; }
};

}