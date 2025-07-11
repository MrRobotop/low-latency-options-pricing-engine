#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>

namespace options::math {

template<typename T>
class FastStatistics {
public:
    static T mean(const std::vector<T>& data) {
        if (data.empty()) return T{0};
        return std::accumulate(data.begin(), data.end(), T{0}) / data.size();
    }

    static T variance(const std::vector<T>& data, bool sample = true) {
        if (data.size() <= 1) return T{0};
        
        const T mu = mean(data);
        T sum_sq_diff = std::accumulate(data.begin(), data.end(), T{0},
            [mu](T acc, T val) { 
                const T diff = val - mu;
                return acc + diff * diff; 
            });
        
        const T denominator = sample ? static_cast<T>(data.size() - 1) : static_cast<T>(data.size());
        return sum_sq_diff / denominator;
    }

    static T standard_deviation(const std::vector<T>& data, bool sample = true) {
        return std::sqrt(variance(data, sample));
    }

    static T skewness(const std::vector<T>& data) {
        if (data.size() < 3) return T{0};
        
        const T mu = mean(data);
        const T sigma = standard_deviation(data, false);
        
        if (sigma == T{0}) return T{0};
        
        T sum_cubed = std::accumulate(data.begin(), data.end(), T{0},
            [mu, sigma](T acc, T val) {
                const T standardized = (val - mu) / sigma;
                return acc + standardized * standardized * standardized;
            });
        
        return sum_cubed / data.size();
    }

    static T kurtosis(const std::vector<T>& data, bool excess = true) {
        if (data.size() < 4) return T{0};
        
        const T mu = mean(data);
        const T sigma = standard_deviation(data, false);
        
        if (sigma == T{0}) return T{0};
        
        T sum_fourth = std::accumulate(data.begin(), data.end(), T{0},
            [mu, sigma](T acc, T val) {
                const T standardized = (val - mu) / sigma;
                const T squared = standardized * standardized;
                return acc + squared * squared;
            });
        
        T kurt = sum_fourth / data.size();
        return excess ? kurt - T{3} : kurt;
    }

    static T percentile(std::vector<T> data, T p) {
        if (data.empty()) return T{0};
        if (p <= T{0}) return *std::min_element(data.begin(), data.end());
        if (p >= T{1}) return *std::max_element(data.begin(), data.end());
        
        std::sort(data.begin(), data.end());
        
        const T index = p * (data.size() - 1);
        const std::size_t lower_index = static_cast<std::size_t>(std::floor(index));
        const std::size_t upper_index = static_cast<std::size_t>(std::ceil(index));
        
        if (lower_index == upper_index) {
            return data[lower_index];
        }
        
        const T weight = index - lower_index;
        return data[lower_index] * (T{1} - weight) + data[upper_index] * weight;
    }

    static T median(std::vector<T> data) {
        return percentile(std::move(data), T{0.5});
    }

    static std::pair<T, T> confidence_interval(const std::vector<T>& data, T confidence_level = T{0.95}) {
        if (data.empty()) return {T{0}, T{0}};
        
        const T alpha = (T{1} - confidence_level) / T{2};
        const T mu = mean(data);
        const T sigma = standard_deviation(data);
        const T sqrt_n = std::sqrt(static_cast<T>(data.size()));
        
        const T t_critical = t_distribution_critical_value(data.size() - 1, T{1} - alpha);
        const T margin_error = t_critical * sigma / sqrt_n;
        
        return {mu - margin_error, mu + margin_error};
    }

    static T correlation(const std::vector<T>& x, const std::vector<T>& y) {
        if (x.size() != y.size() || x.size() < 2) return T{0};
        
        const T mean_x = mean(x);
        const T mean_y = mean(y);
        
        T numerator = T{0};
        T sum_sq_x = T{0};
        T sum_sq_y = T{0};
        
        for (std::size_t i = 0; i < x.size(); ++i) {
            const T diff_x = x[i] - mean_x;
            const T diff_y = y[i] - mean_y;
            
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }
        
        const T denominator = std::sqrt(sum_sq_x * sum_sq_y);
        return (denominator != T{0}) ? numerator / denominator : T{0};
    }

private:
    static T t_distribution_critical_value(std::size_t degrees_freedom, T probability) {
        if (degrees_freedom >= 30) {
            return normal_distribution_critical_value(probability);
        }
        
        static const std::vector<std::vector<T>> t_table = {
            {12.706, 4.303, 3.182, 2.776, 2.571},  // df=1
            {4.303, 3.182, 2.920, 2.571, 2.447},   // df=2
            {3.182, 2.776, 2.353, 2.132, 2.015}    // df=3 (simplified)
        };
        
        if (probability >= T{0.95} && degrees_freedom <= 3) {
            return t_table[degrees_freedom - 1][0];
        }
        
        return normal_distribution_critical_value(probability);
    }

    static T normal_distribution_critical_value(T probability) {
        if (probability >= T{0.975}) return T{1.96};
        if (probability >= T{0.95}) return T{1.645};
        if (probability >= T{0.90}) return T{1.282};
        return T{1.0};
    }
};

class RandomNumberGenerator {
private:
    std::mt19937_64 generator_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;

public:
    explicit RandomNumberGenerator(std::uint64_t seed = std::random_device{}())
        : generator_(seed), normal_dist_(0.0, 1.0), uniform_dist_(0.0, 1.0) {}

    double normal() {
        return normal_dist_(generator_);
    }

    double uniform() {
        return uniform_dist_(generator_);
    }

    void seed(std::uint64_t new_seed) {
        generator_.seed(new_seed);
    }

    template<std::size_t N>
    void fill_normal(std::array<double, N>& arr) {
        for (auto& val : arr) {
            val = normal();
        }
    }

    void fill_normal(std::vector<double>& vec) {
        for (auto& val : vec) {
            val = normal();
        }
    }
};

}