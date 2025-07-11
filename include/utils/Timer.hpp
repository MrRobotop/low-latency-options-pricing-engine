#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace options::utils {

class HighResolutionTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::nanoseconds;
    
    HighResolutionTimer() : start_time_(Clock::now()) {}
    
    void start() noexcept {
        start_time_ = Clock::now();
    }
    
    Duration elapsed() const noexcept {
        return std::chrono::duration_cast<Duration>(Clock::now() - start_time_);
    }
    
    double elapsed_seconds() const noexcept {
        return elapsed().count() * 1e-9;
    }
    
    double elapsed_microseconds() const noexcept {
        return elapsed().count() * 1e-3;
    }
    
    void reset() noexcept {
        start_time_ = Clock::now();
    }

private:
    TimePoint start_time_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(Duration& duration_ref) 
        : duration_ref_(duration_ref), timer_() {}
    
    ~ScopedTimer() {
        duration_ref_ = timer_.elapsed();
    }

private:
    Duration& duration_ref_;
    HighResolutionTimer timer_;
};

class PerformanceProfiler {
public:
    struct ProfileData {
        std::atomic<std::uint64_t> call_count{0};
        std::atomic<std::uint64_t> total_time{0};
        std::atomic<std::uint64_t> min_time{UINT64_MAX};
        std::atomic<std::uint64_t> max_time{0};
        
        void update(std::uint64_t duration_ns) {
            call_count.fetch_add(1, std::memory_order_relaxed);
            total_time.fetch_add(duration_ns, std::memory_order_relaxed);
            
            std::uint64_t current_min = min_time.load(std::memory_order_relaxed);
            while (duration_ns < current_min && 
                   !min_time.compare_exchange_weak(current_min, duration_ns, std::memory_order_relaxed)) {}
            
            std::uint64_t current_max = max_time.load(std::memory_order_relaxed);
            while (duration_ns > current_max && 
                   !max_time.compare_exchange_weak(current_max, duration_ns, std::memory_order_relaxed)) {}
        }
        
        double average_time_ns() const {
            std::uint64_t count = call_count.load(std::memory_order_relaxed);
            return count > 0 ? static_cast<double>(total_time.load(std::memory_order_relaxed)) / count : 0.0;
        }
        
        std::uint64_t get_min_time() const {
            std::uint64_t min_val = min_time.load(std::memory_order_relaxed);
            return min_val == UINT64_MAX ? 0 : min_val;
        }
    };
    
    static PerformanceProfiler& instance() {
        static PerformanceProfiler profiler;
        return profiler;
    }
    
    void start_timing(const std::string& name) {
        thread_local std::unordered_map<std::string, HighResolutionTimer> timers;
        timers[name].start();
    }
    
    void end_timing(const std::string& name) {
        thread_local std::unordered_map<std::string, HighResolutionTimer> timers;
        
        auto it = timers.find(name);
        if (it != timers.end()) {
            auto duration = it->second.elapsed();
            profiles_[name].update(duration.count());
        }
    }
    
    ProfileData get_profile(const std::string& name) const {
        auto it = profiles_.find(name);
        return it != profiles_.end() ? it->second : ProfileData{};
    }
    
    void reset_profile(const std::string& name) {
        profiles_[name] = ProfileData{};
    }
    
    void reset_all_profiles() {
        profiles_.clear();
    }
    
    std::vector<std::pair<std::string, ProfileData>> get_all_profiles() const {
        std::vector<std::pair<std::string, ProfileData>> result;
        for (const auto& [name, data] : profiles_) {
            result.emplace_back(name, data);
        }
        return result;
    }

private:
    mutable std::unordered_map<std::string, ProfileData> profiles_;
};

class AutoProfiler {
public:
    explicit AutoProfiler(const std::string& name) : name_(name) {
        PerformanceProfiler::instance().start_timing(name_);
    }
    
    ~AutoProfiler() {
        PerformanceProfiler::instance().end_timing(name_);
    }

private:
    std::string name_;
};

#define PROFILE_SCOPE(name) AutoProfiler _prof(name)
#define PROFILE_FUNCTION() AutoProfiler _prof(__FUNCTION__)

template<typename Func>
auto time_function(Func&& func) {
    HighResolutionTimer timer;
    if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
        func();
        return timer.elapsed();
    } else {
        auto result = func();
        auto duration = timer.elapsed();
        return std::make_pair(std::move(result), duration);
    }
}

class LatencyMeasurement {
public:
    static constexpr std::size_t MAX_SAMPLES = 10000;
    
    void add_sample(Duration latency) {
        if (samples_.size() < MAX_SAMPLES) {
            samples_.push_back(latency.count());
        } else {
            samples_[next_index_] = latency.count();
            next_index_ = (next_index_ + 1) % MAX_SAMPLES;
        }
    }
    
    double get_percentile(double p) const {
        if (samples_.empty()) return 0.0;
        
        auto sorted_samples = samples_;
        std::sort(sorted_samples.begin(), sorted_samples.end());
        
        const std::size_t index = static_cast<std::size_t>(p * (sorted_samples.size() - 1));
        return static_cast<double>(sorted_samples[index]);
    }
    
    double get_average() const {
        if (samples_.empty()) return 0.0;
        
        std::uint64_t sum = 0;
        for (auto sample : samples_) {
            sum += sample;
        }
        return static_cast<double>(sum) / samples_.size();
    }
    
    std::uint64_t get_min() const {
        return samples_.empty() ? 0 : *std::min_element(samples_.begin(), samples_.end());
    }
    
    std::uint64_t get_max() const {
        return samples_.empty() ? 0 : *std::max_element(samples_.begin(), samples_.end());
    }
    
    void reset() {
        samples_.clear();
        next_index_ = 0;
    }

private:
    std::vector<std::uint64_t> samples_;
    std::size_t next_index_ = 0;
};

}