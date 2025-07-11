#pragma once

#include <functional>
#include <cmath>
#include <limits>
#include <optional>

namespace options::math {

struct OptimizationResult {
    double value = 0.0;
    double function_value = 0.0;
    std::size_t iterations = 0;
    bool converged = false;
    double tolerance_achieved = std::numeric_limits<double>::max();
};

class NewtonRaphsonSolver {
public:
    struct Parameters {
        double tolerance = 1e-8;
        std::size_t max_iterations = 100;
        double step_size = 1.0;
        bool use_adaptive_step = true;
    };

    static OptimizationResult solve(
        std::function<double(double)> f,
        std::function<double(double)> df,
        double initial_guess,
        const Parameters& params = Parameters{}) {
        
        OptimizationResult result;
        result.value = initial_guess;
        
        for (result.iterations = 0; result.iterations < params.max_iterations; ++result.iterations) {
            const double fx = f(result.value);
            const double dfx = df(result.value);
            
            result.function_value = fx;
            result.tolerance_achieved = std::abs(fx);
            
            if (result.tolerance_achieved < params.tolerance) {
                result.converged = true;
                break;
            }
            
            if (std::abs(dfx) < std::numeric_limits<double>::epsilon()) {
                break;
            }
            
            double step = params.step_size * fx / dfx;
            
            if (params.use_adaptive_step) {
                step = adaptive_step_size(f, result.value, step, fx);
            }
            
            result.value -= step;
        }
        
        return result;
    }

private:
    static double adaptive_step_size(
        const std::function<double(double)>& f,
        double x,
        double proposed_step,
        double current_fx) {
        
        constexpr double reduction_factor = 0.5;
        constexpr std::size_t max_reductions = 10;
        
        double step = proposed_step;
        
        for (std::size_t i = 0; i < max_reductions; ++i) {
            const double new_x = x - step;
            const double new_fx = f(new_x);
            
            if (std::abs(new_fx) < std::abs(current_fx)) {
                break;
            }
            
            step *= reduction_factor;
        }
        
        return step;
    }
};

class BrentSolver {
public:
    struct Parameters {
        double tolerance = 1e-12;
        std::size_t max_iterations = 100;
    };

    static OptimizationResult solve(
        std::function<double(double)> f,
        double lower_bound,
        double upper_bound,
        const Parameters& params = Parameters{}) {
        
        OptimizationResult result;
        
        double a = lower_bound;
        double b = upper_bound;
        double fa = f(a);
        double fb = f(b);
        
        if (fa * fb >= 0.0) {
            return result;
        }
        
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
        
        double c = a;
        double fc = fa;
        bool mflag = true;
        double d = 0.0;
        
        for (result.iterations = 0; result.iterations < params.max_iterations; ++result.iterations) {
            if (std::abs(b - a) < params.tolerance) {
                result.converged = true;
                result.value = b;
                result.function_value = fb;
                result.tolerance_achieved = std::abs(b - a);
                break;
            }
            
            double s;
            
            if (fa != fc && fb != fc) {
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
                    (b * fa * fc) / ((fb - fa) * (fb - fc)) +
                    (c * fa * fb) / ((fc - fa) * (fc - fb));
            } else {
                s = b - fb * (b - a) / (fb - fa);
            }
            
            const double tmp1 = (3.0 * a + b) / 4.0;
            const bool condition1 = (s < tmp1 && s < b) || (s > tmp1 && s > b);
            const bool condition2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
            const bool condition3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;
            const bool condition4 = mflag && std::abs(b - c) < params.tolerance;
            const bool condition5 = !mflag && std::abs(c - d) < params.tolerance;
            
            if (condition1 || condition2 || condition3 || condition4 || condition5) {
                s = (a + b) / 2.0;
                mflag = true;
            } else {
                mflag = false;
            }
            
            const double fs = f(s);
            d = c;
            c = b;
            fc = fb;
            
            if (fa * fs < 0.0) {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
            
            if (std::abs(fa) < std::abs(fb)) {
                std::swap(a, b);
                std::swap(fa, fb);
            }
        }
        
        result.value = b;
        result.function_value = fb;
        result.tolerance_achieved = std::abs(fb);
        
        return result;
    }
};

class BisectionSolver {
public:
    struct Parameters {
        double tolerance = 1e-10;
        std::size_t max_iterations = 100;
    };

    static OptimizationResult solve(
        std::function<double(double)> f,
        double lower_bound,
        double upper_bound,
        const Parameters& params = Parameters{}) {
        
        OptimizationResult result;
        
        double a = lower_bound;
        double b = upper_bound;
        double fa = f(a);
        double fb = f(b);
        
        if (fa * fb >= 0.0) {
            return result;
        }
        
        for (result.iterations = 0; result.iterations < params.max_iterations; ++result.iterations) {
            const double c = (a + b) / 2.0;
            const double fc = f(c);
            
            result.value = c;
            result.function_value = fc;
            result.tolerance_achieved = std::abs(fc);
            
            if (result.tolerance_achieved < params.tolerance || (b - a) / 2.0 < params.tolerance) {
                result.converged = true;
                break;
            }
            
            if (fa * fc < 0.0) {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
        }
        
        return result;
    }
};

template<typename T>
class FiniteDifference {
public:
    static constexpr T DEFAULT_H = static_cast<T>(1e-8);

    static T forward_difference(std::function<T(T)> f, T x, T h = DEFAULT_H) {
        return (f(x + h) - f(x)) / h;
    }

    static T backward_difference(std::function<T(T)> f, T x, T h = DEFAULT_H) {
        return (f(x) - f(x - h)) / h;
    }

    static T central_difference(std::function<T(T)> f, T x, T h = DEFAULT_H) {
        return (f(x + h) - f(x - h)) / (T{2} * h);
    }

    static T second_derivative(std::function<T(T)> f, T x, T h = DEFAULT_H) {
        return (f(x + h) - T{2} * f(x) + f(x - h)) / (h * h);
    }

    static T fourth_order_derivative(std::function<T(T)> f, T x, T h = DEFAULT_H) {
        return (-f(x + T{2} * h) + T{8} * f(x + h) - T{8} * f(x - h) + f(x - T{2} * h)) / (T{12} * h);
    }
};

}