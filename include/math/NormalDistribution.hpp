#pragma once

#include <cmath>
#include <immintrin.h>

namespace options::math {

class NormalDistribution {
public:
    static constexpr double SQRT_2_PI = 2.506628274631000502;
    static constexpr double INV_SQRT_2_PI = 0.3989422804014326779;
    static constexpr double SQRT_2 = 1.4142135623730950488;
    static constexpr double INV_SQRT_2 = 0.7071067811865475244;

    static double pdf(double x) noexcept {
        return INV_SQRT_2_PI * std::exp(-0.5 * x * x);
    }

    static double cdf(double x) noexcept {
        if (x >= 0.0) {
            return 0.5 + 0.5 * erf_approx(x * INV_SQRT_2);
        } else {
            return 0.5 - 0.5 * erf_approx(-x * INV_SQRT_2);
        }
    }

    static double inverse_cdf(double p) noexcept {
        if (p <= 0.0) return -std::numeric_limits<double>::infinity();
        if (p >= 1.0) return std::numeric_limits<double>::infinity();
        
        return rational_approximation(p) * SQRT_2;
    }

    static double d1(double S, double K, double T, double r, double vol, double q = 0.0) noexcept {
        const double vol_sqrt_T = vol * std::sqrt(T);
        return (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / vol_sqrt_T;
    }

    static double d2(double S, double K, double T, double r, double vol, double q = 0.0) noexcept {
        return d1(S, K, T, r, vol, q) - vol * std::sqrt(T);
    }

#ifdef __AVX2__
    static void vectorized_cdf(__m256d x, __m256d* result) noexcept {
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d inv_sqrt_2 = _mm256_set1_pd(INV_SQRT_2);
        
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        __m256d erf_val = vectorized_erf(_mm256_mul_pd(abs_x, inv_sqrt_2));
        
        __m256d sign_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ);
        __m256d pos_result = _mm256_add_pd(half, _mm256_mul_pd(half, erf_val));
        __m256d neg_result = _mm256_sub_pd(half, _mm256_mul_pd(half, erf_val));
        
        *result = _mm256_blendv_pd(neg_result, pos_result, sign_mask);
    }
#endif

private:
    static double erf_approx(double x) noexcept {
        const double a1 =  0.254829592;
        const double a2 = -0.284496736;
        const double a3 =  1.421413741;
        const double a4 = -1.453152027;
        const double a5 =  1.061405429;
        const double p  =  0.3275911;

        const double sign = (x >= 0) ? 1.0 : -1.0;
        x = std::abs(x);

        const double t = 1.0 / (1.0 + p * x);
        const double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

        return sign * y;
    }

    static double rational_approximation(double p) noexcept {
        static const double a[] = {
            -3.969683028665376e+01,  2.209460984245205e+02,
            -2.759285104469687e+02,  1.383577518672690e+02,
            -3.066479806614716e+01,  2.506628277459239e+00
        };
        
        static const double b[] = {
            -5.447609879822406e+01,  1.615858368580409e+02,
            -1.556989798598866e+02,  6.680131188771972e+01,
            -1.328068155288572e+01
        };
        
        static const double c[] = {
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00,  2.938163982698783e+00
        };
        
        static const double d[] = {
            7.784695709041462e-03,  3.224671290700398e-01,
            2.445134137142996e+00,  3.754408661907416e+00
        };

        if (p < 0.02425) {
            double q = std::sqrt(-2.0 * std::log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }

        if (p > 0.97575) {
            return -rational_approximation(1.0 - p);
        }

        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }

#ifdef __AVX2__
    static __m256d vectorized_erf(__m256d x) noexcept {
        const __m256d a1 = _mm256_set1_pd(0.254829592);
        const __m256d a2 = _mm256_set1_pd(-0.284496736);
        const __m256d a3 = _mm256_set1_pd(1.421413741);
        const __m256d a4 = _mm256_set1_pd(-1.453152027);
        const __m256d a5 = _mm256_set1_pd(1.061405429);
        const __m256d p = _mm256_set1_pd(0.3275911);
        const __m256d one = _mm256_set1_pd(1.0);

        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        __m256d t = _mm256_div_pd(one, _mm256_add_pd(one, _mm256_mul_pd(p, abs_x)));
        
        __m256d poly = _mm256_add_pd(a1, _mm256_mul_pd(t, a2));
        poly = _mm256_add_pd(poly, _mm256_mul_pd(t, _mm256_mul_pd(t, a3)));
        poly = _mm256_add_pd(poly, _mm256_mul_pd(t, _mm256_mul_pd(t, _mm256_mul_pd(t, a4))));
        poly = _mm256_add_pd(poly, _mm256_mul_pd(t, _mm256_mul_pd(t, _mm256_mul_pd(t, _mm256_mul_pd(t, a5)))));
        
        __m256d exp_term = _mm256_exp_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(abs_x, abs_x)));
        __m256d result = _mm256_sub_pd(one, _mm256_mul_pd(poly, _mm256_mul_pd(t, exp_term)));
        
        __m256d sign_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ);
        __m256d neg_result = _mm256_mul_pd(result, _mm256_set1_pd(-1.0));
        
        return _mm256_blendv_pd(neg_result, result, sign_mask);
    }
#endif
};

}