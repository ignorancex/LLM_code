/*
Copyright (c) 2021 ghcollin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "cpg.hpp"

#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_errno.h>

#include "expint.h"

#include <iostream>
#include <future>

namespace CPG {

template <typename Derived>
decltype(auto) expint_v(double n, const Eigen::DenseBase<Derived>& zs, double c) {
    return zs.derived().unaryExpr([n, c](double z){ return expint::expint_v(n, z, c); });
}

template <typename Derived>
decltype(auto) lower_incomplete_gamma_P(double n, const Eigen::DenseBase<Derived>& zs) {
    return zs.derived().unaryExpr([n](double z){ return gsl_sf_gamma_inc_P(n, z); });
}

double ln_gamma(double x) {
    return gsl_sf_lngamma(x);
}

template <typename Derived>
decltype(auto) expm1(const Eigen::DenseBase<Derived>& zs) {
    return zs.derived().unaryExpr([](double z){ return std::expm1(z); });
}

template <typename Derived>
bool isbad(const Eigen::DenseBase<Derived>& x) {
    return !x.allFinite();
}

bool series_1(SeriesWorkspace::Scratch& ws, MatrixRef power_series, int64_t max_power, constrArrayRef eps, double coeff, double Fb2, double n1) {
    auto& J0 = ws.J0; auto& ln_J0 = ws.ln_J0; auto& kj = ws.scratch1; 
    
    J0 = Fb2 * eps;
    ln_J0 = Eigen::log(J0);

    // j = 0
        double add = -1/(n1-1);
        kj = expint_v(n1, J0, 0.0);
        bool bad = isbad(kj);
        
        power_series.row(0) = coeff*expint_v(n1, J0, add);
    
    double ln_gamma_jp1 = 0;

    auto& t1 = ws.scratch2; auto& kjp1 = ws.scratch3;
    for (int64_t j = 0; j < max_power; j++) {
        double jp1_d = (double)(j+1);
        auto ln_gamma_jp2 = std::log(jp1_d) + ln_gamma_jp1;
        t1 = (1.0 - n1/jp1_d) * kj;

        if (isbad(t1)) {
            kjp1 = expint_v(n1 - jp1_d, J0, 0.0) * Eigen::exp(jp1_d*ln_J0 - ln_gamma_jp2);
            bad = bad || isbad(kjp1);
        } else {
            auto t2 = Eigen::exp((double)j * ln_J0 - J0 - ln_gamma_jp2);
            kjp1 = t1 + t2;
            // t2 is always positive, so cancellation can only occur when t1 is negative
            if ((kjp1 < -1E-3 * t1).any()) {
                kjp1 = expint_v(n1 - jp1_d, J0, 0.0) * Eigen::exp(jp1_d*ln_J0 - ln_gamma_jp2);
                bad = bad || isbad(kjp1);
            }
        }

        power_series.row(j+1) = coeff*kjp1;
        ln_gamma_jp1 = ln_gamma_jp2;
        kj.swap(kjp1);
    }

    return bad;
}

bool series_k(SeriesWorkspace::Scratch& ws, MatrixRef power_series, int64_t max_power, constrArrayRef eps, double coeff, double Fbk, double nk) {
    auto& J0 = ws.J0; auto& ln_J0 = ws.ln_J0; auto& kj = ws.scratch1;

    J0 = Fbk * eps;
    ln_J0 = Eigen::log(J0);
    
    double max_power_d = (double)max_power;
    // j = max_power
        kj = lower_incomplete_gamma_P(1+max_power_d-nk, J0) * Eigen::exp((nk-1)*ln_J0 + ln_gamma(1+max_power_d-nk) - ln_gamma(max_power_d + 1));
        bool bad = isbad(kj);
        power_series.row(max_power).array() += coeff*kj;

    double ln_gamma_jp1 = ln_gamma(max_power_d+1);

    auto& kjm1 = ws.scratch2;
    for (int64_t j = max_power; j > 0; j--) {
        double j_d = (double)j;
        
        auto ln_gamma_j = (j <= 2) ? 0 : ln_gamma_jp1 - std::log(j_d);
        auto t1 =  kj / (1 - nk/j_d);
        auto ln_t2 = (j_d-1) * ln_J0 - J0 - ln_gamma_j;
        
        if (j == 1) {
            kjm1 = t1 + expm1(ln_t2)/(j_d-nk);
        } else {
            kjm1 = t1 + Eigen::exp(ln_t2)/(j_d-nk);
        }

        power_series.row(j-1).array() += coeff*kjm1;
        ln_gamma_jp1 = ln_gamma_j;
        kj.swap(kjm1);
    }

    if (max_power == 0) {
        // Normally this correction gets applied by the expm1 function in the last iteration of the loop;
        // however, if max_power == 0, the loop never runs, so the correction needs to be applied now.
        power_series.row(0).array() += coeff*1/(nk - 1);
    }

    return bad;
}

bool series_model(SeriesWorkspace::Scratch& ws, MatrixRef power_series, int64_t max_power, constrArrayRef eps, double A, constrArrayRef Fbs, constrArrayRef ns) {
    double ln_coeff = 0;
    bool bad = series_1(ws, power_series, max_power, eps, A * std::exp(ln_coeff) * Fbs[0], Fbs[0], ns[0]);
    ln_coeff = 0;
    // TODO: multiple breaks
    if (Fbs.size() > 1) {
        throw std::runtime_error("Only a single flux break is currently implemented.");
    }
    bad = bad || series_k(ws, power_series, max_power, eps, A * std::exp(ln_coeff) * Fbs[Fbs.size()-1], Fbs[Fbs.size()-1], ns[ns.size()-1]);
    return bad;
}

std::tuple<double, double> streaming_log_sum_exp(double x, double norm_sum_exp, double current_max) {
    if (std::isinf(x)) {
        if (x == -std::numeric_limits<double>::infinity()) {
            return std::tuple<double, double>(norm_sum_exp, current_max);
        } else {
            return std::tuple<double, double>(std::numeric_limits<double>::quiet_NaN(), x);
        }
    } else if (x <= current_max) {
        return std::tuple<double,double>(norm_sum_exp + std::exp(x - current_max), current_max);
    } else {
        return std::tuple<double,double>(norm_sum_exp * std::exp(current_max - x) + 1.0, x);
    }
}

double ln_pk(int64_t k, double z0, constrArrayRef ln_z_series, constrArrayRef ln_z_series_reversed_cummax, rArrayRef ln_ps) {
    ln_ps[0] = z0;
    
    ln_ps[1] = ln_z_series[1] + ln_ps[0];

    for (int64_t i = 2; i <= k; i++) {

        double norm_sum_exp = 1.0;
        double current_max = ln_z_series[i] + ln_ps[0]; // j = 0
        for (int64_t j = 1; j < i; j++) {
            double term = std::log(1.0 - ((double)j)/(double)i) + ln_z_series[i - j] + ln_ps[j];
            std::tie(norm_sum_exp, current_max) = streaming_log_sum_exp(term, norm_sum_exp, current_max);
        }
        ln_ps[i] = std::log(norm_sum_exp) + current_max;
    }

    
    // This non-safe log sum exp does not appear to be any faster than the safe version above

    /* ln_ps[1] = ln_z_series_reversed[k-1] + ln_ps[0];
    
    asm("### Begin pk calculation.");
    double i_d = 2.0;
    double max_lnps = std::max(ln_ps[0], ln_ps[1]);
    for (int64_t i = 2; i <= k; i++, i_d++) {
        asm("### Begin term calculation.");
        double scale = ln_ps[i-1];
        ln_ps[i] = std::log(( (1.0 - rArray::LinSpaced(i, 0.0, i_d-1)/i_d).log() + ln_z_series_reversed.segment(k-i, i) + ln_ps.segment(0, i) - scale ).exp().sum()) + scale;
        max_lnps = std::max(max_lnps, ln_ps[i]);
    }
    asm("### End pk calculation.");*/

 
    return ln_ps[k];
}

double p_bin(LLHWorkspace& ws, int64_t k, double lambda) {
    auto&& z_series = ws.z_series.head(k+1);
    auto&& ln_z_series = ws.ln_z_series.head(k+1);
    auto&& ln_ps = ws.ln_ps.head(k+1);

    z_series[0] += - lambda;

    if (k == 0) {
        return z_series[0];
    } else {
        z_series[1] += lambda;
        ln_z_series = Eigen::log(z_series);
    
        double z0 = z_series[0];
        return ln_pk(k, z0, ln_z_series, z_series, ln_ps);
    }
}


bool series(SeriesWorkspace& ws, const std::vector<Model>& models) {

    for (size_t j = ws.models_power_series.size(); j < models.size(); j++) {
        ws.add_power_series();
    }

    bool bad = false;
    for (size_t j = 0; j < models.size(); j++) {
        auto& model = models[j];
        auto& power_series = ws.models_power_series[j];
        
        bad = bad || series_model(ws.scratch, power_series, ws.max_power, model.eps(Eigen::seqN(ws.eps_start, ws.eps_size)), model.A, model.Fbs, model.ns);

    }
    return bad;
}

/*
**
** Multi threaded routines
**
*/

bool series_spool(ThreadedWorkspace& threads_ws, const std::vector<Model>& models, size_t i) {
    auto n_threads = threads_ws.series_spaces.size();

    auto& thread_space = threads_ws.series_spaces[i];
    auto bad_promise = std::async(std::launch::async, series, std::ref(thread_space), std::ref(models));

    bool bad = false;
    if (i+1 < n_threads) {
        bad = series_spool(threads_ws, models, i+1);
    }

    bad = bad || bad_promise.get();

    return bad;
}

void combine_model_threads(LLHWorkspace& ws, ThreadedWorkspace& threads_ws, int64_t k, const std::vector<Model>& models, size_t bin) {
    auto n_threads = threads_ws.series_spaces.size();

    auto&& z_series = ws.z_series.head(k+1);
    bool first = true;

    for (size_t j = 0; j < models.size(); j++) {
        auto& model = models[j];

        for (size_t i = 0; i < n_threads; i++) {
            auto& thread_space = threads_ws.series_spaces[i];
            auto& thread_k_series = thread_space.models_power_series[j];

            auto addition = thread_k_series.topRows(k+1).lazyProduct(model.d_mu_eps(bin, Eigen::seqN(thread_space.eps_start, thread_space.eps_size)).transpose());
            
            if (first) {
                z_series.matrix().noalias() = addition.transpose();
                first = false;
            } else {
                z_series.matrix().noalias() += addition.transpose();
            }
        }
    }
}

double p_image_threads(LLHWorkspace& ws, ThreadedWorkspace& threads_ws, size_t data_offset, constrIntArrayRef data, const std::vector<Model>& models, constrArrayRef lambda_array) {
    double llh = 0;
    for (int i = 0; i < data.size(); i++) {
        auto k = data[i];
        combine_model_threads(ws, threads_ws, k, models, data_offset + i);
        llh += p_bin(ws, k, lambda_array[i]);
    }
    return llh;
}

double p_spool(ThreadedWorkspace& threads_ws, constrIntArrayRef data, const std::vector<Model>& models, constrArrayRef lambda_array, size_t i, size_t data_chunk) {
    auto data_start = i*data_chunk;
    auto data_stop = std::min(data_start + data_chunk, (size_t)data.size());
    auto data_size = data_stop - data_start;

    auto llh_promise = std::async(std::launch::async, p_image_threads, 
        std::ref(threads_ws.llh_spaces[i]),
        std::ref(threads_ws),
        data_start,
        data(Eigen::seqN(data_start, data_size)),
        std::ref(models),
        lambda_array(Eigen::seqN(data_start, data_size))
        );

    double llh = 0;
    if (i+1 < threads_ws.llh_spaces.size()) {
        llh = p_spool(threads_ws, data, models, lambda_array, i+1, data_chunk);
    }

    return llh + llh_promise.get();
}


void check_sanity(constrIntArrayRef data, const std::vector<Model>& models) {
    int model_no = 0;
    for (auto& model : models) {
        if (model.d_mu_eps.cols() != model.eps.cols()) {
            throw std::runtime_error("Number of epsilon bins in d_mu_eps (cols = " + std::to_string(model.d_mu_eps.cols()) + ") does not equal number of bins in eps (cols = " + std::to_string(model.eps.cols()) + ") for model " + std::to_string(model_no));
        }
        if (model.d_mu_eps.rows() != data.cols()) {
            throw std::runtime_error("Number of map bins in d_mu_eps (rows = " + std::to_string(model.d_mu_eps.rows()) + ") does not equal number of map bins in data (cols = " + std::to_string(data.cols()) + ") for model " + std::to_string(model_no));
        }
        model_no += 1;
    }   
}


std::tuple<double, bool> llh_threaded(ThreadedWorkspace& threads_ws, constrIntArrayRef data, const std::vector<Model>& models, constrArrayRef lambda_array) {
    auto gsl_handler = gsl_set_error_handler_off();
    //Eigen::internal::set_is_malloc_allowed(false);

    check_sanity(data, models);

    bool bad = series_spool(threads_ws, models, 0);

    auto n_threads = threads_ws.llh_spaces.size();
    auto data_size = data.size();
    size_t data_chunk = std::ceil(((double)data_size)/((double) n_threads));

    double llh = p_spool(threads_ws, data, models, lambda_array, 0, data_chunk);

    bad = bad || std::isnan(llh) || std::isinf(llh);

    gsl_set_error_handler(gsl_handler);
    //Eigen::internal::set_is_malloc_allowed(true);

    return std::tuple<double, bool>(llh, bad);
}

/*
**
** Single threaded routines
**
*/

void combine_models(LLHWorkspace& ws, SeriesWorkspace& series_ws, int64_t k, const std::vector<Model>& models, size_t bin) {
    auto&& z_series = ws.z_series.head(k+1);
    bool first = true;

    for (size_t j = 0; j < models.size(); j++) {
        auto& model = models[j];
        auto& k_series = series_ws.models_power_series[j];

        auto&& addition = k_series.topRows(k+1).lazyProduct(model.d_mu_eps.row(bin).transpose()).array();
        if (first) {
            z_series = addition;
            first = false;
        } else {
            z_series += addition;
        }
    }
}

double p_image(LLHWorkspace& ws, SeriesWorkspace& series_ws, constrIntArrayRef data, const std::vector<Model>& models, constrArrayRef lambda_array) {
    double llh = 0;
    for (int i = 0; i < data.size(); i++) {
        auto k = data[i];
        combine_models(ws, series_ws, k, models, i);
        llh += p_bin(ws, k, lambda_array[i]);
    }
    return llh;
}

std::tuple<double, bool> llh(Workspace& ws, constrIntArrayRef data, const std::vector<Model>& models, constrArrayRef lambda_array) {
    auto gsl_handler = gsl_set_error_handler_off();
    //Eigen::internal::set_is_malloc_allowed(false);

    check_sanity(data, models);

    bool bad = series(ws.series,  models);

    double llh = p_image(ws.llh, ws.series, data, models, lambda_array);

    bad = bad || std::isnan(llh) || std::isinf(llh);

    gsl_set_error_handler(gsl_handler);
    //Eigen::internal::set_is_malloc_allowed(true);

    return std::tuple<double, bool>(llh, bad);
}

}