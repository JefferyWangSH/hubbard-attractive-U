/*
 *   dqmc_handle.cpp
 * 
 *     Created on: Aug 3, 2023
 *         Author: Jeffery Wang
 * 
 */

#include "dqmc_handle.h"
#include "dqmc_core.h"
#include "measure_handle.h"
#include "utils/progressbar.hpp"

#include <cmath>
#include <cassert>

namespace DQMC {

    // definitions of the static members
    bool Handle::m_show_progress_bar{true};
    unsigned int Handle::m_progress_bar_width{70};
    unsigned int Handle::m_refresh_rate{10};
    char Handle::m_progress_bar_complete_char{'='}, Handle::m_progress_bar_incomplete_char{' '};
    std::chrono::steady_clock::time_point Handle::m_begin_time{}, Handle::m_end_time{};

    // set up whether to show the process bar or not
    void Handle::show_progress_bar(bool show_progress_bar) { Handle::m_show_progress_bar = show_progress_bar; }

    // set up the format of the progress bar
    void Handle::progress_bar_format(unsigned int width, char complete, char incomplete)
    {
        Handle::m_progress_bar_width = width;
        Handle::m_progress_bar_complete_char = complete;
        Handle::m_progress_bar_incomplete_char = incomplete;
    }

    // set up the rate of refreshing the progress bar
    void Handle::set_refresh_rate(unsigned int refresh_rate)
    {
        assert(refresh_rate != 0);
        Handle::m_refresh_rate = refresh_rate;
    }

    // timer functions
    void Handle::timer_begin() { Handle::m_begin_time = std::chrono::steady_clock::now(); }
    void Handle::timer_end()   { Handle::m_end_time = std::chrono::steady_clock::now(); }
    const double Handle::timer() { 
        return std::chrono::duration_cast<std::chrono::milliseconds>(Handle::m_end_time - Handle::m_begin_time).count(); 
    }

    // ------------------------------------------------------------------------------------------
    //
    //                               Thermalization process
    //
    // ------------------------------------------------------------------------------------------
    void Handle::thermalize(DqmcCore& core, const MeasureHandle& meas_handle, Hubbard& model, const Lattice& lattice)
    {
        if (meas_handle.isWarmUp()) {
            // create the progress bar
            progresscpp::ProgressBar progressbar(std::ceil(meas_handle.WarmUpSweeps()/2.),  // total loops 
                                                 Handle::m_progress_bar_width,              // bar width
                                                 Handle::m_progress_bar_complete_char,      // complete character
                                                 Handle::m_progress_bar_incomplete_char     // incomplete character
                                                );
            // display the progress bar
            if (Handle::m_show_progress_bar) { std::cout << ">> Warming up "; progressbar.display(); }

            // warm-up sweeps
            core.set_thermalization(true);
            for (int sweep = 1; sweep <= std::ceil(meas_handle.WarmUpSweeps()/2.); ++sweep) {
                // sweep forth and back without measurments
                core.sweep_from_0_to_beta(model);
                core.sweep_from_beta_to_0(model);

                // record the tick
                ++progressbar;
                if (Handle::m_show_progress_bar && (sweep % Handle::m_refresh_rate == 0)) {
                    std::cout << ">> Warming up "; progressbar.display();
                }
            }
            
            // progress bar finish
            if (Handle::m_show_progress_bar) { std::cout << ">> Warming up "; progressbar.done(); }
        }
    }

    // ------------------------------------------------------------------------------------------
    //
    //                               Measurement process
    //
    // ------------------------------------------------------------------------------------------
    void Handle::measure(DqmcCore& core, MeasureHandle& meas_handle, Hubbard& model, const Lattice& lattice)
    {
        if (meas_handle.isEqualTime() || meas_handle.isDynamic()) {
            // create the progress bar
            const int binsize = (meas_handle.isDynamic())? meas_handle.BinsCapacity() : std::ceil(meas_handle.BinsCapacity()/2.);
            const int total_ticks = meas_handle.BinsNum() * (binsize + std::ceil(meas_handle.SweepsBetweenBins()/2.));
            progresscpp::ProgressBar progressbar(total_ticks,
                                                 Handle::m_progress_bar_width,
                                                 Handle::m_progress_bar_complete_char,
                                                 Handle::m_progress_bar_incomplete_char);
            
            // display the progress bar
            if (Handle::m_show_progress_bar) { std::cout << ">> Measuring  "; progressbar.display(); }

            // measuring sweeps
            core.set_thermalization(false);
            for (int bin = 0; bin < meas_handle.BinsNum(); ++bin) {
                // avoid correlations between adjoining bins
                for (int sweep = 1; sweep <= std::ceil(meas_handle.SweepsBetweenBins()/2.); ++sweep) {
                    // record the tick
                    const int current_tick = sweep + bin*(binsize+std::ceil(meas_handle.SweepsBetweenBins()/2.));
                    core.sweep_from_0_to_beta(model);
                    core.sweep_from_beta_to_0(model);

                    ++progressbar;
                    if (Handle::m_show_progress_bar && (current_tick % Handle::m_refresh_rate == 0)) {
                        std::cout << ">> Measuring  "; progressbar.display();
                    }
                }

                for (int sweep = 1; sweep <= binsize; ++sweep) {
                    // record the tick
                    const int current_tick = sweep + bin*binsize + (bin+1)*std::ceil(meas_handle.SweepsBetweenBins()/2.);

                    // sweep forth from 0 to beta
                    if (meas_handle.isDynamic()) {
                        core.sweep_for_dynamic_green_functions(model);
                        meas_handle.measure_dynamic_observables(core, model, lattice);
                        // donot measure equal-time observables at this step
                    }
                    else {
                        core.sweep_from_0_to_beta(model);
                        meas_handle.measure_equaltime_observables(core, model, lattice);
                    }

                    // sweep back from beta to 0
                    core.sweep_from_beta_to_0(model);
                    if (meas_handle.isEqualTime()) {
                        meas_handle.measure_equaltime_observables(core, model, lattice);
                    }

                    ++progressbar;
                    if (Handle::m_show_progress_bar && (current_tick % Handle::m_refresh_rate == 0)) {
                        std::cout << ">> Measuring  "; progressbar.display();
                    }
                }
                
                // store the collected data within one bin
                meas_handle.normalize_cache();
                meas_handle.push_cache_to_data(bin);
                meas_handle.clear_cache();
            }
            
            // progress bar finish
            if (Handle::m_show_progress_bar) { std::cout << ">> Measuring  "; progressbar.done(); }
        }
    }

    void Handle::analyse(MeasureHandle& meas_handle)
    {
        // analyse the collected data when the measuring process is done
        meas_handle.analyse();
    }
}