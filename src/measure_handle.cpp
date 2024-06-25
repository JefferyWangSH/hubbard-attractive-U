/*
 *   measure_handle.cpp
 * 
 *     Created on: May 10, 2024
 *         Author: Jeffery Wang
 * 
 */

#include <stdexcept>
#include "measure_handle.h"
#include "observable.hpp"
#include "dqmc_params.hpp"
#include "square_lattice.h"

namespace Measurement {

    void Handle::initialize(const DqmcParams& params, const Lattice& lattice)
    {
        this->m_sweeps_warmup = params.sweeps_warmup;
        this->m_bin_num = params.bin_num;
        this->m_bin_capacity = params.bin_capacity;
        this->m_sweeps_between_bins = params.sweeps_between_bins;

        this->m_list = params.observable_list;

        if (auto momentum_list = std::get_if<std::string>(&params.momentum_list)) {
            if (*momentum_list == "AllKstars") { this->m_momentum_list = lattice.allKstars(); }
            else if (*momentum_list == "X2K2XLine") { this->m_momentum_list = lattice.x2k2xLine(); }
            else if (*momentum_list == "Gamma2MLine") { this->m_momentum_list = lattice.gamma2mLine(); }
            else { throw std::runtime_error("Measurement::Handle::initialize(): invalid momentum list."); }
        }
        else if (auto momentum_list = std::get_if<std::vector<int>>(&params.momentum_list)) {
            const int nkstars = lattice.allKstars().size();
            if (std::any_of(momentum_list->begin(), momentum_list->end(), [=](int i){return i >= nkstars;})) {
                throw std::runtime_error("Measurement::Handle::initialize(): index of momentum out of range.");
            }
            this->m_momentum_list = *momentum_list;
        }

        // initialize Observable::Handle
        Observable::Handle::initialize(this->m_list);

        this->m_is_warmup = (this->m_sweeps_warmup > 0);
        this->m_is_equaltime = (!this->m_obs_list_equaltime.empty()) && (this->m_bin_num > 0) && (this->m_bin_capacity > 0);
        this->m_is_dynamic = (!this->m_obs_list_dynamic.empty()) && (this->m_bin_num > 0) && (this->m_bin_capacity > 0);
        
        // ---------------------------------------------------
        //
        //      Set up dimensions for the observables
        //
        // ---------------------------------------------------
        const long unsigned nt = params.nt;
        const long unsigned nk = this->m_momentum_list.size();
        for (auto& it : this->m_obs_map) {
            // customize the dimension info for different observables
            // NOTE: empty input {} for 0-dimensional (scalar) observables
            const auto& obsname = it.first;
            auto& obs = it.second;
            if (obsname == "FillingNumber"                        ) { obs->set_shape(this->m_bin_num, {}); }
            if (obsname == "DoubleOccupation"                     ) { obs->set_shape(this->m_bin_num, {}); }
            if (obsname == "StaticSWavePairingCorrelation"        ) { obs->set_shape(this->m_bin_num, {}); }
            if (obsname == "DynamicGreenFunctions"                ) { obs->set_shape(this->m_bin_num, {nk, nt}); }
            if (obsname == "LocalDensityOfStates"                 ) { obs->set_shape(this->m_bin_num, {nt}); }
            if (obsname == "LocalDynamicTransverseSpinCorrelation") { obs->set_shape(this->m_bin_num, {nt}); }
        }
    }

    void Handle::measure_equaltime_observables(const DqmcCore& core, const Hubbard& model, const Lattice& lattice)
    {
        for (auto& obs : this->m_obs_list_equaltime) {
            obs->measure(obs->cache(), core, *this, model, lattice);
        }
    }

    void Handle::measure_dynamic_observables(const DqmcCore& core, const Hubbard& model, const Lattice& lattice)
    {
        for (auto& obs : this->m_obs_list_dynamic) {
            obs->measure(obs->cache(), core, *this, model, lattice);
        }
    }

    void Handle::normalize_cache()
    {
        for (auto& obs : this->m_obs_list_equaltime) { obs->normalize_cache(); }
        for (auto& obs : this->m_obs_list_dynamic) { obs->normalize_cache(); }
    }

    void Handle::push_cache_to_data(const std::size_t bin)
    {
        for (auto& obs : this->m_obs_list_equaltime) { obs->push_cache_to_data(bin); }
        for (auto& obs : this->m_obs_list_dynamic) { obs->push_cache_to_data(bin); }
    }

    void Handle::analyse()
    {
        for (auto& obs : this->m_obs_list_equaltime) { obs->analyse(); }
        for (auto& obs : this->m_obs_list_dynamic) { obs->analyse(); }
    }

    void Handle::clear_cache()
    {
        for (auto& obs : this->m_obs_list_equaltime) { obs->clear_cache(); }
        for (auto& obs : this->m_obs_list_dynamic) { obs->clear_cache(); }
    }
}