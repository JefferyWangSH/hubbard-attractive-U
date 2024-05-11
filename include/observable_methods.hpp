/*
 *   observable_methods.hpp
 * 
 *     Created on: May 9, 2024
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef OBSERVABLE_METHODS_HPP
#define OBSERVABLE_METHODS_HPP

#include "observable.hpp"
#include "dqmc_core.h"
#include "square_lattice.h"
#include "measure_handle.h"

namespace Observable {
    class Methods {
        public:
            using obs_struct = ObservableReal::obs_struct; // basically observables are real valued.
            using DqmcCore = DQMC::Core;
            using MeasureHandle = Measurement::Handle;
            using Hubbard = Model::HubbardAttractiveU;
            using Lattice = Lattice::SquareLattice;
            using GF = DQMC::GF;

            /*
             *
             *  TODO
             *
             */
            static void measure_double_occupation(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int nt = core.m_nt;
                const int ns = core.m_ns;

                double temp_double_occu = 0.0;
                for (auto t = 0; t < nt; ++t) {
                    // SU2 symmetric Green's function, artificial consequence of our HS decomposition
                    const GF& gfttup = (*core.m_vecgftt)[t];
                    const GF& gfttdn = (*core.m_vecgftt)[t];
                    for (auto i = 0; i < ns; ++i) {
                        temp_double_occu += (1-gfttup(i,i)) * (1-gfttdn(i,i));
                    }
                }
                cache += temp_double_occu / (ns*nt);
            }

            static void measure_green_functions(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int ns = core.m_ns;
                const int nt = core.m_nt;
                const int nk = handle.MomentumList().size();

                for (auto t = 0; t < nt; ++t) {
                    const GF& gft0 = (t==0)? (*core.m_vecgftt)[t] : (*core.m_vecgft0)[t]; // TODO: double check this
                    for (auto k = 0; k < nk; ++k) {
                        for (auto i = 0; i < ns; ++i) {
                            for (auto j = 0; j < ns; ++j) {
                                const auto fourierfactor = lattice.fourierFactor(lattice.displacement(i,j), handle.MomentumList(k));
                                cache(k,t) += (fourierfactor * gft0(j,i)).real() / ns;
                            }
                        }
                    }
                }
            }

            static void measure_density_of_states(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int nt = core.m_nt;
                const int ns = core.m_ns;

                for (auto t = 0; t < nt; ++t) {
                    const GF& gft0 = (t==0)? (*core.m_vecgftt)[t] : (*core.m_vecgft0)[t]; // TODO: double check this
                    cache(t) += gft0.trace() / ns;
                }
            }

            static void measure_swave_pairing_correlation(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {

            }

            static void measure_dynamic_spin_susceptibility(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {

            }
    };

}

#endif