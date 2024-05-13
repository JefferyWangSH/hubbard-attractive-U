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
             *  Filling number of electrons
             *
             *      n = 1/N \sum_i ( n_up(i) + n_dn(i) )
             *
             */
            static void measure_filling_number(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int nt = core.m_nt;
                const int ns = core.m_ns;
                
                double filling_num = 0.0;
                for (auto t = 0; t < nt; ++t) {
                    // SU2 symmetric Green's function, artificial consequence of our HS decomposition
                    const GF& gfttup = (*core.m_vecgftt)[t];
                    const GF& gfttdn = (*core.m_vecgftt)[t];
                    filling_num += (2. - (gfttup.trace()+gfttdn.trace()) / ns);
                }
                cache += filling_num / nt;
            }

            /*
             *
             *  Double occupation = 1/N \sum_i ( n_up(i) * n_dn(i) )
             *
             */
            static void measure_double_occupation(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int nt = core.m_nt;
                const int ns = core.m_ns;

                double temp_double_occu = 0.0;
                for (auto t = 0; t < nt; ++t) {
                    const GF& gfttup = (*core.m_vecgftt)[t];
                    const GF& gfttdn = (*core.m_vecgftt)[t];
                    for (auto i = 0; i < ns; ++i) {
                        temp_double_occu += (1.-gfttup(i,i)) * (1.-gfttdn(i,i));
                    }
                }
                cache += temp_double_occu / (ns*nt);
            }

            /*
             *
             *  Green's function
             *
             *    G(k,t) = < T c(k,t) c^+(k,0) >
             *           = 1/N \sum_{ij} exp( -i k*(rj-ri) ) * < T c(j,t) * c^+(i,0)>
             *
             */
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

            /*
             *
             *  Local density of state D = 1/N \sum_i < T c(i,t) * c^+(i,0) >
             *
             */
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

            /*
             *
             *  S-wave superconducting pairing \Delta = 1/sqrt(N) \sum_i c_dn(i) * c_up(i)
             *  Pairing correlation function Ps:
             *
             *      Ps  =  Delta^+ * Delta + Delta * Delta^+
             *          =  1/N \sum_{ij} ( (1-Gup)(j,i) * (1-Gdn)(j,i) + Gup(i,j) * Gdn(i,j) )
             *
             */
            static void measure_swave_pairing_correlation(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int nt = core.m_nt;
                const int ns = core.m_ns;
            
                double temp_s_wave_pairing = 0.0;
                for (auto t = 0; t < nt; ++t) {
                    // gf(i,j) = < c(i) c^+(j) >, and gfc(i,j) = < c^+(j) c(i) >
                    const GF& gfttup = (*core.m_vecgftt)[t];
                    const GF& gfttdn = (*core.m_vecgftt)[t];
                    const GF& gfcttup = GF::Identity(core.m_ng,core.m_ng) - gfttup;
                    const GF& gfcttdn = gfcttup;
                    for (auto i = 0; i < ns; ++i) {
                        for (auto j = 0; j < ns; ++j) {
                            temp_s_wave_pairing += gfcttup(j,i)*gfcttdn(j,i) + gfttup(i,j)*gfttdn(i,j);
                        }
                    }
                }
                cache += temp_s_wave_pairing / (ns*nt);
            }

            /*
             *
             *  (Local) Dynamic spin susceptibility (zz)
             *
             *      chi_{zz} = \sum_q < T Sz(q,t) Sz(q,0) >
             *               = 
             *
             */
            static void measure_dynamic_spin_susceptibility_zz(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                // TODO
            }

            /*
             *
             *  (Local) Dynamic spin susceptibility (-+)
             *
             *      chi_{-+} = \sum_q < T S_-(q,t) S_+(q,0) >
             *               = 
             *
             */
            static void measure_dynamic_spin_susceptibility_mp(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                // TODO
            }
    };

}

#endif