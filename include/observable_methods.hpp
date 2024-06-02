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

            // NOTE: for the attractive-U hubbard model,
            // both equal-time and dynamic Green's functions are automatically SU(2) symmetric, i.e. Gup = Gdn = G.
            // this is, however, an artificial consequence of our HS decomposition.

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
                    const GF& gfttup = (*core.m_vecgftt)[t];
                    const GF& gfttdn = gfttup;
                    filling_num += 2. - (gfttup.trace()+gfttdn.trace()) / ns;
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
                    const GF& gfttdn = gfttup;
                    for (auto i = 0; i < ns; ++i) {
                        temp_double_occu += (1.-gfttup(i,i)) * (1.-gfttdn(i,i));
                    }
                }
                cache += temp_double_occu / (ns*nt);
            }

            /*
             *
             *  Dynamic Green's functions
             *
             *    G(k,t) = < T c(k,t) c^+(k,0) >
             *           = 1/N \sum_{ij} exp( -i k*(rj-ri) ) * < T c(j,t) * c^+(i,0)>
             *
             */
            static void measure_dynamic_green_functions(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int ns = core.m_ns;
                const int nt = core.m_nt;
                const int nk = handle.MomentumList().size();

                for (auto t = 0; t < nt; ++t) {
                    // at t = 0, gft0 automatically degenerates to gf00
                    const GF& gft0 = (*core.m_vecgft0)[t];
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
             *  Local density of states D = 1/N \sum_i < T c(i,t) * c^+(i,0) >
             *
             */
            static void measure_local_density_of_states(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int nt = core.m_nt;
                const int ns = core.m_ns;

                for (auto t = 0; t < nt; ++t) {
                    const GF& gft0 = (*core.m_vecgft0)[t];
                    cache(t) += gft0.trace() / ns;
                }
            }

            /*
             *
             *  Static s-wave superconducting pairing \Delta = 1/sqrt(N) \sum_i c_dn(i) * c_up(i)
             *  Pairing correlation function Ps:
             *
             *      Ps  =  Delta^+ * Delta + Delta * Delta^+
             *          =  1/N \sum_{ij} ( (1-Gup)(j,i) * (1-Gdn)(j,i) + Gup(i,j) * Gdn(i,j) )
             *
             */
            static void measure_static_swave_pairing_correlation(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int nt = core.m_nt;
                const int ns = core.m_ns;
            
                double temp_s_wave_pairing = 0.0;
                for (auto t = 0; t < nt; ++t) {
                    // gf(i,j) = < c(i) c^+(j) >, and gfc(i,j) = < c^+(j) c(i) >
                    const GF& gfttup = (*core.m_vecgftt)[t];
                    const GF& gfcttup = GF::Identity(core.m_ng, core.m_ng) - gfttup;
                    const GF& gfttdn = gfttup;
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
             *  Local dynamic transverse spin correlation function
             *
             *      \chi_{\perp}(t) = chi_{xx} + chi_{yy} = 1/2 ( chi_{-+} + chi_{+-} )
             *                      = 1/(2N) \sum_q ( < T S^-(q,t) S^+(q,0) > + < T S^+(q,t) S^-(q,0) > )
             *                      = 1/(2N) \sum_i ( < T S^-(i,t) S^+(i,0) > + < T S^+(i,t) S^-(i,0) > )
             *
             *                      = 1/(2N) \sum_i ( < T cdn^+(i,t) cdn(i,0) > < T cup(i,t) cup^+(i,0) >
             *                                      + < T cup^+(i,t) cup(i,0) > < T cdn(i,t) cdn^+(i,0) > )
             *
             *                      = 1/(2N) \sum_i ( [delta_{t,0} I - Gdn(0,t)]_{ii} * [Gup(t,0)]_{ii}
             *                                      + [delta_{t,0} I - Gup(0,t)]_{ii} * [Gdn(t,0)]_{ii} )
             *
             *                      = 1/N \sum_i ( delta_{t,0} - G(0,t)_{ii} ) * G(t,0)_{ii}
             *  
             *  where in the last step we have made use of the fact that Gup = Gdn = G.
             *  For the attractive-U hubbard model, the HS decomposition we adopted preserves the SU(2) symmetry,
             *  such that it's equivalent to measure either \chi_perp or 2\chi_zz for evaluating the transeverse spin correlation.
             *
             *  Due to our convention that G0t(t->0) = G(0,0) - 1 and Gt0(t->0) = G(0,0), the above formula can be simplified as
             * 
             *      \chi_{\perp}(t) = 1/N \sum_i (- G0t(i,i)) * Gt0(i,i)
             *
             */
            static void measure_local_dynamic_transverse_spin_correlation(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                const int nt = core.m_nt;
                const int ns = core.m_ns;

                for (auto t = 0; t < nt; ++t) {
                    const GF& gft0up = (*core.m_vecgft0)[t];
                    const GF& gf0tup = (*core.m_vecgf0t)[t];
                    const GF& gft0dn = gft0up;
                    const GF& gf0tdn = gf0tup;
                    double temp_trans_spin_corr = 0.0;
                    for (auto i = 0; i < ns; ++i) {
                        temp_trans_spin_corr -= gf0tdn(i,i) * gft0up(i,i) + gf0tup(i,i) * gft0dn(i,i);
                    }
                    cache(t) += temp_trans_spin_corr / (2*ns);
                }
            }
    };

}

#endif