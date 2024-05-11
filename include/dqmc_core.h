/*
 *   dqmc_core.h
 * 
 *     Created on: May 8, 2024
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef DQMC_CORE_H
#define DQMC_CORE_H

#include <memory>
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>
#include "utils/svd_stack.hpp"

namespace Model { class HubbardAttractiveU; }
namespace Observable { class Methods; }
namespace Measurement { class Handle; }

namespace DQMC {

    struct Params;

    using GF = Eigen::MatrixXd;
    using vecGF = std::vector<GF>;
    using ptrGF = std::unique_ptr<GF>;
    using ptrVecGF = std::unique_ptr<vecGF>;
    using svdStack = Utils::SvdStack<double>;
    using ptrSvdStack = std::unique_ptr<svdStack>;

    // --------------------------------------------------------  DQMC::Core  --------------------------------------------------------------
    class Core {
        protected:
            // ------------------------------------------------  Core params  --------------------------------------------------------
            int m_nl{};
            int m_ns{};
            int m_ng{};
            int m_nt{};
            double m_dt{};
            int m_stabilization_pace{};

            bool m_is_thermalization{false};
            bool m_is_equaltime{};
            bool m_is_dynamic{};
            double m_equaltime_wrap_error{0.};
            double m_dynamic_wrap_error{0.};
            int m_current_t{0};

            // ------------------------------------  Equal-time and dynamic Green's functions  ---------------------------------------
            // equal-time green's function G(t,t)
            // spin index is omitted since our HS decomposition preserves SU2 symm.
            ptrGF m_gftt{};
            ptrVecGF m_vecgftt{};

            // time-displaced green's functions G(t,0) and G(0,t)
            ptrGF m_gft0{};
            ptrGF m_gf0t{};
            ptrVecGF m_vecgft0{};
            ptrVecGF m_vecgf0t{};

            // -------------------------------------  SvdStacks for numerical stabilization  -----------------------------------------
            ptrSvdStack m_svdstack_left{};
            ptrSvdStack m_svdstack_right{};
        
        public:
            friend class Observable::Methods;

            using DqmcParams = DQMC::Params;
            using Hubbard = Model::HubbardAttractiveU;
            using MeasureHandle = Measurement::Handle;
            using refGreenFunc = Eigen::Ref<Eigen::MatrixXd>;

            const refGreenFunc current_gftt() const { assert(this->m_gftt); return *this->m_gftt; }
            const refGreenFunc gftt(int t) const { assert(this->m_vecgftt && t>=0 && t<this->m_nt); return (*this->m_vecgftt)[t]; }
            const refGreenFunc gft0(int t) const { assert(this->m_vecgft0 && t>=0 && t<this->m_nt); return (*this->m_vecgft0)[t]; }
            const refGreenFunc gf0t(int t) const { assert(this->m_vecgf0t && t>=0 && t<this->m_nt); return (*this->m_vecgf0t)[t]; }
            const double equaltimeWrapError() const { return this->m_equaltime_wrap_error; }
            const double dynamicWrapError() const { return this->m_dynamic_wrap_error; }

            // -----------------------------------------------  Initializations  -----------------------------------------------------
            void initialize(const DqmcParams& params, const MeasureHandle& meas_handle);
            void initialize_svdstacks(const Hubbard& model);
            void initialize_green_functions();

            // set up whether the MC simulation is in the thermalization phase
            // if that's the case, avoid collecting the equal-time Green's functions during MC sweeps.
            void set_thermalization(bool is_thermalization) { this->m_is_thermalization = is_thermalization; }

            // ----------------------------------------------  Monte Carlo sweeps  ---------------------------------------------------
            void sweep_from_0_to_beta(Hubbard& model);
            void sweep_from_beta_to_0(Hubbard& model);

            // sweep forwards from 0 to beta and collect the dynamic Green's functions
            // with the ising fields unchanged
            void sweep_for_dynamic_green_functions(Hubbard& model);

        protected:
            void allocate_svdstacks();
            void allocate_green_functions();

            // update the ising fields at time slice t using Metropolis algorithm
            void metropolis_update(Hubbard& model, int t);
            
            // wrap the equal-time Green's functions from time t to t+1(t-1)
            void wrap_from_0_to_beta(const Hubbard& model, int t);
            void wrap_from_beta_to_0(const Hubbard& model, int t);

            const int t2index(const int t);
    };
}

#endif