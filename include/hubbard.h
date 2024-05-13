/*
 *   hubbard.h
 * 
 *     Created on: May 8, 2024
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef HUBBARD_H
#define HUBBARD_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>
#include <string>
#include <memory>
#include <functional>
#include "utils/fft_solver.hpp"

namespace DQMC { struct Params; class Core; }
namespace Lattice { class SquareLattice; }
namespace Model { class HubbardAttractiveU; }
namespace DQMC { namespace IO {
    void save_ising_fields_configs(std::string, const Model::HubbardAttractiveU&);
    void load_ising_fields_configs(std::string, Model::HubbardAttractiveU&);
}}

namespace Model {

    // ---------------------------------------------  Model::HubbardAttractiveU  --------------------------------------------------
    class HubbardAttractiveU {
        protected:
            int m_nl{}; 
            int m_ns{}; 
            int m_ng{};
            int m_nt{};
            double m_dt{};

            // ------------------------------------------  Model parameters  --------------------------------------------------
            double m_nnt{};
            double m_mu{};
            double m_u{};
            double m_alpha{};

            Eigen::ArrayXXi m_ising_fields{};

        public:
            friend void DQMC::IO::save_ising_fields_configs(std::string, const HubbardAttractiveU&);
            friend void DQMC::IO::load_ising_fields_configs(std::string, HubbardAttractiveU&);

            using DqmcParams = DQMC::Params;
            using DqmcCore = DQMC::Core;
            using Lattice = Lattice::SquareLattice;
            using refGreenFunc = Eigen::Ref<Eigen::MatrixXd>;

            // -------------------------------------------  Initializations  --------------------------------------------------
            void initialize(const DqmcParams& params, const Lattice& lattice);
            void set_ising_fields_to_random();

            // -------------------------------------------  Warpping methods  -------------------------------------------------
            // t takes values from 0 to nt-1, which is directly mapped to the ising fields at slice t
            // t=nt is not permitted although it's equvalent to t=0 physically due to the PBC.
            void multiply_B_from_left(refGreenFunc gf, const int t) const;
            void multiply_B_from_right(refGreenFunc gf, const int t) const;
            void multiply_invB_from_left(refGreenFunc gf, const int t) const;
            void multiply_invB_from_right(refGreenFunc gf, const int t) const;
            void multiply_adjB_from_left(refGreenFunc gf, const int t) const;
            
            // -----------------------------------------  Monte Carlo updates  ------------------------------------------------
            void locally_update_ising_field(int t, int i);
            const double get_acceptance_ratio(const DqmcCore& core, int t, int i) const;
            void update_green_function(DqmcCore& core, int t, int i);

        protected:
            // ------------------------------  Multiply the exponentials of hopping kernels  ----------------------------------
            bool m_is_fft{};
            std::function<void(refGreenFunc)> m_multiply_expK_from_left{};
            std::function<void(refGreenFunc)> m_multiply_expK_from_right{};
            std::function<void(refGreenFunc)> m_multiply_inv_expK_from_left{};
            std::function<void(refGreenFunc)> m_multiply_inv_expK_from_right{};
            std::function<void(refGreenFunc)> m_multiply_adj_expK_from_left{};

            void link2naive();
            void link2fft();
            
            Eigen::MatrixXd m_expK{};
            Eigen::MatrixXd m_inv_expK{};
            void multiply_expK_from_left(refGreenFunc gf) const;
            void multiply_expK_from_right(refGreenFunc gf) const;
            void multiply_inv_expK_from_left(refGreenFunc gf) const;
            void multiply_inv_expK_from_right(refGreenFunc gf) const;
            void multiply_adj_expK_from_left(refGreenFunc gf) const;
            
            using fftsolver = Utils::FFTSolver<double,2>;
            using ptrfftsolver = std::unique_ptr<fftsolver>;
            ptrfftsolver m_fftsolver{};
            Eigen::VectorXd m_expK_eigens{};
            Eigen::VectorXd m_inv_expK_eigens{};
            void multiply_expK_from_left_with_fft(refGreenFunc gf) const;
            void multiply_expK_from_right_with_fft(refGreenFunc gf) const;
            void multiply_inv_expK_from_left_with_fft(refGreenFunc gf) const;
            void multiply_inv_expK_from_right_with_fft(refGreenFunc gf) const;
            void multiply_adj_expK_from_left_with_fft(refGreenFunc gf) const;

            // -----------------------------  Multiply the exponentials of coupling kernels  ----------------------------------
            void multiply_expV_from_left(refGreenFunc gf, const int t) const;
            void multiply_expV_from_right(refGreenFunc gf, const int t) const;
            void multiply_inv_expV_from_left(refGreenFunc gf, const int t) const;
            void multiply_inv_expV_from_right(refGreenFunc gf, const int t) const;
            void multiply_adj_expV_from_left(refGreenFunc gf, const int t) const;
    };
}

#endif