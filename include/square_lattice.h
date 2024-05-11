/*
 *   square_lattice.h
 * 
 *     Created on: May 8, 2024
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef SQUARE_LATTICE_H
#define SQUARE_LATTICE_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>

namespace DQMC { struct Params; }

namespace Lattice {

    // ------------------------------------------  Lattice::SquareLattice  ----------------------------------------------
    class SquareLattice {
        protected:
            int m_nl{};
            int m_ns{};
            int m_nkstars{};

            Eigen::ArrayXXi m_nn_table{};
            Eigen::MatrixXd m_nn_hoppings{};
            Eigen::ArrayXXi m_displacement_table{};
            Eigen::ArrayXXd m_kstars_table{};
            Eigen::ArrayXXcd m_fourier_factor_table{};

            std::vector<int> m_all_kstars{};
            std::vector<int> m_x2k2x_line{};
            std::vector<int> m_gamma2m_line{};

        public:
            // --------------------------------------  Initializations  ------------------------------------------
            using DqmcParams = DQMC::Params;
            void initialize(const DqmcParams& params);
            void initialize_nn_table();
            void initialize_nn_hoppings();
            void initialize_kstars_table();
            void initialize_displacement_table();
            void initialize_momentum_lists();
            void initialize_fourier_factor_table();

            // ----------------------------------------  Interfaces  ---------------------------------------------
            const int nn(int i, int dir) const { return this->m_nn_table(i, dir); }
            const Eigen::MatrixXd nn_hoppings() const { return this->m_nn_hoppings; }

            const Eigen::Array2d momentum(const int k) const { return this->m_kstars_table.row(k); }
            const int displacement(const int i, const int j) const { return this->m_displacement_table(i,j); }
            const std::complex<double> fourierFactor(const int i, const int k) const { return this->m_fourier_factor_table(i,k); }

            const std::vector<int> allKstars() const { return this->m_all_kstars; }
            const std::vector<int> x2k2xLine() const { return this->m_x2k2x_line; }
            const std::vector<int> gamma2mLine() const { return this->m_gamma2m_line; }
    };
}

#endif