/*
 *   square_lattice.cpp
 * 
 *     Created on: May 8, 2024
 *         Author: Jeffery Wang
 * 
 */

#include "square_lattice.h"
#include "dqmc_params.hpp"
#include <cmath>

namespace Lattice {
    void SquareLattice::initialize(const DqmcParams& params)
    {
        this->m_nl = params.nl;
        this->m_ns = params.ns;
        // k stars live in the zone encircled by the loop (0,0) -> (pi,0) -> (pi,pi) -> (0,pi) -> (0,0).
        this->m_nkstars = (std::floor(this->m_nl/2.)+1) * (std::floor(this->m_nl/2.)+1);

        this->initialize_kstars_table();
        this->initialize_momentum_lists();
        
        this->initialize_nn_table();
        this->initialize_nn_hoppings();
        this->initialize_displacement_table();        
        this->initialize_fourier_factor_table();
    }

    void SquareLattice::initialize_nn_table()
    {
        // the coordination number for 2d square lattice is 4
        // correspondense between the table index and the displacement direction:
        // 0: (x+1, y)    1: (x, y+1)
        // 2: (x-1, y)    3: (x, y-1)
        this->m_nn_table.resize(this->m_ns, 4);
        this->m_nn_table.setZero();
        for (auto i = 0; i < this->m_ns; ++i) {
            const auto x = i % this->m_nl;
            const auto y = i / this->m_nl;
            this->m_nn_table(i,0) = ((x+1)%this->m_nl) + this->m_nl*y;
            this->m_nn_table(i,2) = ((x-1+this->m_nl)%this->m_nl) + this->m_nl*y;
            this->m_nn_table(i,1) = x + this->m_nl*((y+1)%this->m_nl);
            this->m_nn_table(i,3) = x + this->m_nl*((y-1+this->m_nl)%this->m_nl);
        }
    }

    void SquareLattice::initialize_nn_hoppings()
    {
        this->m_nn_hoppings.resize(this->m_ns, this->m_ns);
        this->m_nn_hoppings.setZero();
        for (auto i = 0; i < this->m_ns; ++i) {
            // direction 0 for x+1 and 1 for y+1 
            const auto xplus1 = this->nn(i,0);
            const auto yplus1 = this->nn(i,1);
            this->m_nn_hoppings(i, xplus1) += 1.0;
            this->m_nn_hoppings(xplus1, i) += 1.0;
            this->m_nn_hoppings(i, yplus1) += 1.0;
            this->m_nn_hoppings(yplus1, i) += 1.0;
        }
    }

    void SquareLattice::initialize_displacement_table()
    {
        this->m_displacement_table.resize(this->m_ns, this->m_ns);
        this->m_displacement_table.setZero();
        for (auto i = 0; i < this->m_ns; ++i) {
            const auto xi = i % this->m_nl;
            const auto yi = i / this->m_nl;
            
            for (auto j = 0; j < this->m_ns; ++j) {
                const auto xj = j % this->m_nl;
                const auto yj = j / this->m_nl;

                // displacement pointing from site i to site j, i.e. rj - ri
                const auto dx = (xj-xi+this->m_nl) % this->m_nl;
                const auto dy = (yj-yi+this->m_nl) % this->m_nl;
                this->m_displacement_table(i,j) = dx + dy*this->m_nl;
            }
        }
    }

    void SquareLattice::initialize_kstars_table()
    {
        this->m_kstars_table.resize(this->m_nkstars, 2);
        this->m_kstars_table.setZero();
        int n = 0;
        for (auto i = std::ceil(this->m_nl/2.); i <= this->m_nl; ++i) {
            for (auto j = std::ceil(this->m_nl/2.); j <= this->m_nl; ++j) {
                this->m_kstars_table(n,0) = static_cast<double>(i)/this->m_nl * 2*M_PI - M_PI;
                this->m_kstars_table(n,1) = static_cast<double>(j)/this->m_nl * 2*M_PI - M_PI;
                n++;
            }
        }
    }

    void SquareLattice::initialize_momentum_lists()
    {
        this->m_all_kstars.clear();
        this->m_x2k2x_line.clear();
        this->m_gamma2m_line.clear();
        this->m_all_kstars.shrink_to_fit();
        this->m_x2k2x_line.shrink_to_fit();
        this->m_gamma2m_line.shrink_to_fit();
        
        this->m_all_kstars.reserve(this->m_nkstars);
        for (auto i = 0; i < this->m_nkstars; ++i) {
            this->m_all_kstars.emplace_back(i);
        }
        this->m_x2k2x_line.reserve(std::floor(this->m_nl/2.)+1);
        for (auto i = 0; i < std::floor(this->m_nl/2.)+1; ++i) {
            this->m_x2k2x_line.emplace_back((i+1) * std::floor(this->m_nl/2.));
        }
        this->m_gamma2m_line.reserve(std::floor(this->m_nl/2.)+1);
        for (auto i = 0; i < std::floor(this->m_nl/2.)+1; ++i) {
            this->m_gamma2m_line.emplace_back(i * (std::floor(this->m_nl/2.)+2));
        }
    }

    void SquareLattice::initialize_fourier_factor_table()
    {
        // exp(-ikr) for lattice displacement r and momentum k 
        this->m_fourier_factor_table.resize(this->m_ns, this->m_nkstars);
        this->m_fourier_factor_table.setZero();
        const std::complex<double> id(0., 1.);
        for (auto i = 0; i < this->m_ns; ++i) {
            const auto x = i % this->m_nl;
            const auto y = i / this->m_nl;
            for (auto k = 0; k < this->m_nkstars; ++k) {
                const double kx = this->m_kstars_table(k,0);
                const double ky = this->m_kstars_table(k,1);
                this->m_fourier_factor_table(i,k) = std::cos(x*kx+y*ky) - id * std::sin(x*kx+y*ky);
            }
        }
    }
}