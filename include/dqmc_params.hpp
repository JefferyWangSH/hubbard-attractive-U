/*
 *   dqmc_params.hpp
 * 
 *     Created on: May 8, 2024
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef DQMC_PARAMS_H
#define DQMC_PARAMS_H

#include <set>
#include <string>

namespace DQMC {

    // --------------------------------------------  DQMC::Params  -------------------------------------------------
    struct Params {
        int nl{};               // linear size of lattice
        int ns{};               // total spatial sites
        int ng{};               // dimension of green's function

        int nt{};               // imaginary-time slices
        double dt{};            // imaginary-time spacing
        double beta{};          // inverse temperature

        double nnt{};           // nearest-neighbor (NN) hopping (absolute value)
        double mu{};            // chemical_potential
        double u{};             // Hubbard on-site interaction

        int sweeps_warmup{};        // local MC sweeps for the warmup
        int stabilization_pace{};   // pace of numerical stabilization

        std::set<std::string> observable_list{};        // list of observables to be measured
        int bin_num{};                                  // number of bins for measurements
        int bin_capacity{};                             // capcity of MC samples in one bin
        int sweeps_between_bins{};                      // MC sweeps between two adjoining bins

        std::string momentum_list{};                    // lattice momenta for momentum-dependent observables
    };
}

#endif