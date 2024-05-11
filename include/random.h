/*
 *   random.h
 * 
 *     Created on: Aug 4, 2023
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef UTILS_RANDOM_H
#define UTILS_RANDOM_H

#include <random>

namespace Utils {

    // ----------------------  Utils::Random class for generating random seeds used in MPI program  ------------------------
    class Random {
        public:
            static std::default_random_engine Engine;

            // explicitly setup seeds for the random engine
            // e.g. set_seed(123) with fixed seed for debug usage
            // or set_seed(time(nullptr)+rank) to setup different seeds for different MPI processors
            static void set_seed(const int seed);
    };

} // namespace Utils

#endif // UTILS_RANDOM_H