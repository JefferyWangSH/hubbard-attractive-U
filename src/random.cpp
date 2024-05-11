/*
 *   random.cpp
 * 
 *     Created on: Aug 4, 2023
 *         Author: Jeffery Wang
 * 
 */

#include "random.h"

namespace Utils {

    // initialization of the static class member
    std::default_random_engine Random::Engine(time(nullptr));
    void Random::set_seed(const int seed) {Engine.seed(seed);}

} // namespace Utils