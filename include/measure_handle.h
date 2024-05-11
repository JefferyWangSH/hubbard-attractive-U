/*
 *   measure_handle.h
 * 
 *     Created on: May 10, 2024
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef MEASURE_HANDLE_H
#define MEASURE_HANDLE_H

#include "observable_handle.h"

namespace DQMC { struct Params; class Core; }
namespace Model { class HubbardAttractiveU; }
namespace Lattice { class SquareLattice; }
namespace boost { namespace mpi { class communicator; } }
namespace Measurement { class Handle; }
namespace Utils { namespace MPI {
    void gather_measurement_results_from_processors(const boost::mpi::communicator&, Measurement::Handle&);
}}

namespace Measurement {

    // ----------------------------------------------  Measurement::Handle  ----------------------------------------------------------
    class Handle : public Observable::Handle {
        private:
            using DqmcCore = DQMC::Core;
            using DqmcParams = DQMC::Params;
            using Hubbard = ::Model::HubbardAttractiveU;
            using Lattice = ::Lattice::SquareLattice;
            using observable_name_list = ::Observable::Handle::observable_name_list;
            using momentum_index = int;
            using momentum_list = std::vector<momentum_index>;

            bool m_is_warmup{};                 // whether to warm up the system
            bool m_is_equaltime{};              // whether to perform equal-time measurements
            bool m_is_dynamic{};                // whether to perform dynamic measurements

            int m_sweeps_warmup{};              // number of the MC sweeps for the warm-up process
            int m_bin_num{};                    // number of bins for measurements 
            int m_bin_capacity{};               // capacity of samples in one bin
            int m_sweeps_between_bins{};        // number of the MC sweeps between two adjoining bins
    
            observable_name_list m_list{};      // list of observables to be measured
            momentum_list m_momentum_list{};     // list of momenta for momentum-dependent observables

        public:
            friend void Utils::MPI::gather_measurement_results_from_processors(const boost::mpi::communicator&, Handle&);
        
            // -------------------------------------------  Interfaces  -----------------------------------------------------

            const bool isWarmUp() const { return this->m_is_warmup; }
            const bool isEqualTime() const { return this->m_is_equaltime; }
            const bool isDynamic() const { return this->m_is_dynamic; }
            const int WarmUpSweeps() const { return this->m_sweeps_warmup; }
            const int SweepsBetweenBins() const { return this->m_sweeps_between_bins; }
            const int BinsNum() const { return this->m_bin_num; }
            const int BinsCapacity() const { return this->m_bin_capacity; }
            const momentum_list& MomentumList() const { return this->m_momentum_list; }
            const momentum_index MomentumList(const int i) const { return this->m_momentum_list[i]; }

            // ----------------------------------------  Initialization  ---------------------------------------------------

            void initialize(const DqmcParams& params, const Lattice& lattice);

            // -----------------------------------  Measurements and statistics  --------------------------------------------
            
            // equal-time / dynamic measurements
            void measure_equaltime_observables(const DqmcCore& core, const Hubbard& model, const Lattice& lattice);
            void measure_dynamic_observables  (const DqmcCore& core, const Hubbard& model, const Lattice& lattice);

            // normalize the observable cache
            void normalize_cache();
            
            // store the cache observable into the data structure
            void push_cache_to_data(const std::size_t bin);

            // perform statistical analysis, e.g. calculating the mean, stddev, and stderr
            void analyse();

            // clear the cache data
            void clear_cache();  
    };
}

#endif