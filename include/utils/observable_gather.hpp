/*
 *   observable_gather.hpp
 * 
 *     Created on: Sep 16, 2023
 *         Author: Jeffery Wang
 *   
 *   This file includes subroutines for collecting instances of Observable::Observable class
 *   among a set of processors through MPI communication.
 */

#pragma once
#ifndef UTILS_OBSERVABLE_GATHER_HPP
#define UTILS_OBSERVABLE_GATHER_HPP

#include <boost/mpi.hpp>
#include <xtensor/xnoalias.hpp>
#include "observable.hpp"
#include "measure_handle.h"
#include "utils/boost_serialization_xtensor.hpp"

namespace Utils {
    namespace MPI {
        
        template <typename _Scalar> using observable = Observable::Observable<_Scalar>;
        template <typename _Scalar> using obs_struct = typename Observable::Observable<_Scalar>::obs_struct;
        using communicator = boost::mpi::communicator;
        using measure_handle = Measurement::Handle;

        template <typename _Scalar>
        void gather_obervables_from_processors(const communicator& world, observable<_Scalar>& obs)
        {
            // rank of the current processor
            const int master = 0;
            const int rank = world.rank();

            // collect observable data from all other processors
            if (rank == master) {
                std::vector<obs_struct<_Scalar>> cache(world.size());
                std::vector<boost::mpi::request> recvs;

                // save the data in master processor to the cache
                cache[0] = *obs.data();

                // for the master processor, receive messages from other processors
                for (auto proc = 1; proc < world.size(); ++proc) {
                    recvs.push_back(world.irecv(proc, proc, cache[proc]));
                }
                boost::mpi::wait_all(recvs.begin(), recvs.end());

                // reset the observable shape
                obs.set_shape(world.size()*obs.nbin(), obs.obsShape());
                
                // gather observable data to the main object
                xt::noalias(*obs.data()) = cache[0];
                for (auto proc = 1; proc < world.size(); ++proc) {
                    *obs.data() = xt::concatenate(xt::xtuple(*obs.data(), cache[proc]), 0);
                }
            }
            else {
                // for all subject processors, send observable data to the master
                std::vector<boost::mpi::request> sends;
                sends.push_back(world.isend(master, rank, *obs.data()));
                boost::mpi::wait_all(sends.begin(), sends.end());
            }
        }

        void gather_measurement_results_from_processors(const communicator& world, measure_handle& meas_handle)
        {
            for (auto& obs : meas_handle.m_obs_list_equaltime) {
                gather_obervables_from_processors(world, *obs);
            }
            for (auto& obs : meas_handle.m_obs_list_dynamic ) {
                gather_obervables_from_processors(world, *obs);
            }
        }

    } // namespace MPI
} // namespace Utils

#endif // UTILS_OBSERVABLE_GATHER_HPP