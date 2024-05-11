/*
 *   dqmc_io.hpp
 * 
 *     Created on: May 11, 2024
 *         Author: Jeffery Wang
 *   
 */

#pragma once
#ifndef DQMC_IO_HPP
#define DQMC_IO_HPP

#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>

#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xnpy.hpp>

#include "dqmc_params.hpp"
#include "dqmc_handle.h"
#include "dqmc_core.h"
#include "measure_handle.h"
#include "hubbard.h"
#include "square_lattice.h"
#include "observable.hpp"

namespace DQMC {
    namespace IO {
        
        // ------------------------------------------------------------------------
        //
        //      Output DQMC information: initialization and summary
        //
        // ------------------------------------------------------------------------
        template <typename _ostream>
        void print_initialization_info(_ostream& ostream, const DQMC::Params& params, const Measurement::Handle& meas_handle, std::size_t nproc)
        {
            if (!ostream) {
                throw std::runtime_error("DQMC::IO::print_initialization_info(): ostream failed.");
            }

            // output formats
            boost::format fmt_param_str   ("%| 30s|%| 7s|%| 20s|\n");
            boost::format fmt_param_int   ("%| 30s|%| 7s|%| 20d|\n");
            boost::format fmt_param_double("%| 30s|%| 7s|%| 20.3f|\n");
            const std::string_view joiner = "->";
            auto bool2str = [](bool b) {if (b) return "True"; else return "False";};

            ostream << ">> Model: HubbardAttractiveU\n\n"
                    << fmt_param_double % "Nearest hopping 'nnt'" % joiner % params.nnt
                    << fmt_param_double % "Chemical potential 'mu'" % joiner % params.mu
                    << fmt_param_double % "On-site interaction 'u'" % joiner % params.u
                    << std::endl;

            ostream << ">> Lattice: SquareLattice\n\n"
                    << fmt_param_int % "Linear size 'nl'" % joiner % params.nl
                    << fmt_param_str % "Momentum list" % joiner % params.momentum_list << std::endl;

            ostream << ">> MonteCarlo Params:\n\n"
                    << fmt_param_double % "Inverse temperature" % joiner % params.beta
                    << fmt_param_int % "Number of imag-time slice " % joiner % params.nt
                    << fmt_param_double % "Imag-time spacing" % joiner % params.dt
                    << fmt_param_int % "Stabilization pace" % joiner % params.stabilization_pace
                    << std::endl;

            ostream << ">> Measurement Params:\n\n"
                    << fmt_param_str % "Warm up" % joiner % bool2str(meas_handle.isWarmUp())
                    << fmt_param_str % "Equal-time measure" % joiner % bool2str(meas_handle.isEqualTime())
                    << fmt_param_str % "Dynamic measure" % joiner % bool2str(meas_handle.isDynamic());
                
            ostream << fmt_param_int % "Sweeps for warmup" % joiner % meas_handle.WarmUpSweeps()
                    << fmt_param_int % "Number of bins" % joiner % (meas_handle.BinsNum() * nproc)
                    << fmt_param_int % "Sweeps per bin" % joiner % meas_handle.BinsCapacity()
                    << fmt_param_int % "Sweeps between bins" % joiner % meas_handle.SweepsBetweenBins()
                    << std::endl;
        }
        
        template <typename _ostream>
        void print_dqmc_summary(_ostream& ostream, const DQMC::Core& core, const Measurement::Handle& meas_handle)
        {
            if (!ostream) {
                throw std::runtime_error("DQMC::IO::print_dqmc_summary(): ostream failed.");
            }
            
            // parse the time duration
            const double duration = static_cast<double>(DQMC::Handle::timer());
            const int day = std::floor(duration/86400000);
            const int hour = std::floor((duration/1000 - day*86400)/3600);
            const int minute = std::floor((duration/1000 - day*86400 - hour*3600)/60);
            const double sec = duration/1000 - 86400*day - 3600*hour - 60*minute;

            // print the time cost of the simulation
            if (day) { ostream << boost::format("\n>> The simulation finished in %d d %d h %d m %.2f s.\n") % day % hour % minute % sec << std::endl; }
            else if (hour) { ostream << boost::format("\n>> The simulation finished in %d h %d m %.2f s.\n") % hour % minute % sec << std::endl; }
            else if (minute) { ostream << boost::format("\n>> The simulation finished in %d m %.2f s.\n") % minute % sec << std::endl; }
            else { ostream << boost::format("\n>> The simulation finished in %.2f s.\n") % sec << std::endl; }

            // print the equal-time/dynamic wrapping errors
            if (meas_handle.isEqualTime() && meas_handle.isDynamic()) {
                ostream << boost::format(">> Maximum of the wrapping error: %.5e (equal-time) and %.5e (dynamic).\n")
                        % core.equaltimeWrapError() % core.dynamicWrapError() << std::endl;
            }
            else if (meas_handle.isEqualTime()) {
                ostream << boost::format(">> Maximum of the wrapping error: %.5e (equal-time).\n") % core.equaltimeWrapError() << std::endl;
            }
            else if (meas_handle.isDynamic()) {
                ostream << boost::format(">> Maximum of the wrapping error: %.5e (dynamic).\n") % core.dynamicWrapError() << std::endl;
            }
        }

        // ------------------------------------------------------------------------
        //
        //      Output measurements of observables
        //
        // ------------------------------------------------------------------------
        template <typename _ostream>
        void print_observable(_ostream& ostream, const Observable::ObservableReal& obs, bool show_header=true)
        {
            if (!ostream) {
                throw std::runtime_error("DQMC::IO::print_observable(): ostream failed.");
            }

            using obs_index = std::size_t;
            using obs_shape = typename Observable::ObservableReal::obs_shape;
            const obs_shape shape = obs.obsShape();
            const obs_index dim = shape.size();
            const obs_index size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<obs_index>());
            obs_shape accumulated = shape;

            boost::format fmt_index("%| 15d|");
            boost::format fmt_data ("%| 30.15f|%| 30.15f|%| 30.15f|");
            if (dim == 0) {
                // 0-dim observable is equivalent to a scalar
                if (show_header) { ostream << fmt_index % 1 << std::endl; }
                ostream << fmt_index % 0 << fmt_data % obs.mean() % obs.stddev() % obs.stderr() << std::endl;
            }
            else {
                // observable with one and higher dimension
                for (auto it = accumulated.begin(); it != accumulated.end(); ++it) {
                    *it = std::accumulate(it+1, accumulated.end(), 1, std::multiplies<obs_index>());
                }
                const auto flatten_mean   = xt::flatten(obs.mean());
                const auto flatten_stddev = xt::flatten(obs.stddev());
                const auto flatten_stderr = xt::flatten(obs.stderr());
                if (show_header) {
                    // header info, i.e. dimensions of the observable
                    for (obs_index d = 0; d < dim; ++d) {
                        ostream << fmt_index % shape[d];
                    }
                    ostream << std::endl;
                }
                for (obs_index i = 0; i < size; ++i) {
                    for (obs_index d = 0; d < dim; ++d) {
                        ostream << fmt_index % ((i/accumulated[d])%shape[d]);
                    }
                    ostream << fmt_data % flatten_mean(i) % flatten_stddev(i) % flatten_stderr(i) << std::endl;
                }
            }
        }

        template <typename _ostream>
        void print_observable_data(_ostream& ostream, const Observable::ObservableReal& obs, bool show_header=true)
        {
            if (!ostream) {
                throw std::runtime_error("DQMC::IO::print_observable_data(): ostream failed.");
            }

            using data_index = std::size_t;
            using data_shape = typename Observable::ObservableReal::data_shape;
            const data_shape shape = obs.dataShape();
            const data_index dim = shape.size();
            const data_index size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<data_index>());
            data_shape accumulated = shape;
            for (auto it = accumulated.begin(); it != accumulated.end(); ++it) {
                *it = std::accumulate(it+1, accumulated.end(), 1, std::multiplies<data_index>());
            }
            const auto flatten_data = xt::flatten(*obs.data());
            boost::format fmt_index("%| 15d|");
            boost::format fmt_data ("%| 30.15f|");
            if (show_header) {
                // header info, i.e. dimensions of the observable
                for (data_index d = 0; d < dim; ++d) {
                    ostream << fmt_index % shape[d];
                }
                ostream << std::endl;
            }
            for (data_index i = 0; i < size; ++i) {
                for (data_index d = 0; d < dim; ++d) {
                    ostream << fmt_index % ((i/accumulated[d]) % shape[d]);
                }
                ostream << fmt_data % flatten_data(i) << std::endl;
            }
        }

        template <typename _Scalar>
        void save_obsevable_data_to_file(const Observable::Observable<_Scalar>& obs, std::string file, std::string fmt="npy")
        {            
            if (fmt == "npy") {
                xt::dump_npy(file, *obs.data());
            }
            else if (fmt == "csv") {
                std::ofstream ofile(file, std::ios::out|std::ios::trunc);
                if (!ofile.is_open()) {
                    throw std::runtime_error(
                        boost::str(boost::format("DQMC::IO::save_observable_data_to_file(): fail to open '%s'.") % file)
                    );
                }
                xt::dump_csv(ofile, *obs.data());
            }
            else {
                throw std::runtime_error("DQMC::IO::save_observable_data_to_file(): invalid type of format.");
            }
        }

        // ------------------------------------------------------------------------
        //
        //      Save/Load configurations of Ising fields to/from file
        //
        // ------------------------------------------------------------------------
        void save_ising_fields_configs(std::string file, const Model::HubbardAttractiveU& model)
        {
            std::ofstream ofile(file, std::ios::out|std::ios::trunc);
            if (!ofile.is_open()) {
                throw std::runtime_error(
                    boost::str(boost::format("DQMC::IO::save_ising_fields_configs(): fail to open '%s'.") % file)
                );
            }

            // output the current configurations of ising fields
            boost::format fmt_fields_info("%| 15d|%| 15d|");
            boost::format fmt_fields("%| 15d|%| 15d|%| 15d|");
            const std::size_t nt = model.m_nt;
            const std::size_t ns = model.m_ns;
            ofile << fmt_fields_info % nt % ns << std::endl;
            for (std::size_t t = 0; t < nt; ++t) {
                for (std::size_t i = 0; i < ns; ++i) {
                    ofile << fmt_fields % t % i % model.m_ising_fields(t,i) << std::endl;
                }
            }
            ofile.close();
        }

        void load_ising_fields_configs(std::string file, Model::HubbardAttractiveU& model)
        {
            std::ifstream ifile(file, std::ios::in);
            if (!ifile.is_open()) {
                throw std::runtime_error(
                    boost::str(boost::format("DQMC::IO::load_ising_fields_configs(): fail to open '%s'.") % file)
                );
            }

            // temporary variables
            std::string line;
            std::vector<std::string> data;
            // consistency check of the config parameters
            // read the first line which contains the dimensions of fields
            getline(ifile, line);
            boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
            data.erase(std::remove(std::begin(data), std::end(data), ""), std::end(data));

            const std::size_t nt = boost::lexical_cast<std::size_t>(data[0]);
            const std::size_t ns = boost::lexical_cast<std::size_t>(data[1]);
            if ((nt != static_cast<std::size_t>(model.m_nt)) || (ns != static_cast<std::size_t>(model.m_ns))) {
                throw std::runtime_error("DQMC::IO::load_ising_fields_configs(): contradictory dimensions of the ising fields.");
            }
            // read the configurations of ising fields
            std::size_t t, i;
            while(getline(ifile, line)) {
                boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
                data.erase(std::remove(std::begin(data), std::end(data), ""), std::end(data));
                t = boost::lexical_cast<std::size_t>(data[0]);
                i = boost::lexical_cast<std::size_t>(data[1]);
                model.m_ising_fields(t,i) = boost::lexical_cast<int>(data[2]);
            }
            ifile.close();
        }

        // ------------------------------------------------------------------------
        //
        //      Output other information
        //
        // ------------------------------------------------------------------------
        template <typename _ostream>
        void print_momentum_list(_ostream& ostream, const Measurement::Handle& handle, const Lattice::SquareLattice& lattice)
        {
            if (!ostream) {
                throw std::runtime_error("DQMC::IO::print_momentum_list(): ostream failed.");
            }
            boost::format fmt_info("%| 15d|%| 15d|");
            boost::format fmt_momentum("%| 15d|%| 15d|%| 30.15f|%| 30.15f|");
            const std::size_t nkstars = lattice.allKstars().size();
            const std::size_t nk = handle.MomentumList().size();
            ostream << fmt_info % nkstars % nk << std::endl;
            for (std::size_t k = 0; k < nk; ++k) {
                const auto momentum_index = handle.MomentumList(k);
                const auto momentum = lattice.momentum(momentum_index);
                ostream << fmt_momentum % k % momentum_index % momentum(0) % momentum(1) << std::endl;
            }
        }

        template <typename _ostream>
        void print_imaginary_time_grids(_ostream& ostream, const DQMC::Params& params)
        {
            if (!ostream) {
                throw std::runtime_error("DQMC::IO::print_imaginary_time_grids(): ostream failed.");
            }
            // output the imaginary-time grids
            boost::format fmt_tgrids_info("%| 15d|%| 30.8f|%| 30.8f|");
            boost::format fmt_tgrids("%| 15d|%| 30.8f|");
            ostream << fmt_tgrids_info % params.nt % params.beta % params.dt << std::endl;
            for (std::size_t t = 0; t < params.nt; ++t) {
                ostream << fmt_tgrids % t % static_cast<double>(t*params.dt) << std::endl;
            }
        }
    }
}

#endif