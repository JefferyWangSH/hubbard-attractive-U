/*
 *   dqmc_parser.hpp
 * 
 *     Created on: May 10, 2024
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef DQMC_PARSER_HPP
#define DQMC_PARSER_HPP

#include <string_view>
#include <string>
#include <stdexcept>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <boost/format.hpp>

#include "dqmc_params.hpp"
#include "observable_handle.h"
#include "utils/toml.hpp"

namespace DQMC {
    namespace Parser {

        template <typename T>
        void range_check(const T val, const std::string name, const T lower_bound)
        {
            if (val < lower_bound) {
                throw std::runtime_error(
                    boost::str(boost::format("DQMC::Parser::range_check<T>(): '%s' out of range.") % name)
                );
            }
        }

        template <typename T>
        void range_check(const T val, const std::string name, const T lower_bound, const T upper_bound)
        {
            if (val < lower_bound || val > upper_bound) {
                throw std::runtime_error(
                    boost::str(boost::format("DQMC::Parser::range_check<T>(): '%s' out of range.") % name)
                );
            }
        }

        void parse_toml_config(std::string_view toml_config, int world_size, DQMC::Params& params)
        {
            // parse the configuration file
            toml::table config;
            try { config = toml::parse_file(toml_config); }
            catch (const toml::parse_error& err) {
                throw std::runtime_error(
                    boost::str(boost::format("DQMC::Parser::parse_toml_config(): parsing failed.\n%s") % err)
                );
            }

            params.nnt = config["Model"]["HubbardAttractiveU"]["nnt"].value_or(1.0);
            params.u = config["Model"]["HubbardAttractiveU"]["u"].value_or(4.0);
            params.mu = config["Model"]["HubbardAttractiveU"]["mu"].value_or(0.0);

            params.nl = config["Lattice"]["SquareLattice"]["nl"].value_or(4);
            params.ns = params.nl * params.nl;
            params.ng = params.ns;

            auto momentum_list = config["Lattice"]["SquareLattice"]["momentum_list"];
            if (momentum_list.is_string()) {
                params.momentum_list = momentum_list.value_or("");
            }
            else if (momentum_list.is_array()) {
                std::vector<int> momenta;
                toml::array* momentum_arr = momentum_list.as_array();
                if (momentum_arr && momentum_arr->is_homogeneous<int64_t>()) {
                    momenta.reserve(momentum_arr->size());
                    for (auto&& el : *momentum_arr) {
                        momenta.emplace_back(el.value_or(0));
                    }
                    params.momentum_list = momenta;
                }
                else { throw std::runtime_error("DQMC::Parser::parse_toml_config(): invalid input momentum list."); }
            }
            else { throw std::runtime_error("DQMC::Parser::parse_toml_config(): invalid input momentum list."); }

            params.beta = config["MonteCarlo"]["beta"].value_or(8.0);
            params.nt = config["MonteCarlo"]["nt"].value_or(160);
            params.dt = params.beta / params.nt;
            params.stabilization_pace = config["MonteCarlo"]["stabilization_pace"].value_or(10);
            params.is_fft = config["MonteCarlo"]["is_fft"].value_or(false);

            params.sweeps_warmup = config["Measurement"]["sweeps_warmup"].value_or(1000);
            // distribute the measurement tasks to a batch of processors
            params.bin_num = std::ceil(config["Measurement"]["bin_num"].value_or(20) / world_size);
            params.bin_capacity = config["Measurement"]["bin_capacity"].value_or(100);
            params.sweeps_between_bins = config["Measurement"]["sweeps_between_bins"].value_or(10);
            
            std::vector<std::string> observables;
            toml::array* observable_arr = config["Measurement"]["observables"].as_array();
            if (observable_arr && observable_arr->is_homogeneous<std::string>()) {
                observables.reserve(observable_arr->size());
                for (auto&& el : *observable_arr) {
                    observables.emplace_back(el.value_or(""));
                }
            }
            else {
                throw std::runtime_error("DQMC::Parser::parse_toml_config(): invalid input observables.");
            }

            // deal with special keywords (all/All, none/None)
            bool is_all  = (std::find(observables.begin(), observables.end(), "all") != observables.end()
                         || std::find(observables.begin(), observables.end(), "All") != observables.end());
            bool is_none = (std::find(observables.begin(), observables.end(), "none") != observables.end()
                         || std::find(observables.begin(), observables.end(), "None") != observables.end());
            if ( is_all && !is_none) { params.observable_list = Observable::Handle::allObservables; }
            if (!is_all &&  is_none) { params.observable_list = {}; }
            if (!is_all && !is_none) { params.observable_list = std::set<std::string>(observables.begin(), observables.end()); }
            if ( is_all &&  is_none) {
                throw std::runtime_error("DQMC::Parser::parse_toml_config(): "
                "recieved conflict observable options 'all/All' and 'none/None'.");
            }

            // check the range of input parameters
            range_check(params.u, "u", 0.);
            range_check(params.nl, "nl", 1);
            range_check(params.beta, "beta", 0.);
            range_check(params.nt, "nt", 1);
            range_check(params.stabilization_pace, "stabilization_pace", 1);
            range_check(params.sweeps_warmup, "sweeps_warmup", 0);
            range_check(params.bin_num, "bin_num", 0);
            range_check(params.bin_capacity, "bin_capacity", 0);
            range_check(params.sweeps_between_bins, "sweeps_between_bins", 0);
        }
    }
}

#endif