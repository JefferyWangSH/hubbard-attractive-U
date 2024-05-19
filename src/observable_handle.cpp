/*
 *   observable_handle.cpp
 * 
 *     Created on: May 9, 2024
 *         Author: Jeffery Wang
 * 
 */

#include "observable_handle.h"
#include "observable.hpp"
#include "observable_methods.hpp"
#include <stdexcept>
#include <ostream>

namespace Observable {

    std::map<Handle::observable_name, Handle::metadata_type> Handle::m_metadata = {
        {"FillingNumber",                         {"Filling number",                            "equaltime"}},
        {"DoubleOccupation",                      {"Double occupation",                         "equaltime"}},
        {"StaticSWavePairingCorrelation",         {"Static s-wave pairing correlation",         "equaltime"}},
        {"DynamicGreenFunctions",                 {"Dynamic Green's functions",                 "dynamic"  }},
        {"LocalDensityOfStates",                  {"Local density of states",                   "dynamic"  }},
        {"LocalDynamicTransverseSpinCorrelation", {"Local dynamic transverse spin correlation", "dynamic"  }},
    };

    Handle::observable_name_list Handle::allObservables = {
        "FillingNumber",
        "DoubleOccupation",
        "StaticSWavePairingCorrelation",
        "DynamicGreenFunctions",
        "LocalDensityOfStates",
        "LocalDynamicTransverseSpinCorrelation",
    };

    bool Handle::check_validity(const observable_name name)
    {
        return (Handle::allObservables.find(name) != Handle::allObservables.end());
    }

    bool Handle::is_equaltime(const Handle::observable_name name) {
        if (!Handle::check_validity(name)) {
            throw std::invalid_argument("Observable::Handle::is_equaltime(): invalid observable name.");
        }
        return (Handle::m_metadata[name][1] == "equaltime");
    }

    bool Handle::is_dynamic(const Handle::observable_name name) {
        if (!Handle::check_validity(name)) {
            throw std::invalid_argument("Observable::Handle::is_dynamic(): invalid observable name.");
        }
        return (Handle::m_metadata[name][1] == "dynamic");
    }

    void Handle::initialize(const observable_name_list& list)
    {
        // release the pointers and clear the observable list/map (in case) 
        for (auto& it  : this->m_obs_map) { it.second.reset(); }
        for (auto& ptr : this->m_obs_list_equaltime) { ptr.reset(); }
        for (auto& ptr : this->m_obs_list_dynamic) { ptr.reset(); }
        this->m_obs_map.clear();
        this->m_obs_list_equaltime.clear();
        this->m_obs_list_dynamic.clear();
        this->m_obs_list_equaltime.shrink_to_fit();
        this->m_obs_list_dynamic.shrink_to_fit();

        // redundant objects are automatically removed due to the input list type (std::set)
        // check the validity of the input
        for (const auto& obsname : list) {
            if (!this->check_validity(obsname)) {
                throw std::invalid_argument("Observable::Handle::initialize(): "
                "received invalid observable from the input.");
            }
        }

        // create instances for observables
        for (const auto& obsname : list) {
            ptr_observable ptrobs = std::make_shared<observable>();
            const auto obsdesc = Handle::m_metadata[obsname][0];
            ptrobs->set_name_and_desc(obsname, obsdesc);

            // link to methods
            if (obsname == "FillingNumber"                        ) { ptrobs->link2method(Methods::measure_filling_number); }
            if (obsname == "DoubleOccupation"                     ) { ptrobs->link2method(Methods::measure_double_occupation); }
            if (obsname == "DynamicGreenFunctions"                ) { ptrobs->link2method(Methods::measure_dynamic_green_functions); }
            if (obsname == "LocalDensityOfStates"                 ) { ptrobs->link2method(Methods::measure_local_density_of_states); }
            if (obsname == "StaticSWavePairingCorrelation"        ) { ptrobs->link2method(Methods::measure_static_swave_pairing_correlation); }
            if (obsname == "LocalDynamicTransverseSpinCorrelation") { ptrobs->link2method(Methods::measure_local_dynamic_transverse_spin_correlation); }

            if (Handle::is_equaltime(obsname)) { this->m_obs_list_equaltime.emplace_back(ptrobs); }
            if (Handle::is_dynamic(obsname)) { this->m_obs_list_dynamic.emplace_back(ptrobs); }
            this->m_obs_map[obsname] = ptrobs;
        }
    }

    bool Handle::is_found(const Handle::observable_name name) const
    {
        return (this->m_obs_map.find(name) != this->m_obs_map.end());
    }

    const Handle::observable Handle::find(const observable_name name)
    {
        if (this->is_found(name)) {
            // if the observable is found in the map, then return the copy
            return *(this->m_obs_map[name]);
        }
        else {
            // else return an empty observable object
            std::cerr << "Warning at Observable::Handle::find(): "
                      << "no observable found with given name, and note that an empty Observable object was returned."
                      << std::endl;
            return observable();
        }
    }
}