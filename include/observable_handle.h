/*
 *   observable_handle.h
 * 
 *     Created on: May 9, 2024
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef OBSERVABLE_HANDLE_H
#define OBSERVABLE_HANDLE_H

#include <map>
#include <string>
#include <array>
#include <vector>
#include <memory>
#include <set>

namespace Observable {

    template <typename _scalar> class Observable;
    using ObservableReal = Observable<double>;

    // ----------------------------------------------  Observable::Handle  ----------------------------------------------------
    class Handle {
        protected:
            using observable = ObservableReal; // basically observables are real valued.
            using ptr_observable = std::shared_ptr<ObservableReal>;
            using observable_list = std::vector<ptr_observable>;
            using observable_name = std::string;
            using observable_map = std::map<observable_name, ptr_observable>;
            using observable_name_list = std::set<observable_name>;
            using metadata_type = std::array<std::string,2>;

            static std::map<observable_name, metadata_type> m_metadata;
            static bool is_equaltime(const observable_name name);
            static bool is_dynamic(const observable_name name);

            // check the validity of the input observable name
            static bool check_validity(const observable_name name);

            observable_map  m_obs_map{};
            observable_list m_obs_list_equaltime{};
            observable_list m_obs_list_dynamic{};
        
        public:
            // name list of all supported observables (for external calls)
            static observable_name_list allObservables;

            // initialize the handle according to the input observable list
            void initialize(const observable_name_list& list);

            // get the instance of observable by name
            bool is_found(const observable_name name) const;
            const observable find(const observable_name name);
    };
}

#endif