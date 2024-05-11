/*
 *   observable.hpp
 * 
 *     Created on: Sep 13, 2023
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef OBSERVABLE_HPP
#define OBSERVABLE_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xbroadcast.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xio.hpp>

#include <algorithm>
#include <functional>

namespace Model { class HubbardAttractiveU; }
namespace Lattice { class SquareLattice; }
namespace DQMC { class Core; }
namespace Measurement { class Handle; }

namespace Observable {

    // -------------------------------------------  Observable::Observable<_scalar>  -------------------------------------------------
    template <typename _scalar>
    class Observable
    {
        public:
            using data_struct = xt::xarray<_scalar>;
            using obs_struct = xt::xarray<_scalar>;
            using data_shape = typename xt::xarray<_scalar>::shape_type;
            using obs_shape = typename xt::xarray<_scalar>::shape_type;
            using cache_view = xt::xstrided_slice_vector;

            using Lattice = Lattice::SquareLattice;
            using Hubbard = Model::HubbardAttractiveU;
            using MeasureHandle = Measurement::Handle;
            using DqmcCore = DQMC::Core;
            using measure_method = void (obs_struct&, const DqmcCore&, const MeasureHandle&, const Hubbard&, const Lattice&);
        
        protected:
            // shape of data and observable
            std::size_t m_nbin{};
            obs_shape   m_obs_shape{};
            data_shape  m_data_shape{};
            
            // data structures
            data_struct* m_data{};
            obs_struct* m_mean{};
            obs_struct* m_stddev{};
            obs_struct* m_stderr{};
            obs_struct* m_cache{};

            // name and description to the observable
            std::string m_name{};
            std::string m_desc{};
            std::size_t m_count{};

            // customized method for the observable measurement
            std::function<measure_method> m_method{};

        public:
            // ----------------------------------  Construction and deconstruction  ------------------------------------------
            Observable()
            {
                this->m_data = new data_struct();
                this->m_mean = new obs_struct();
                this->m_stddev = new obs_struct();
                this->m_stderr = new obs_struct();
                this->m_cache = new obs_struct();
            }

            Observable(const Observable<_scalar>& obs)
            {
                this->m_nbin = obs.m_nbin;
                this->m_obs_shape = obs.m_obs_shape;
                this->m_data_shape = obs.m_data_shape;
                this->m_data = new data_struct();
                this->m_mean = new obs_struct();
                this->m_stddev = new obs_struct();
                this->m_stderr = new obs_struct();
                this->m_cache = new obs_struct();
                xt::noalias(*this->m_data) = *obs.m_data;
                xt::noalias(*this->m_mean) = *obs.m_mean;
                xt::noalias(*this->m_stddev) = *obs.m_stddev;
                xt::noalias(*this->m_stderr) = *obs.m_stderr;
                xt::noalias(*this->m_cache) = *obs.m_cache;
                this->m_name = obs.m_name;
                this->m_desc = obs.m_desc;
                this->m_count = this->m_count;
                this->m_method = obs.m_method;
            }

            ~Observable()
            {
                delete this->m_data;
                delete this->m_mean;
                delete this->m_stddev;
                delete this->m_stderr;
                delete this->m_cache;
            }

            // --------------------------------------------  Interfaces  -----------------------------------------------------
            std::size_t& counts() { return this->m_count; }
            std::size_t  counts() const { return this->m_count; }
            const std::string name() const { return this->m_name; }
            const std::string description() const { return this->m_desc; }
            std::size_t nbin() const { return this->m_nbin; }
            const obs_shape obsShape() const { return this->m_obs_shape; }

            // access to the bin data
            data_struct* data() { return this->m_data; }
            const data_struct* data() const { return this->m_data; }
            const std::size_t dataSize() { return  this->m_data->size(); }
            const data_shape& dataShape() const { return this->m_data_shape; }

            // statistical mean value and error bar
            // DANGEROUS! MIND EMPTY POINTERS!
            const obs_struct& mean() const { return *this->m_mean; }
            const obs_struct& stddev() const { return *this->m_stddev; }
            const obs_struct& stderr() const { return *this->m_stderr; }
            obs_struct& cache() const { return *this->m_cache; }

            // ------------------------------------------  Setup functions  --------------------------------------------------
            void set_shape(const std::size_t nbin, const obs_shape& obs_shape)
            {
                this->m_nbin = nbin;
                this->m_obs_shape = obs_shape;

                std::vector<std::size_t> shape;
                shape.reserve(obs_shape.size()+1);
                shape.emplace_back(nbin);
                shape.insert(shape.end(), obs_shape.begin(), obs_shape.end());
                this->m_data_shape = shape;

                // shape the data structure
                xt::noalias(*this->m_data) = xt::zeros<_scalar>(this->m_data_shape);
                xt::noalias(*this->m_mean) = xt::zeros<_scalar>(this->m_obs_shape); // optional
                xt::noalias(*this->m_stddev) = xt::zeros<_scalar>(this->m_obs_shape); // optional
                xt::noalias(*this->m_stderr) = xt::zeros<_scalar>(this->m_obs_shape); // optional
                xt::noalias(*this->m_cache) = xt::zeros<_scalar>(this->m_obs_shape);
            }

            void set_name_and_desc(const std::string name, const std::string desc)
            {
                this->m_name = name;
                this->m_desc = desc;
            }
            
            void link2method(const std::function<measure_method>& method) { this->m_method = method; }

            // --------------------------------  Observable measurements and statistics  -------------------------------------
            // customized method for the measurement of observable ( core method )
            void measure(obs_struct& cache, const DqmcCore& core,
                const MeasureHandle& handle, const Hubbard& model, const Lattice& lattice)
            {
                this->m_method(cache, core, handle, model, lattice);
                this->m_count++;
            }
            
            // normalize the cache data by the counting number
            void normalize_cache()
            {
                if ( this->m_count > 0 ) {
                    *this->m_cache /= this->m_count;
                }
                else {
                    throw std::runtime_error(
                        "Observable::Observable<_scalar>::normalize_cache(): cache divided by zero."
                    );
                }
            }

            // store the cache observable into the data structure
            void push_cache_to_data(const std::size_t n)
            {
                if ( n >= 0 && n < this->m_nbin ) {
                    cache_view view({n});
                    for ( std::size_t i = 0; i < this->m_obs_shape.size(); ++i ) {
                        view.push_back(xt::all());
                    }
                    xt::strided_view(*this->m_data, view) = *this->m_cache;
                }
                else {
                    throw std::out_of_range(
                        "Observable::Observable<_scalar>::push_cache_to_data(): index of bin out of range."
                    );
                }
            }

            // perform the statistical analysis, especially evaluating the mean, stddev, and stderr
            void analyse()
            {
                // clear previous statistical results
                this->clear_stats();
                
                // evaluate the mean, stddev, and stderr with respect to the first dimension (bins)
                xt::noalias(*this->m_mean) = xt::mean(*this->m_data, {0});
                xt::noalias(*this->m_stddev) = xt::stddev(*this->m_data, {0});
                xt::noalias(*this->m_stderr) = *this->m_stddev / std::sqrt(this->m_nbin);
            }

            // clear the statistical data, preparing for a new measurement
            void clear_stats()
            {
                xt::noalias(*this->m_mean) = xt::zeros<_scalar>(this->m_obs_shape);
                xt::noalias(*this->m_stddev) = xt::zeros<_scalar>(this->m_obs_shape);
                xt::noalias(*this->m_stderr) = xt::zeros<_scalar>(this->m_obs_shape);
            }
            
            // clear the cache data
            void clear_cache()
            {
                xt::noalias(*this->m_cache) = xt::zeros<_scalar>(this->m_obs_shape);
                this->m_count = 0;
            }

            // clear data (QMC bin samples of the observable measurements)
            void clear_data() { xt::noalias(*this->m_data) = xt::zeros<_scalar>(this->m_data_shape); }
    };

    // some aliases
    using ObservableReal = Observable<double>;
    using ObservableCpx  = Observable<std::complex<double>>;
}

#endif