/*
 *   boost_serialization_xtensor.hpp
 * 
 *     Created on: Sep 13, 2023
 *         Author: Jeffery Wang and Bing AI
 * 
 *   Serialization of xt::xarray and xt::xtensor.
 *   TODO: xt::xtensor_fixed.
 */

#pragma once
#ifndef BOOST_SERIALIZATION_XTENSOR_HPP
#define BOOST_SERIALIZATION_XTENSOR_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/complex.hpp>

namespace boost {
    namespace serialization {

        // ------------------------------  xt::xarray  --------------------------------

        template<class Archive, class T, xt::layout_type L, class SC, class Tag>
        void save(Archive& ar, const xt::xarray<T, L, SC, Tag>& arr, const unsigned int version) {
            auto shape = arr.shape();
            ar & shape;
            ar & boost::serialization::make_array(arr.data(), arr.size());
        }

        template<class Archive, class T, xt::layout_type L, class SC, class Tag>
        void load(Archive& ar, xt::xarray<T, L, SC, Tag>& arr, const unsigned int version) {
            typename xt::xarray<T, L, SC, Tag>::shape_type shape;
            ar & shape;
            arr.resize(shape);
            ar & boost::serialization::make_array(arr.data(), arr.size());
        }

        template<class Archive, class T, xt::layout_type L, class SC, class Tag>
        void serialize(Archive& ar, xt::xarray<T, L, SC, Tag>& arr, const unsigned int version) {
            boost::serialization::split_free(ar, arr, version);
        }


        // ------------------------------  xt::svector  --------------------------------

        template<class Archive, class T>
        void save(Archive& ar, const xt::svector<T>& vec, const unsigned int version) {
            std::size_t size = vec.size();
            ar & size;
            ar & boost::serialization::make_array(vec.data(), vec.size());
        }

        template<class Archive, class T>
        void load(Archive& ar, xt::svector<T>& vec, const unsigned int version) {
            std::size_t size;
            ar & size;
            vec.resize(size);
            ar & boost::serialization::make_array(vec.data(), vec.size());
        }

        template<class Archive, class T>
        void serialize(Archive& ar, xt::svector<T>& vec, const unsigned int version) {
            boost::serialization::split_free(ar, vec, version);
        }


        // ------------------------------  xt::xtensor  --------------------------------

        template<class Archive, class T, std::size_t N, xt::layout_type L, class Tag>
        void save(Archive& ar, const xt::xtensor<T, N, L, Tag>& tensor, const unsigned int version) {
            auto shape = tensor.shape();
            ar & shape;
            ar & boost::serialization::make_array(tensor.data(), tensor.size());
        }

        template<class Archive, class T, std::size_t N, xt::layout_type L, class Tag>
        void load(Archive& ar, xt::xtensor<T, N, L, Tag>& tensor, const unsigned int version) {
            typename xt::xtensor<T, N, L, Tag>::shape_type shape;
            ar & shape;
            tensor.resize(shape);
            ar & boost::serialization::make_array(tensor.data(), tensor.size());
        }

        template<class Archive, class T, std::size_t N, xt::layout_type L, class Tag>
        void serialize(Archive& ar, xt::xtensor<T, N, L, Tag>& tensor, const unsigned int version) {
            boost::serialization::split_free(ar, tensor, version);
        }

    }  // namespace serialization
}  // namespace boost

#endif // BOOST_SERIALIZATION_XTENSOR_HPP