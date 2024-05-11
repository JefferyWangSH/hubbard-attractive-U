/*
 *   fft_solver.hpp
 * 
 *     Created on: Aug 13, 2023
 *         Author: Jeffery Wang and Bing AI
 * 
 */

#pragma once
#ifndef UTILS_FFT_SOLVER_HPP
#define UTILS_FFT_SOLVER_HPP

#include <mkl_dfti.h>
#include <complex>
#include <type_traits>
#include <array>

namespace Utils {

    template <typename T, std::size_t N>
    class FFTSolver {
    public:
        FFTSolver(const std::array<std::size_t, N>& sizes) : sizes_(sizes) {
            initialize();
        }

        ~FFTSolver() {
            DftiFreeDescriptor(&descriptor_);
        }

        void resize(const std::array<std::size_t, N>& new_sizes) {
            if (new_sizes != sizes_) {
                sizes_ = new_sizes;
                DftiFreeDescriptor(&descriptor_);
                initialize();
            }
        }

        auto fft(T* in, std::complex<double>* out) const {
            return DftiComputeForward(descriptor_, in, out);
        }

        auto ifft(std::complex<double>* in, T* out) const {
            return DftiComputeBackward(descriptor_, in, out);
        }

    private:
        std::array<std::size_t, N> sizes_;
        DFTI_DESCRIPTOR_HANDLE descriptor_;

        auto initialize() {
            if (sizes_.size() != N) {
                throw std::invalid_argument("Size of input sizes array must be equal to template parameter N");
            }
            MKL_LONG status;
            if (N == 1) {
                MKL_LONG size = static_cast<MKL_LONG>(sizes_[0]);
                double scale = 1.0 / static_cast<double>(sizes_[0]);
                if (std::is_same<T, double>::value) {
                    status = DftiCreateDescriptor(&descriptor_, DFTI_DOUBLE, DFTI_REAL, N, size);
                    status = DftiSetValue(descriptor_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
                } else if (std::is_same<T, std::complex<double>>::value) {
                    status = DftiCreateDescriptor(&descriptor_, DFTI_DOUBLE, DFTI_COMPLEX, N, size);
                } else {
                    throw std::invalid_argument("Template parameter T must be either double or std::complex<double>");
                }
                status = DftiSetValue(descriptor_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                status = DftiSetValue(descriptor_, DFTI_BACKWARD_SCALE, scale);
                status = DftiCommitDescriptor(descriptor_);
            } else {
                MKL_LONG mkl_sizes[N];
                double scale = 1.0;
                for (std::size_t i = 0; i < N; ++i) {
                    mkl_sizes[i] = static_cast<MKL_LONG>(sizes_[i]);
                    scale /= static_cast<double>(sizes_[i]);
                }
                if (std::is_same<T, double>::value) {
                    status = DftiCreateDescriptor(&descriptor_, DFTI_DOUBLE, DFTI_REAL, N, mkl_sizes);
                    status = DftiSetValue(descriptor_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
                } else if (std::is_same<T, std::complex<double>>::value) {
                    status = DftiCreateDescriptor(&descriptor_, DFTI_DOUBLE, DFTI_COMPLEX, N, mkl_sizes);
                } else {
                    throw std::invalid_argument("Template parameter T must be either double or std::complex<double>");
                }
                status = DftiSetValue(descriptor_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                status = DftiSetValue(descriptor_, DFTI_BACKWARD_SCALE, scale);
                status = DftiCommitDescriptor(descriptor_);
                return status;
            }
        }
    };

} // namespace Utils

#endif // UTILS_FFT_SOLVER_H