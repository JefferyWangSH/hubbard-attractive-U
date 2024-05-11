/*
 *   svd_stack.hpp
 * 
 *     Created on: Sep 7, 2023
 *         Author: Jeffery Wang
 * 
 *   This head file includes SvdData and SvdStack class
 *   for the stable multiplication of long chains of dense matrices.
 *   BLAS and LAPACK libraries (MKL implemented) are needed for the svd decomposition.
 */

#pragma once
#ifndef UTILS_SVD_STACK_HPP
#define UTILS_SVD_STACK_HPP

#include <complex>
#include <stdexcept>
#include <vector>
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>

#include "utils/linear_algebra.hpp"

namespace Utils {
    
    // --------------------------------------------  Utils::SvdData< Scalar >  ------------------------------------------------
    template< typename Scalar > class SvdData {
        private:
            using uMatrix = Eigen::Matrix< Scalar, Eigen::Dynamic, Eigen::Dynamic >;
            using vMatrix = Eigen::Matrix< Scalar, Eigen::Dynamic, Eigen::Dynamic >;
            using sVector = Eigen::VectorXd;

            uMatrix m_u{};
            vMatrix m_v{};
            sVector m_s{};

        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            SvdData() = default;
            explicit SvdData( int dim ): m_u(dim, dim), m_v(dim, dim), m_s(dim) {}

            uMatrix& MatrixU()        { return this->m_u; }
            vMatrix& MatrixV()        { return this->m_v; }
            sVector& SingularValues() { return this->m_s; }
    };

    // -------------------------------------------  Utils::SvdStack< Scalar >  ------------------------------------------------
    // udv stack for a chain of matrix products: u * d * vt = An * ... * A2 * A1 * A0
    template< typename Scalar > class SvdStack {
        private:
            using SvdDataVector = std::vector< SvdData<Scalar> >;
            using Matrix = Eigen::Matrix< Scalar, Eigen::Dynamic, Eigen::Dynamic >;
            using Vector = Eigen::VectorXd;

            SvdDataVector m_stack{};        // matrix stacks
            int m_mat_dim{};                // dimension of the matrices in the stack
            int m_stack_length{0};          // curreng length of the stack
            Matrix m_tmp_matrix{};          // intermediate variable for computing the svd

        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            // ----------------------------------------  Constructions  ------------------------------------------------
            SvdStack() = default;

            explicit SvdStack( int mat_dim, int stack_length ) 
                : m_mat_dim(mat_dim), m_tmp_matrix(mat_dim, mat_dim)
            {
                this->m_stack.reserve( stack_length );
                for ( auto i = 0; i < stack_length; ++i ) {
                    this->m_stack.emplace_back( mat_dim );
                }
            }

            // ------------------------------------------  Interfaces  -------------------------------------------------
            bool empty() const { return this->m_stack_length == 0; }
            int MatDim() const { return this->m_mat_dim; }
            int StackLength() const { return this->m_stack_length; }

            // return the udv decomposition matrices of the current stack
            const Vector SingularValues()
            {
                assert( this->m_stack_length > 0 );
                return this->m_stack[this->m_stack_length-1].SingularValues();
            }

            const Matrix MatrixU()
            {
                assert( this->m_stack_length > 0 );
                return this->m_stack[this->m_stack_length-1].MatrixU();
            }

            const Matrix MatrixV()
            {
                assert( this->m_stack_length > 0 );
                Matrix r = this->m_stack[0].MatrixV();
                for ( auto i = 1; i < this->m_stack_length; ++i ) {
                    r = r * this->m_stack[i].MatrixV();
                }
                return r;
            }

            // ---------------------------  Stack operations push(), pop() and clear()  --------------------------------
            // multply a matrix to the stack from the left
            void push( const Matrix& matrix )
            {
                assert( matrix.rows() == this->m_mat_dim && matrix.cols() == this->m_mat_dim );
                assert( this->m_stack_length < (int)this->m_stack.size() );

                if ( this->m_stack_length == 0 ) {
                    if constexpr ( std::is_same_v< Scalar, double > ) {
                        // SVD decomposition of real matrix
                        Utils::LinearAlgebra::mkl_lapack_dgesvd
                        (
                            this->m_mat_dim, 
                            this->m_mat_dim, 
                            matrix, 
                            this->m_stack[this->m_stack_length].MatrixU(), 
                            this->m_stack[this->m_stack_length].SingularValues(), 
                            this->m_stack[this->m_stack_length].MatrixV()
                        );
                    }
                    else if constexpr ( std::is_same_v< Scalar, std::complex<double> > ) {
                        // SVD decomposition of complex matrix
                        Utils::LinearAlgebra::mkl_lapack_zgesvd
                        (
                            this->m_mat_dim, 
                            this->m_mat_dim, 
                            matrix, 
                            this->m_stack[this->m_stack_length].MatrixU(), 
                            this->m_stack[this->m_stack_length].SingularValues(), 
                            this->m_stack[this->m_stack_length].MatrixV()
                        );
                    }
                    else {
                        throw std::invalid_argument("Utils::SvdStack<Scalar>::push(): invalid scalar type.");
                    }
                }
                else {
                    // important: mind the order of multiplication.
                    // avoid mixing of different numerical scales here
                    this->m_tmp_matrix = ( matrix * this->MatrixU() ) * this->SingularValues().asDiagonal();

                    if constexpr ( std::is_same_v< Scalar, double > ) {
                        // SVD decomposition of real matrix
                        Utils::LinearAlgebra::mkl_lapack_dgesvd
                        (
                            this->m_mat_dim, 
                            this->m_mat_dim, 
                            this->m_tmp_matrix, 
                            this->m_stack[this->m_stack_length].MatrixU(), 
                            this->m_stack[this->m_stack_length].SingularValues(), 
                            this->m_stack[this->m_stack_length].MatrixV()
                        );
                    }
                    else if constexpr ( std::is_same_v< Scalar, std::complex<double> > ) {
                        // SVD decomposition of complex matrix
                        Utils::LinearAlgebra::mkl_lapack_zgesvd
                        (
                            this->m_mat_dim, 
                            this->m_mat_dim, 
                            this->m_tmp_matrix, 
                            this->m_stack[this->m_stack_length].MatrixU(), 
                            this->m_stack[this->m_stack_length].SingularValues(), 
                            this->m_stack[this->m_stack_length].MatrixV()
                        );
                    }
                    else {
                        throw std::invalid_argument("Utils::SvdStack<Scalar>::push(): invalid scalar type.");
                    }
                }
                this->m_stack_length += 1;
            }

            // remove the most left matrix from the stack
            void pop() 
            {
                // notice that the memory is not actually released
                assert( this->m_stack_length > 0 );
                this->m_stack_length -= 1;
            }

            // clear the stack
            // simply set stack_length = 0, note that the memory is not really deallocated.
            void clear() { this->m_stack_length = 0; }
    };

    // some aliases
    using SvdStackReal = SvdStack<double>;
    using SvdStackCpx  = SvdStack<std::complex<double>>;

} // namespace Utils

#endif // UTILS_SVD_STACK_HPP