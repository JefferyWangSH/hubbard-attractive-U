/*
 *   linear_algebra.hpp
 * 
 *     Created on: Apr 12, 2023
 *         Author: Jeffery Wang
 * 
 *   This source file includes some diagonalizing tools with C++/Eigen interface
 *   for diagonalizing (long) real and (long) complex matrices using MKL-implemented lapack.
 *   including:
 *
 *     1. SVD decomposition for generic M * N real/complex matrix ( dgesvd and zgesvd )
 *     2. optimized diagonalizing algorithms for N * N real symmetric and Hermitian matrix ( dsyev and zheev )
 *
 *   the numerical accuracy and efficiency of the calculations are guaranteed.
 */

#pragma once
#ifndef UTILS_LINEAR_ALGEBRA_HPP
#define UTILS_LINEAR_ALGEBRA_HPP

#include <stdexcept>
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>
#include <mkl_lapacke.h>

namespace Utils {
    // ---------------------------------------  Utils::LinearAlgebra class  ------------------------------------------------
    class LinearAlgebra {
        public:

            /*
             *  SVD decomposition of an arbitrary M * N real matrix, using LAPACKE_dgesvd:
             *
             *       A  ->  U * S * V^T
             *
             *  note that V is returned in this subroutine, not V transposed.
             *
             *  @param row -> number of rows.
             *  @param col -> number of cols.
             *  @param mat -> an arbitrary `row` * `col` real matrix to be solved.
             *  @param u   -> `row` * `min(row,col)` orthogonal u matrix of type Eigen::MatrixXd.
             *  @param s   -> `min(row,col)` eigenvalues s of type Eigen::VectorXd, sorted in descending order.
             *  @param v   -> `min(row,col)` * `col` orthogonal v matrix of type Eigen::MatrixXd.
             */
            static void mkl_lapack_dgesvd( const int row,
                                           const int col,
                                           const Eigen::MatrixXd& mat,
                                           Eigen::MatrixXd& u,
                                           Eigen::VectorXd& s,
                                           Eigen::MatrixXd& v  )
            {
                assert(row == mat.rows());
                assert(col == mat.cols());

                // info of the input matrix
                MKL_INT matrix_layout, lda, sdim;
                MKL_INT info, ldu = row, ldvt = col;
                if (row >= col) { matrix_layout = LAPACK_COL_MAJOR; lda = row; sdim = col; }
                else { matrix_layout = LAPACK_ROW_MAJOR; lda = col; sdim = row; }

                // local arrays
                double* temp_s = new double[sdim];
                double* temp_u = new double[ldu * row];
                double* temp_vt = new double[ldvt * col];
                double* temp_a = new double[row * col];
                double* temp_superb = new double[sdim-1];

                // convert the eigen matrix to c-style array
                if (row >= col) { Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(&temp_a[0], row, col) = mat; }
                else { Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(&temp_a[0], row, col) = mat; }

                // compute SVD
                info = LAPACKE_dgesvd(matrix_layout, 'S', 'S', row, col, temp_a, lda, temp_s, temp_u, ldu, temp_vt, ldvt, temp_superb);

                // check for convergence
                if (info > 0) {
                    throw std::runtime_error("Utils::LinearAlgebra::mkl_lapack_dgesvd(): "
                    "the algorithm failed to converge.");
                }

                // convert the results into the Eigen style
                s = Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>>(temp_s, 1, sdim);
                if (row >= col) {
                    u = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(temp_u, row, sdim);
                    v = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(temp_vt, col, sdim);
                }
                else {
                    u = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(temp_u, row, sdim);
                    v = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(temp_vt, col, sdim);
                }

                // release memory of local arrays
                delete [] temp_s;
                delete [] temp_u;
                delete [] temp_vt;
                delete [] temp_a;
                delete [] temp_superb;
            }

            /*
             *  SVD decomposition of an arbitrary M * N complex matrix, using LAPACKE_zgesvd:
             *
             *       A  ->  U * S * V^H
             *
             *  note that V is returned in this subroutine, not V conjugate transposed.
             *
             *  @param row -> number of rows.
             *  @param col -> number of cols.
             *  @param mat -> an arbitrary `row` * `col` complex matrix to be solved.
             *  @param u   -> `row` * `min(row,col)` unitary u matrix of type Eigen::MatrixXcd, 
             *  @param s   -> `min(row,col)` eigenvalues s of type Eigen::VectorXd, which are real, non-negative and sorted in descending order.
             *  @param v   -> `min(row,col)` * `col` unitary v matrix of type Eigen::MatrixXcd
             */
            static void mkl_lapack_zgesvd( const int row,
                                           const int col,
                                           const Eigen::MatrixXcd& mat,
                                           Eigen::MatrixXcd& u,
                                           Eigen::VectorXd& s,
                                           Eigen::MatrixXcd& v  )
            {
                assert(row == mat.rows());
                assert(col == mat.cols());

                // info of the input matrix
                MKL_INT matrix_layout, lda, sdim;
                MKL_INT info, ldu = row, ldvh = col;
                if (row >= col) { matrix_layout = LAPACK_COL_MAJOR; lda = row; sdim = col; }
                else { matrix_layout = LAPACK_ROW_MAJOR; lda = col; sdim = row; }

                // local arrays
                double* temp_s = new double[sdim];
                double* temp_superb = new double[sdim-1];
                MKL_Complex16* temp_a = new MKL_Complex16[row * col];
                MKL_Complex16* temp_u = new MKL_Complex16[ldu * row];
                MKL_Complex16* temp_vh = new MKL_Complex16[ldvh * col];

                // convert the eigen matrix to c-style array
                if (row >= col) { Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(reinterpret_cast<std::complex<double>*>(temp_a), row, col) = mat; }
                else { Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(reinterpret_cast<std::complex<double>*>(temp_a), row, col) = mat; }

                // compute SVD
                info = LAPACKE_zgesvd(matrix_layout, 'S', 'S', row, col, temp_a, lda, temp_s, temp_u, ldu, temp_vh, ldvh, temp_superb);

                // check for convergence
                if (info > 0) {
                    throw std::runtime_error("Utils::LinearAlgebra::mkl_lapack_zgesvd(): "
                    "the algorithm failed to converge.");
                }

                // convert the results into the Eigen style
                s = Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>>(temp_s, 1, sdim);
                if (row >= col) {
                    u = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(reinterpret_cast<std::complex<double>*>(temp_u), row, sdim);
                    v = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(reinterpret_cast<std::complex<double>*>(temp_vh), col, sdim).conjugate();
                }
                else {
                    u = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(reinterpret_cast<std::complex<double>*>(temp_u), row, sdim);
                    v = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(reinterpret_cast<std::complex<double>*>(temp_vh), col, sdim).conjugate();
                }

                // release memory of local arrays
                delete [] temp_s;
                delete [] temp_superb;
                delete [] temp_a;
                delete [] temp_u;
                delete [] temp_vh;
            }

            /*
             *  Given an arbitrary N * N real symmetric matrix, calculate the eigenvalues and eigenstates using LAPACKE_dsyev
             *
             *        A  ->  T^\dag * S * T
             *
             *  where T is the rotation matrix, which contains orthogonal eigenvectors;
             *  and S is a diagonal matrix with the eigenvalues being diagonal elements.
             *
             *  @param N   -> number of rows/cols.
             *  @param mat -> an arbitrary `N` * `N` real symmetric matrix to be solved.
             *  @param s   -> diagonal eigen matrix.
             *  @param t   -> rotation matrix, whose columns are the corresponding eigenstates.
             */
            static void mkl_lapack_dsyev( const int N,
                                          const Eigen::MatrixXd& mat,
                                          Eigen::VectorXd& s,
                                          Eigen::MatrixXd& t  )
            {
                assert(mat.rows() == N);
                assert(mat.cols() == N);
                // make sure the input matrix is symmetric
                const double tolerance = 1e-12;
                if (!mat.isApprox(mat.transpose(), tolerance)) {
                    throw std::runtime_error("Utils::LinearAlgebra::mkl_lapack_dsyev(): "
                    "the input matrix is not symmetric under the given tolerance 1e-12.");
                }

                // locals params
                MKL_INT matrix_layout = LAPACK_ROW_MAJOR;
                MKL_INT n = N, lda = N, info;
                double* temp_s = new double[n];
                double* temp_a = new double[lda * n];

                // convert the eigen matrix to c-style array
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(temp_a, lda, n) = mat;

                // solve the eigen problem
                info = LAPACKE_dsyev(matrix_layout, 'V', 'U', n, temp_a, lda, temp_s);

                // check for convergence
                if (info > 0) {
                    throw std::runtime_error("Utils::LinearAlgebra::mkl_lapack_dsyev(): "
                    "the algorithm failed to converge.");
                }

                // convert the results into the Eigen style
                s = Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>>(temp_s, 1, n);
                t = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(temp_a, n, n);

                // release memory of local arrays
                delete [] temp_s;
                delete [] temp_a;
            }

            /*
             *  Given an arbitrary N * N Hermitian matrix, calculate the eigenvalues and eigenstates using LAPACKE_zheev
             *
             *        A  ->  T^H * S * T
             *
             *  where T is the rotation matrix, which contains orthogonal eigenvectors;
             *  and S is a diagonal matrix with the eigenvalues being diagonal elements.
             *
             *  @param N   -> number of rows/cols.
             *  @param mat -> an arbitrary `N` * `N` Hermitian matrix to be solved.
             *  @param s   -> diagonal eigen matrix.
             *  @param t   -> rotation matrix, whose columns are the corresponding eigenstates.
             */
            static void mkl_lapack_zheev( const int N,
                                          const Eigen::MatrixXcd& mat,
                                          Eigen::VectorXd& s,
                                          Eigen::MatrixXcd& t  )
            {
                assert(mat.rows() == N);
                assert(mat.cols() == N);
                // make sure the input matrix is Hermitian
                const double tolerance = 1e-12;
                if (!mat.isApprox(mat.adjoint(), tolerance)) {
                    throw std::runtime_error("Utils::LinearAlgebra::mkl_lapack_zheev(): "
                    "the input matrix is not Hermitian under the given tolerance 1e-12.");
                }

                // locals params
                MKL_INT matrix_layout = LAPACK_ROW_MAJOR;
                MKL_INT n = N, lda = N, info;
                double* temp_s = new double[n];
                MKL_Complex16* temp_a = new MKL_Complex16[lda * n];

                // convert the eigen matrix to c-style array
                Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(reinterpret_cast<std::complex<double>*>(temp_a), lda, n) = mat;

                // solve the eigen problem
                info = LAPACKE_zheev(matrix_layout, 'V', 'U', n, temp_a, lda, temp_s);

                // check for convergence
                if (info > 0) {
                    throw std::runtime_error("Utils::LinearAlgebra::mkl_lapack_zheev(): "
                    "the algorithm failed to converge.");
                }

                // convert the results into the Eigen style
                s = Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>>(temp_s, 1, n);
                t = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(reinterpret_cast<std::complex<double>*>(temp_a), n, n).conjugate();
            
                // release memory of local arrays
                delete [] temp_s;
                delete [] temp_a;
            }
    };

} // namespace Utils

#endif // UTILS_LINEAR_ALGEBRA_HPP