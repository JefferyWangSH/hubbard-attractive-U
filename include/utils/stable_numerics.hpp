/*
 *   stable_numerics.hpp
 * 
 *     Created on: Sep 8, 2023
 *         Author: Jeffery Wang
 * 
 *   This head file defines the static class Utils::StableNumerics,
 *   containing subroutines for computing equal-time and time-displaced (dynamic) Green's functions
 *   from SVD stacks in a stable manner.
 */

#pragma once
#ifndef UTILS_STABLE_NUMERICS_HPP
#define UTILS_STABLE_NUMERICS_HPP

#include <stdexcept>
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include "utils/svd_stack.hpp"

namespace Utils {

    // -------------------------------------------------  Utils::StableNumerics class  -------------------------------------------------------
    class StableNumerics {
        public:

            template<typename Scalar>
            using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;            
            using Vector = Eigen::VectorXd;

            // Subroutine to return the maximum difference of two matrices with the same size.
            // Input: mat1, mat2
            // Output: the maximum difference -> error
            template<typename Scalar>
            static void matrix_compare_error( const Matrix<Scalar>& mat1, const Matrix<Scalar>& mat2, double& error )
            {
                assert( mat1.rows() == mat2.rows() );
                assert( mat1.cols() == mat2.cols() );
                if constexpr ( std::is_same_v< Scalar, double > ) { error = ( mat1-mat2 ).maxCoeff(); }
                else if constexpr ( std::is_same_v< Scalar, std::complex<double> > ) { error = ( mat1-mat2 ).cwiseAbs2().cwiseSqrt().maxCoeff(); }
                else {
                    throw std::invalid_argument("Utils::StableNumerics::matrix_compare_error<Scalar>(): "
                    "invalid scalar type, supporting double and complex double only.");
                }
            }


            // Subroutine to perform the decomposition of a vector, dvec = dmax * dmin,
            // to ensure all elements that greater than one are stored in dmax,
            // and all elements that less than one are stored in dmin.
            // Input:  dvec
            // Output: dmax, dmin
            static void div_dvec_max_min( const Vector& dvec, Vector& dmax, Vector& dmin )
            {
                assert( dvec.size() == dmax.size() );
                assert( dvec.size() == dmin.size() );
                assert( dvec.minCoeff() >= 0.0 );
                Vector ones = Vector::Ones( dvec.size() );
                dmax = dvec.cwiseMax( ones );
                dmin = dvec.cwiseMin( ones );
            }


            // Subroutine to perform dense matrix * (diagonal matrix)^-1 * dense matrix
            // Input:  vmat, dvec, umat
            // Output: zmat
            template<typename Scalar>
            static void mult_v_invd_u( const Matrix<Scalar>& vmat, const Vector& dvec, const Matrix<Scalar>& umat, Matrix<Scalar>& zmat )
            {
                assert( vmat.cols() == umat.cols() );
                assert( vmat.cols() == zmat.cols() );
                assert( vmat.rows() == umat.rows() );
                assert( vmat.rows() == zmat.rows() );
                assert( vmat.rows() == vmat.cols() );
                assert( vmat.cols() == dvec.size() );

                const int ndim = static_cast<int>(vmat.rows());
                for ( int i = 0; i < ndim; ++i ) {
                    for ( int j = 0; j < ndim; ++j ) {
                        Scalar* ztmp;
                        if constexpr ( std::is_same_v< Scalar, double > ) { ztmp = new double(0.0); }
                        else if constexpr ( std::is_same_v< Scalar, std::complex<double> > ) { ztmp = new std::complex<double>(0.0,0.0); }
                        else {
                            throw std::invalid_argument("Utils::StableNumerics::mult_v_invd_u<Scalar>(): "
                            "invalid scalar type, supporting double and complex double only.");
                        }
                        for ( int k = 0; k < ndim; ++k ) {
                            *ztmp += vmat(j, k) * umat(k, i) / dvec(k);
                        }
                        zmat(j, i) = *ztmp;
                        delete ztmp;
                    }
                }
            }


            // Subroutine to perform dense matrix * diagonal matrix * dense matrix
            // Input:  vmat, dvec, umat
            // Output: zmat
            template<typename Scalar>
            static void mult_v_d_u( const Matrix<Scalar>& vmat, const Vector& dvec, const Matrix<Scalar>& umat, Matrix<Scalar>& zmat )
            {
                assert( vmat.cols() == umat.cols() );
                assert( vmat.cols() == zmat.cols() );
                assert( vmat.rows() == umat.rows() );
                assert( vmat.rows() == zmat.rows() );
                assert( vmat.rows() == vmat.cols() );
                assert( vmat.cols() == dvec.size() );

                const int ndim = static_cast<int>(vmat.rows());
                for ( int i = 0; i < ndim; ++i ) {
                    for ( int j = 0; j < ndim; ++j ) {
                        Scalar* ztmp;
                        if constexpr ( std::is_same_v< Scalar, double > ) { ztmp = new double(0.0); }
                        else if constexpr ( std::is_same_v< Scalar, std::complex<double> > ) { ztmp = new std::complex<double>(0.0,0.0); }
                        else {
                            throw std::invalid_argument("Utils::StableNumerics::mult_v_d_u<Scalar>(): "
                            "invalid scalar type, supporting double and complex double only.");
                        }
                        for ( int k = 0; k < ndim; ++k ) {
                            *ztmp += vmat(j, k) * umat(k, i) * dvec(k);
                        }
                        zmat(j, i) = *ztmp;
                        delete ztmp;
                    }
                }
            }


            // return (1 + USV^H)^-1, with method of QR decomposition
            // to obtain equal-time Green's functions G(t,t)
            template<typename Scalar>
            static void compute_gf_00bb( const Matrix<Scalar>& u, const Vector& s, const Matrix<Scalar>& v, Matrix<Scalar>& gtt )
            {
                assert( s.minCoeff() >= 0.0 );
                
                // split s = sl^-1 * sr
                Vector sl(s.size());
                Vector sr(s.size());
                for ( int i = 0; i < s.size(); ++i ) {
                    if ( s(i) > 1.0 ) { sl(i) = 1.0 / s(i); sr(i) = 1.0; }
                    else { sl(i) = 1.0; sr(i) = s(i); }
                }

                // compute (1 + USV^H)^-1 in a stable manner
                // note H is well-defined, which only contains information of small scales.
                // for real-valued inputs, adjoint() degenerates to transpose() automatically.
                Matrix<Scalar> H = sl.asDiagonal() * u.adjoint() + sr.asDiagonal() * v.adjoint();

                // compute gtt using QR decomposition
                gtt = H.fullPivHouseholderQr().solve( sl.asDiagonal() * u.adjoint() );
            }


            // return (1 + USV^H)^-1 * USV^H, with method of QR decomposition
            // to obtain time-displaced Green's functions G(beta, 0)
            template<typename Scalar>
            static void compute_gf_b0( const Matrix<Scalar>& u, const Vector& s, const Matrix<Scalar>& v, Matrix<Scalar>& gt0 )
            {
                assert( s.minCoeff() >= 0.0 );

                // split s = sl^-1 * sr
                Vector sl(s.size());
                Vector sr(s.size());
                for ( int i = 0; i < s.size(); ++i ) {
                    if( s(i) > 1.0 ) { sl(i) = 1.0 / s(i); sr(i) = 1.0; }
                    else { sl(i) = 1.0; sr(i) = s(i); }
                }

                // compute (1 + USV^H)^-1 * USV^H in a stable manner
                // note H is well-defined, which only contains information of small scale.
                // for real-valued inputs, adjoint() degenerates to transpose() automatically.
                Matrix<Scalar> H = sl.asDiagonal() * u.adjoint() + sr.asDiagonal() * v.adjoint();

                // compute gtt using QR decomposition
                gt0 = H.fullPivHouseholderQr().solve( sr.asDiagonal() * v.adjoint() );
            }


            // return (1 + left * right^H)^-1 in a stable manner, with method of MGS factorization
            // note:  (1 + left * right^H)^-1 = (1 + (USV^H)_left * (VSU^H)_right)^-1
            template<typename Scalar>
            static void compute_equaltime_gf( SvdStack<Scalar>& left, SvdStack<Scalar>& right, Matrix<Scalar>& gtt )
            {
                assert( left.MatDim() == right.MatDim() );
                const int ndim = left.MatDim();

                // at time slice t = 0
                if ( left.empty() ) {
                    compute_gf_00bb<Scalar>( right.MatrixV(), right.SingularValues(), right.MatrixU(), gtt );
                    return;
                }

                // at time slice t = nt (beta)
                if ( right.empty() ) {
                    compute_gf_00bb<Scalar>( left.MatrixU(), left.SingularValues(), left.MatrixV(), gtt );
                    return;
                }

                // local params
                const Matrix<Scalar> ul = left.MatrixU();
                const Matrix<Scalar> vl = left.MatrixV();
                const Matrix<Scalar> ur = right.MatrixU();
                const Matrix<Scalar> vr = right.MatrixV();
                const Vector dl = left.SingularValues();
                const Vector dr = right.SingularValues();

                Vector dlmax(dl.size()), dlmin(dl.size());
                Vector drmax(dr.size()), drmin(dr.size());

                Matrix<Scalar> Atmp(ndim, ndim), Btmp(ndim, ndim), tmp(ndim, ndim);

                // modified Gram-Schmidt (MGS) factorization
                // perfrom the breakups dr = drmax * drmin , dl = dlmax * dlmin
                div_dvec_max_min( dl, dlmax, dlmin );
                div_dvec_max_min( dr, drmax, drmin );

                // Atmp = ul^H * ur and Btmp = vl^H * vr
                Atmp = ul.adjoint() * ur;
                Btmp = vl.adjoint() * vr;

                // Atmp = dlmax^-1 * (ul^H * ur) * drmax^-1
                // Btmp = dlmin * (vl^H * vr) * drmin
                for ( int j = 0; j < ndim; ++j ) {
                    for ( int i = 0; i < ndim; ++i ) {
                        Atmp(i, j) = Atmp(i, j) / ( dlmax(i) * drmax(j) );
                        Btmp(i, j) = Btmp(i, j) * dlmin(i) * drmin(j);
                    }
                }

                tmp = Atmp + Btmp;
                mult_v_invd_u<Scalar>( ur, drmax, tmp.inverse(), Atmp );

                // finally compute gtt
                mult_v_invd_u<Scalar>( Atmp, dlmax, ul.adjoint(), gtt );
            }


            // return time-displaced Green's function in a stable manner,
            // with the method of MGS factorization
            template<typename Scalar>
            static void compute_dynamic_gf( SvdStack<Scalar>& left, SvdStack<Scalar>& right, Matrix<Scalar> &gt0, Matrix<Scalar>& g0t )
            {
                assert( left.MatDim() == right.MatDim() );
                const int ndim = left.MatDim();

                // at time slice t = 0
                if( left.empty() ) {
                    // gt0 = gtt at t = 0
                    compute_gf_00bb<Scalar>( right.MatrixV(), right.SingularValues(), right.MatrixU(), gt0 );

                    // g0t = - ( 1 - gtt ）at t = 0
                    g0t = - ( Matrix<Scalar>::Identity(ndim, ndim) - gt0 );
                    return;
                }

                // at time slice t = nt (beta)
                if( right.empty() ) {
                    // gt0 = ( 1 + B(beta, 0) )^-1 * B(beta, 0)
                    compute_gf_b0<Scalar>( left.MatrixU(), left.SingularValues(), left.MatrixV(), gt0 );

                    // g0t = -gtt at t = beta
                    // e.g. it can be checked that: g0t(t=beta) = - B^{-1} (1-(1+B)^{-1}) = - (1+B)^{-1} = -gtt(t=0/beta)
                    compute_gf_00bb<Scalar>( left.MatrixU(), left.SingularValues(), left.MatrixV(), g0t );
                    g0t = - g0t;
                    return;
                }

                // local params
                const Matrix<Scalar> ul = left.MatrixU();
                const Matrix<Scalar> vl = left.MatrixV();
                const Matrix<Scalar> ur = right.MatrixU();
                const Matrix<Scalar> vr = right.MatrixV();
                const Vector dl = left.SingularValues();
                const Vector dr = right.SingularValues();

                Vector dlmax(dl.size()), dlmin(dl.size());
                Vector drmax(dr.size()), drmin(dr.size());

                Matrix<Scalar> Atmp(ndim, ndim), Btmp(ndim, ndim);
                Matrix<Scalar> Xtmp(ndim, ndim), Ytmp(ndim, ndim);
                Matrix<Scalar> tmp(ndim, ndim);

                // modified Gram-Schmidt (MGS) factorization
                // perfrom the breakups dr = drmax * drmin , dl = dlmax * dlmin
                div_dvec_max_min( dl, dlmax, dlmin );
                div_dvec_max_min( dr, drmax, drmin );

                // compute gt0
                // Atmp = ul^H * ur and Btmp = vl^H * vr
                Atmp = ul.adjoint() * ur;
                Btmp = vl.adjoint() * vr;

                // Atmp = dlmax^-1 * (ul^H * ur) * drmax^-1
                // Btmp = dlmin * (vl^H * vr) * drmin
                for ( int j = 0; j < ndim; ++j ) {
                    for ( int i = 0; i < ndim; ++i ) {
                        Atmp(i, j) = Atmp(i, j) / ( dlmax(i) * drmax(j) );
                        Btmp(i, j) = Btmp(i, j) * dlmin(i) * drmin(j);
                    }
                }
                tmp = Atmp + Btmp;
                mult_v_invd_u<Scalar>( ur, drmax, tmp.inverse(), Atmp );
                mult_v_d_u<Scalar>( Atmp, dlmin, vl.adjoint(), gt0 );

                // compute g0t
                // Xtmp = vr^H * vl, Ytmp = ur^H * ul
                Xtmp = vr.adjoint() * vl;
                Ytmp = ur.adjoint() * ul;

                // Xtmp = drmax^-1 * (vr^H * vl) * dlmax^-1
                // Ytmp = drmin * (ur^H * ul) * dlmin
                for ( int j = 0; j < ndim; ++j ) {
                    for ( int i = 0; i < ndim; ++i ) {
                        Xtmp(i, j) = Xtmp(i, j) / ( drmax(i) * dlmax(j) );
                        Ytmp(i, j) = Ytmp(i, j) * drmin(i) * dlmin(j);
                    }
                }
                tmp = Xtmp + Ytmp;
                mult_v_invd_u<Scalar>( -vl, dlmax, tmp.inverse(), Xtmp );
                mult_v_d_u<Scalar>( Xtmp, drmin, ur.adjoint(), g0t );
            }

    };

} // namespace Utils

#endif // UTILS_STABLE_NUMERICS_HPP