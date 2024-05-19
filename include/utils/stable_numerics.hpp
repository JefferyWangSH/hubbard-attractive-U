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
        private:
            template<typename _scalar>
            using _matrix = Eigen::Matrix<_scalar, Eigen::Dynamic, Eigen::Dynamic>;            
            using _vector = Eigen::VectorXd;
        
        public:
            // Subroutine to return the maximum difference of two matrices with the same size.
            // Input: mat1, mat2
            // Output: the maximum difference -> error
            template<typename _scalar>
            static void matrix_compare_error(const _matrix<_scalar>& mat1, const _matrix<_scalar>& mat2, double& error)
            {
                assert(mat1.rows() == mat2.rows());
                assert(mat1.cols() == mat2.cols());
                if constexpr (std::is_same_v<_scalar,double>) { error = (mat1-mat2).maxCoeff(); }
                else if constexpr (std::is_same_v<_scalar,std::complex<double>>) { error = (mat1-mat2).cwiseAbs2().cwiseSqrt().maxCoeff(); }
                else {
                    throw std::invalid_argument("Utils::StableNumerics::matrix_compare_error<_scalar>(): "
                    "invalid scalar type, supporting double and complex double only.");
                }
            }


            // Subroutine to perform the decomposition of a vector, dvec = dmax * dmin,
            // to ensure all elements that greater than one are stored in dmax,
            // and all elements that less than one are stored in dmin.
            // Input:  dvec
            // Output: dmax, dmin
            static void div_dvec_max_min(const _vector& dvec, _vector& dmax, _vector& dmin)
            {
                assert(dvec.size() == dmax.size());
                assert(dvec.size() == dmin.size());
                assert(dvec.minCoeff() >= 0.);
                _vector ones = _vector::Ones(dvec.size());
                dmax = dvec.cwiseMax(ones);
                dmin = dvec.cwiseMin(ones);
            }


            // Subroutine to perform the multiplication: dense_matrix * (diagonal_matrix)^-1 * dense_matrix
            // Input:  vmat, dvec, umat
            // Output: zmat
            template<typename _scalar>
            static void mult_v_invd_u(const _matrix<_scalar>& vmat, const _vector& dvec, const _matrix<_scalar>& umat, _matrix<_scalar>& zmat)
            {
                assert(vmat.cols() == umat.cols());
                assert(vmat.cols() == zmat.cols());
                assert(vmat.rows() == umat.rows());
                assert(vmat.rows() == zmat.rows());
                assert(vmat.rows() == vmat.cols());
                assert(vmat.cols() == dvec.size());

                const int ndim = static_cast<int>(vmat.rows());
                for (int i = 0; i < ndim; ++i) {
                    for (int j = 0; j < ndim; ++j) {
                        _scalar* ztemp;
                        if constexpr (std::is_same_v<_scalar,double>) { ztemp = new double(0.); }
                        else if constexpr (std::is_same_v<_scalar,std::complex<double>>) { ztemp = new std::complex<double>(0.,0.); }
                        else {
                            throw std::invalid_argument("Utils::StableNumerics::mult_v_invd_u<_scalar>(): "
                            "invalid scalar type, supporting double and complex double only.");
                        }
                        for (int k = 0; k < ndim; ++k) {
                            *ztemp += vmat(j,k) * umat(k,i) / dvec(k);
                        }
                        zmat(j,i) = *ztemp;
                        delete ztemp;
                    }
                }
            }


            // Subroutine to perform the multiplication: dense_matrix * diagonal_matrix * dense_matrix
            // Input:  vmat, dvec, umat
            // Output: zmat
            template<typename _scalar>
            static void mult_v_d_u(const _matrix<_scalar>& vmat, const _vector& dvec, const _matrix<_scalar>& umat, _matrix<_scalar>& zmat)
            {
                assert(vmat.cols() == umat.cols());
                assert(vmat.cols() == zmat.cols());
                assert(vmat.rows() == umat.rows());
                assert(vmat.rows() == zmat.rows());
                assert(vmat.rows() == vmat.cols());
                assert(vmat.cols() == dvec.size());

                const int ndim = static_cast<int>(vmat.rows());
                for (int i = 0; i < ndim; ++i) {
                    for (int j = 0; j < ndim; ++j) {
                        _scalar* ztemp;
                        if constexpr (std::is_same_v<_scalar,double>) { ztemp = new double(0.); }
                        else if constexpr (std::is_same_v<_scalar,std::complex<double>>) { ztemp = new std::complex<double>(0.,0.); }
                        else {
                            throw std::invalid_argument("Utils::StableNumerics::mult_v_d_u<_scalar>(): "
                            "invalid scalar type, supporting double and complex double only.");
                        }
                        for (int k = 0; k < ndim; ++k) {
                            *ztemp += vmat(j,k) * umat(k,i) * dvec(k);
                        }
                        zmat(j,i) = *ztemp;
                        delete ztemp;
                    }
                }
            }


            // return (1 + USV^H)^-1, with method of QR decomposition
            // to obtain equal-time Green's functions G(t,t)
            template<typename _scalar>
            static void compute_gf_00bb(const _matrix<_scalar>& u, const _vector& s, const _matrix<_scalar>& v, _matrix<_scalar>& gtt)
            {
                assert(s.minCoeff() >= 0.);
                
                // split s = sl^-1 * sr
                _vector sl(s.size());
                _vector sr(s.size());
                for (int i = 0; i < s.size(); ++i) {
                    if (s(i) > 1.) { sl(i) = 1./s(i); sr(i) = 1.; }
                    else { sl(i) = 1.; sr(i) = s(i); }
                }

                // compute (1 + USV^H)^-1 in a stable manner
                // note H is well-defined, which only contains information of small scales.
                // for real-valued inputs, adjoint() degenerates to transpose() automatically.
                _matrix<_scalar> H = sl.asDiagonal() * u.adjoint() + sr.asDiagonal() * v.adjoint();

                // compute gtt using QR decomposition
                gtt = H.fullPivHouseholderQr().solve( sl.asDiagonal()*u.adjoint() );
            }


            // return (1 + USV^H)^-1 * USV^H, with method of QR decomposition
            // to obtain time-displaced Green's functions G(beta, 0)
            template<typename _scalar>
            static void compute_gf_b0(const _matrix<_scalar>& u, const _vector& s, const _matrix<_scalar>& v, _matrix<_scalar>& gt0)
            {
                assert(s.minCoeff() >= 0.);

                // split s = sl^-1 * sr
                _vector sl(s.size());
                _vector sr(s.size());
                for (int i = 0; i < s.size(); ++i) {
                    if(s(i) > 1.) { sl(i) = 1./s(i); sr(i) = 1.; }
                    else { sl(i) = 1.; sr(i) = s(i); }
                }

                // compute (1 + USV^H)^-1 * USV^H in a stable manner
                // note H is well-defined, which only contains information of small scale.
                // for real-valued inputs, adjoint() degenerates to transpose() automatically.
                _matrix<_scalar> H = sl.asDiagonal() * u.adjoint() + sr.asDiagonal() * v.adjoint();

                // compute gtt using QR decomposition
                gt0 = H.fullPivHouseholderQr().solve( sr.asDiagonal()*v.adjoint() );
            }


            // return (1 + left * right^H)^-1 in a stable manner, with method of MGS factorization
            // note:  (1 + left * right^H)^-1 = (1 + (USV^H)_left * (VSU^H)_right)^-1
            template<typename _scalar>
            static void compute_equaltime_gf(SvdStack<_scalar>& left, SvdStack<_scalar>& right, _matrix<_scalar>& gtt)
            {
                assert(left.MatDim() == right.MatDim());
                const int ndim = left.MatDim();

                // at time slice t = 0
                if (left.empty()) {
                    compute_gf_00bb<_scalar>(right.MatrixV(), right.SingularValues(), right.MatrixU(), gtt);
                    return;
                }

                // at time slice t = nt (beta)
                if (right.empty()) {
                    compute_gf_00bb<_scalar>(left.MatrixU(), left.SingularValues(), left.MatrixV(), gtt);
                    return;
                }

                // local params
                const _matrix<_scalar> ul = left.MatrixU();
                const _matrix<_scalar> vl = left.MatrixV();
                const _matrix<_scalar> ur = right.MatrixU();
                const _matrix<_scalar> vr = right.MatrixV();
                const _vector dl = left.SingularValues();
                const _vector dr = right.SingularValues();

                _vector dlmax(dl.size()), dlmin(dl.size());
                _vector drmax(dr.size()), drmin(dr.size());

                _matrix<_scalar> tempA(ndim, ndim), tempB(ndim, ndim), temp(ndim, ndim);

                // modified Gram-Schmidt (MGS) factorization
                // perfrom the breakups dr = drmax * drmin , dl = dlmax * dlmin
                div_dvec_max_min(dl, dlmax, dlmin);
                div_dvec_max_min(dr, drmax, drmin);

                // tempA = ul^H * ur and tempB = vl^H * vr
                tempA = ul.adjoint() * ur;
                tempB = vl.adjoint() * vr;

                // tempA = dlmax^-1 * (ul^H * ur) * drmax^-1
                // tempB = dlmin * (vl^H * vr) * drmin
                for (int j = 0; j < ndim; ++j) {
                    for (int i = 0; i < ndim; ++i) {
                        tempA(i,j) = tempA(i,j) / (dlmax(i)*drmax(j));
                        tempB(i,j) = tempB(i,j) * dlmin(i)*drmin(j);
                    }
                }

                temp = tempA + tempB;
                mult_v_invd_u<_scalar>(ur, drmax, temp.inverse(), tempA);

                // finally compute gtt
                mult_v_invd_u<_scalar>(tempA, dlmax, ul.adjoint(), gtt);
            }


            // return time-displaced Green's function in a stable manner,
            // with the method of MGS factorization
            template<typename _scalar>
            static void compute_dynamic_gf(SvdStack<_scalar>& left, SvdStack<_scalar>& right, _matrix<_scalar>& gt0, _matrix<_scalar>& g0t, _matrix<_scalar>& gtt)
            {
                assert(left.MatDim() == right.MatDim());
                const int ndim = left.MatDim();

                // at time slice t = 0
                if(left.empty()) {
                    // gt0 = gtt(t=0/beta), and g0t = -(1 - gtt(t=0/beta)）at t = 0
                    compute_gf_00bb<_scalar>(right.MatrixV(), right.SingularValues(), right.MatrixU(), gtt);
                    gt0 = gtt;
                    g0t = - (_matrix<_scalar>::Identity(ndim, ndim) - gtt);
                    return;
                }

                // at time slice t = nt (beta)
                if(right.empty()) {
                    // gt0 = (1+B(beta,0))^-1 * B(beta,0) = 1 - gtt(t=0/beta) at t = beta
                    // g0t = -gtt(t=0/beta) at t = beta
                    // e.g. it can be checked that: g0t(t=beta) = -B(beta,0)^-1 (1 - (1+B(beta,0))^-1) = -(1+B(beta,0))^-1 = -gtt(t=0/beta)
                    compute_gf_00bb<_scalar>(left.MatrixU(), left.SingularValues(), left.MatrixV(), gtt);
                    gt0 = _matrix<_scalar>::Identity(ndim, ndim) - gtt;
                    g0t = - gtt;
                    return;
                }

                // local params
                const _matrix<_scalar> ul = left.MatrixU();
                const _matrix<_scalar> vl = left.MatrixV();
                const _matrix<_scalar> ur = right.MatrixU();
                const _matrix<_scalar> vr = right.MatrixV();
                const _vector dl = left.SingularValues();
                const _vector dr = right.SingularValues();

                _vector dlmax(dl.size()), dlmin(dl.size());
                _vector drmax(dr.size()), drmin(dr.size());

                _matrix<_scalar> tempA(ndim, ndim), tempB(ndim, ndim);
                _matrix<_scalar> tempC(ndim, ndim), tempD(ndim, ndim);
                _matrix<_scalar> temp(ndim, ndim);

                // modified Gram-Schmidt (MGS) factorization
                // perfrom the breakups dr = drmax * drmin , dl = dlmax * dlmin
                div_dvec_max_min(dl, dlmax, dlmin);
                div_dvec_max_min(dr, drmax, drmin);

                // compute gt0
                // tempA = ul^H * ur and tempB = vl^H * vr
                tempA = ul.adjoint() * ur;
                tempB = vl.adjoint() * vr;

                // tempA = dlmax^-1 * (ul^H * ur) * drmax^-1
                // tempB = dlmin * (vl^H * vr) * drmin
                for (int j = 0; j < ndim; ++j) {
                    for (int i = 0; i < ndim; ++i) {
                        tempA(i,j) = tempA(i,j) / (dlmax(i)*drmax(j));
                        tempB(i,j) = tempB(i,j) * dlmin(i)*drmin(j);
                    }
                }
                temp = tempA + tempB;
                mult_v_invd_u<_scalar>(ur, drmax, temp.inverse(), tempA);
                mult_v_invd_u<_scalar>(tempA, dlmax, ul.adjoint(), gtt);
                mult_v_d_u<_scalar>(tempA, dlmin, vl.adjoint(), gt0);

                // compute g0t
                // tempC = vr^H * vl, tempD = ur^H * ul
                tempC = vr.adjoint() * vl;
                tempD = ur.adjoint() * ul;

                // tempC = drmax^-1 * (vr^H * vl) * dlmax^-1
                // tempD = drmin * (ur^H * ul) * dlmin
                for (int j = 0; j < ndim; ++j) {
                    for (int i = 0; i < ndim; ++i) {
                        tempC(i,j) = tempC(i,j) / (drmax(i)*dlmax(j));
                        tempD(i,j) = tempD(i,j) * drmin(i)*dlmin(j);
                    }
                }
                temp = tempC + tempD;
                mult_v_invd_u<_scalar>(-vl, dlmax, temp.inverse(), tempC);
                mult_v_d_u<_scalar>(tempC, drmin, ur.adjoint(), g0t);
            }
    };

} // namespace Utils

#endif // UTILS_STABLE_NUMERICS_HPP