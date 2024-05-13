/*
 *   hubbard.cpp
 * 
 *     Created on: May 8, 2024
 *         Author: Jeffery Wang
 * 
 */

#include "hubbard.h"
#include "dqmc_params.hpp"
#include "square_lattice.h"
#include "random.h"
#include "dqmc_core.h"

#include <cmath>
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <unsupported/Eigen/MatrixFunctions>

namespace Model {

    using fftsolver = Utils::FFTSolver<double, 2>;
    using ptrfftsolver = std::unique_ptr<fftsolver>;

    void HubbardAttractiveU::initialize(const DqmcParams& params, const Lattice& lattice)
    {
        this->m_nl = params.nl;
        this->m_ns = params.ns;
        this->m_ng = params.ng;
        this->m_nt = params.nt;
        this->m_dt = params.dt;
        this->m_nnt = params.nnt;
        this->m_mu = params.mu;
        this->m_u = params.u;
        this->m_alpha = acosh(exp(0.5*this->m_dt*this->m_u));

        this->m_ising_fields.resize(this->m_nt, this->m_ns);

        this->m_is_fft = params.is_fft;
        using Hamiltonian = Eigen::MatrixXd;
        Hamiltonian K = -this->m_nnt * lattice.nn_hoppings()
                        -this->m_mu * Hamiltonian::Identity(this->m_ng,this->m_ng);
        if (!this->m_is_fft){
            this->m_expK.resize(this->m_ng, this->m_ng);
            this->m_inv_expK.resize(this->m_ng, this->m_ng);
            this->m_expK = (-this->m_dt*K).exp();
            this->m_inv_expK = (+this->m_dt*K).exp();
            this->link2naive();
        }
        else {
            std::array<std::size_t,2> sizes = {static_cast<std::size_t>(this->m_nl),
                                               static_cast<std::size_t>(this->m_nl)};
            if (this->m_fftsolver) { this->m_fftsolver.reset(); }
            this->m_fftsolver = std::make_unique<fftsolver>(sizes);
            this->m_expK_eigens.resize(this->m_ns);
            this->m_inv_expK_eigens.resize(this->m_ns);

            // diagonalize the free Hamiltonian K using fft to obatin the band dispersion E(k), i.e.,
            //
            //      E(k) = F K F^-1,
            //
            // where F denotes the Fourier transformation.
            Eigen::MatrixXcd dispersion = Eigen::MatrixXcd(this->m_ns, this->m_ns);
            std::complex<double>* cache = new std::complex<double>[this->m_ns];
            Eigen::VectorXd cache_vec(this->m_ns);
            for (auto i = 0; i < this->m_ns; ++i) {
                this->m_fftsolver->fft(K.data()+i*this->m_ns, cache);
                dispersion.col(i) = Eigen::Map<Eigen::VectorXcd>(cache, this->m_ns, 1);
            }
            for (auto j = 0; j < this->m_ns; ++j) {
                // copy happens here to ensure that the memory is continual
                Eigen::Map<Eigen::VectorXcd>(&cache[0], this->m_ns, 1) = dispersion.row(j);
                this->m_fftsolver->ifft(cache, cache_vec.data());
                dispersion.row(j) = cache_vec;
            }
            this->m_expK_eigens = (-this->m_dt * dispersion.diagonal().real()).array().exp();
            this->m_inv_expK_eigens = (+this->m_dt * dispersion.diagonal().real()).array().exp();
            delete [] cache;

            this->link2fft();
        }
    }

    void HubbardAttractiveU::link2naive()
    {
        this->m_multiply_expK_from_left = std::bind(&HubbardAttractiveU::multiply_expK_from_left, this, std::placeholders::_1);
        this->m_multiply_expK_from_right = std::bind(&HubbardAttractiveU::multiply_expK_from_right, this, std::placeholders::_1);
        this->m_multiply_inv_expK_from_left = std::bind(&HubbardAttractiveU::multiply_inv_expK_from_left, this, std::placeholders::_1);
        this->m_multiply_inv_expK_from_right = std::bind(&HubbardAttractiveU::multiply_inv_expK_from_right, this, std::placeholders::_1);
        this->m_multiply_adj_expK_from_left = std::bind(&HubbardAttractiveU::multiply_adj_expK_from_left, this, std::placeholders::_1);
    }

    void HubbardAttractiveU::link2fft()
    {
        this->m_multiply_expK_from_left = std::bind(&HubbardAttractiveU::multiply_expK_from_left_with_fft, this, std::placeholders::_1);
        this->m_multiply_expK_from_right = std::bind(&HubbardAttractiveU::multiply_expK_from_right_with_fft, this, std::placeholders::_1);
        this->m_multiply_inv_expK_from_left = std::bind(&HubbardAttractiveU::multiply_inv_expK_from_left_with_fft, this, std::placeholders::_1);
        this->m_multiply_inv_expK_from_right = std::bind(&HubbardAttractiveU::multiply_inv_expK_from_right_with_fft, this, std::placeholders::_1);
        this->m_multiply_adj_expK_from_left = std::bind(&HubbardAttractiveU::multiply_adj_expK_from_left_with_fft, this, std::placeholders::_1);
    }

    void HubbardAttractiveU::set_ising_fields_to_random()
    {
        std::bernoulli_distribution bernoulli_dist(0.5);
        for (auto t=0; t<this->m_nt; ++t) {
            for (auto i=0; i<this->m_ns; ++i) {
                this->m_ising_fields(t,i) = bernoulli_dist(Utils::Random::Engine)? +1:-1;
            }
        }
    }

    // --------------------------------------------------------------------------------------------------------
    // 
    //                       Monte Carlo and Green's function updates
    //
    // --------------------------------------------------------------------------------------------------------

    void HubbardAttractiveU::locally_update_ising_field(int t, int i)
    {
        this->m_ising_fields(t,i) = - this->m_ising_fields(t,i);
    }

    const double HubbardAttractiveU::get_acceptance_ratio(const DqmcCore& core, int t, int i) const
    {
        // the HS decomposition for attractive-U hubbard model preserves the spin rotation symm.,
        // such that for spin-up/down fermions their Green's functions are identical.
        const refGreenFunc gftt = core.current_gftt();
        const double delta = std::exp(-2*this->m_alpha*this->m_ising_fields(t,i)) - 1;
        return std::pow(1+(1-gftt(i,i))*delta, 2) / (1+delta);
    }

    void HubbardAttractiveU::update_green_function(DqmcCore& core, int t, int i)
    {
        // update the equal-time green's functions by supposing the ising field at (t,i) is flipped.
        // however the ising field at (t,i) remains unchanged.
        refGreenFunc gftt = core.current_gftt();
        const double delta = std::exp(-2*this->m_alpha*this->m_ising_fields(t,i))-1;
        gftt -= delta/(1+(1-gftt(i,i))*delta) * gftt.col(i) * (Eigen::Matrix<double,1,Eigen::Dynamic>::Unit(this->m_ns,i)-gftt.row(i));
    }

    // --------------------------------------------------------------------------------------------------------
    // 
    //                       Multiplications of hopping kernel K
    //
    // --------------------------------------------------------------------------------------------------------

    // naive implementation
    void HubbardAttractiveU::multiply_expK_from_left(refGreenFunc gf) const {gf = this->m_expK * gf;}
    void HubbardAttractiveU::multiply_expK_from_right(refGreenFunc gf) const {gf = gf * this->m_expK;}
    void HubbardAttractiveU::multiply_inv_expK_from_left(refGreenFunc gf) const {gf = this->m_inv_expK * gf;}
    void HubbardAttractiveU::multiply_inv_expK_from_right(refGreenFunc gf) const {gf = gf * this->m_inv_expK;}
    void HubbardAttractiveU::multiply_adj_expK_from_left(refGreenFunc gf) const
    {
        // K is Hermitian matrix
        this->multiply_expK_from_left(gf);
    }

    //
    // FFT implementation (require K is translational invariant.)
    //
    // THE MEMORY OF MATRIX MUST BE CONTINUOUS!
    // FOR COLUMN-MAJOR (DEFAULT) EIGEN MATRIX, MULTIPLY FROM THE LEFT ONLY! (COPY AVOIDED)
    // IF MULTIPLICATIONS FROM THE RIGHT NEEDED, FIRST TRANSPOSE THE MATRIX.
    // ( note that F^T = F, and [expK]^+ = expK for hermitian K )
    // e.g.
    //
    //      (inv)expK * G = fft_mult_from_left(G)
    //
    //      G * (inv)expK = [ [(inv)expK]^T G^T ]^T
    //                    = [ fft_mult_conj_from_left(G^T) ]^T
    //
    inline void fft_mult_from_left(Eigen::Ref<Eigen::MatrixXd> mat, const Eigen::VectorXd& eigens, const fftsolver& solver)
    {
        // -------------------------------------------------------
        //
        //        (inv)expK = F^-1 diag( e^{-/+ dt E(k)} ) F
        //
        // -------------------------------------------------------
        const auto rows = mat.rows();
        const auto cols = mat.cols();
        std::complex<double>* cache = new std::complex<double>[rows];
        for (auto j = 0; j < cols; ++j) {
            solver.fft(mat.data()+j*rows, cache); // for column-major matrix, the memory is continuous
            Eigen::Map<Eigen::VectorXcd> cache_vec(&cache[0], rows);
            cache_vec = cache_vec.array() * eigens.array();
            solver.ifft(cache_vec.data(), mat.data()+j*rows);
        }
        delete [] cache;
    }

    inline void fft_mult_conj_from_left(Eigen::Ref<Eigen::MatrixXd> mat, const Eigen::VectorXd& eigens, const fftsolver& solver)
    {
        // --------------------------------------------------------------------
        //
        //    [(inv)expK]^* = [(inv)expK]^T
        //                  = F diag( e^{-/+ dt E(k)} ) F^-1
        //
        //  the first '=' uses the fact that K is hermitian
        //  and the second '=' holds due to F^T = F.
        //
        //  it's equivalent to first call fft_mult_from_left(A^*)
        //  and then take the conjugation of it,
        //  i.e.,
        //      fft_mult_conj_from_left(A) = [ fft_mult_from_left(A^*) ]^*.
        //
        // --------------------------------------------------------------------
        
        // in our case with real-valued K, [(inv)expK]^* = [(inv)expK],
        // hence fft_mult_conj_from_left(A) = fft_mult_from_left(A).
        fft_mult_from_left(mat, eigens, solver);

        // // but the following codes work for general complex-valued K.
        // mat = mat.conjugate();
        // fft_mult_from_left(mat, eigens, solver);
        // mat = mat.conjugate();
    }

    void HubbardAttractiveU::multiply_expK_from_left_with_fft(refGreenFunc gf) const
    {
        fft_mult_from_left(gf, this->m_expK_eigens, *this->m_fftsolver);
    }

    void HubbardAttractiveU::multiply_expK_from_right_with_fft(refGreenFunc gf) const
    {
        gf.transposeInPlace();
        fft_mult_conj_from_left(gf, this->m_expK_eigens, *this->m_fftsolver);
        gf.transposeInPlace();
    }

    void HubbardAttractiveU::multiply_inv_expK_from_left_with_fft(refGreenFunc gf) const
    {
        fft_mult_from_left(gf, this->m_inv_expK_eigens, *this->m_fftsolver);
    }

    void HubbardAttractiveU::multiply_inv_expK_from_right_with_fft(refGreenFunc gf) const
    {
        gf.transposeInPlace();
        fft_mult_conj_from_left(gf, this->m_inv_expK_eigens, *this->m_fftsolver);
        gf.transposeInPlace();
    }

    void HubbardAttractiveU::multiply_adj_expK_from_left_with_fft(refGreenFunc gf) const
    {
        // K is Hermitian matrix
        this->multiply_expK_from_left_with_fft(gf);
    }

    // --------------------------------------------------------------------------------------------------------
    // 
    //                       Multiplications of coupling kernel V
    //
    // --------------------------------------------------------------------------------------------------------
    
    void HubbardAttractiveU::multiply_expV_from_left(refGreenFunc gf, const int t) const
    {
        for (auto i=0; i<this->m_ns; ++i) {
            gf.row(i) *= exp(this->m_alpha*this->m_ising_fields(t,i));
        }
    }

    void HubbardAttractiveU::multiply_expV_from_right(refGreenFunc gf, const int t) const
    {
        for (auto i=0; i<this->m_ns; ++i) {
            gf.col(i) *= exp(this->m_alpha*this->m_ising_fields(t,i));
        }
    }

    void HubbardAttractiveU::multiply_inv_expV_from_left(refGreenFunc gf, const int t) const
    {
        for (auto i=0; i<this->m_ns; ++i) {
            gf.row(i) *= exp(-this->m_alpha*this->m_ising_fields(t,i));
        }
    }

    void HubbardAttractiveU::multiply_inv_expV_from_right(refGreenFunc gf, const int t) const
    {
        for (auto i=0; i<this->m_ns; ++i) {
            gf.col(i) *= exp(-this->m_alpha*this->m_ising_fields(t,i));
        }
    }

    void HubbardAttractiveU::multiply_adj_expV_from_left(refGreenFunc gf, const int t) const
    {
        this->multiply_expV_from_left(gf, t);
    }

    // --------------------------------------------------------------------------------------------------------
    // 
    //                              Multiplications of B matrices
    //
    // --------------------------------------------------------------------------------------------------------

    void HubbardAttractiveU::multiply_B_from_left(refGreenFunc gf, const int t) const
    {
        // Multiply the green's function, from the left by B(t)
        //      G  ->  B(t) * G = exp( -dt V(t) ) * exp( -dt K ) * G
        // The green's function is changed in place.
        assert(t >= 0 && t < this->m_nt);
        this->m_multiply_expK_from_left(gf);
        this->multiply_expV_from_left(gf, t);
    }

    void HubbardAttractiveU::multiply_B_from_right(refGreenFunc gf, const int t) const
    {
        // Multiply the green's function, from the right by B(t)
        //      G  ->  G * B(t) = G * exp( -dt V(t) ) * exp( -dt K )
        // The green's function is changed in place.
        assert(t >= 0 && t < this->m_nt);
        this->multiply_expV_from_right(gf, t);
        this->m_multiply_expK_from_right(gf);
    }

    void HubbardAttractiveU::multiply_invB_from_left(refGreenFunc gf, const int t) const
    {
        // Multiply the green's function, from the left by B(t)^-1
        //      G  ->  B(t)^-1 * G = exp( +dt K ) * exp( +dt V(t) ) * G
        // The green's function is changed in place.
        assert(t >= 0 && t < this->m_nt);
        this->multiply_inv_expV_from_left(gf, t);
        this->m_multiply_inv_expK_from_left(gf);
    }

    void HubbardAttractiveU::multiply_invB_from_right(refGreenFunc gf, const int t) const
    {
        // Multiply the green's function, from the right by B(t)^-1
        //      G  ->  G * B(t)^-1 = G * exp( +dt K ) * exp( +dt V(t) )
        // The green's function is changed in place.
        assert(t >= 0 && t < this->m_nt);
        this->m_multiply_inv_expK_from_right(gf);
        this->multiply_inv_expV_from_right(gf, t);
    }

    void HubbardAttractiveU::multiply_adjB_from_left(refGreenFunc gf, const int t) const
    {
        // Multiply the green's function, from the left by B(t)^+
        //      G  ->  B(t)^+ * G = exp( -dt K )^+ * exp( -dt V(t) )^+ * G
        // The green's function is changed in place.
        assert(t >= 0 && t < this->m_nt);
        this->multiply_adj_expV_from_left(gf, t);
        this->m_multiply_adj_expK_from_left(gf);
    }
}