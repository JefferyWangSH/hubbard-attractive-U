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

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <unsupported/Eigen/MatrixFunctions>

namespace Model {
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

        this->m_expK.resize(this->m_ng, this->m_ng);
        this->m_inv_expK.resize(this->m_ng, this->m_ng);
        using Hamiltonian = Eigen::MatrixXd;
        const Hamiltonian K = -this->m_nnt * lattice.nn_hoppings() - this->m_mu*Hamiltonian::Identity(this->m_ng,this->m_ng);
        this->m_expK = (-this->m_dt*K).exp();
        this->m_inv_expK = (+this->m_dt*K).exp();
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
    //                             Monte Carlo and Green's function updates
    //
    // --------------------------------------------------------------------------------------------------------

    void HubbardAttractiveU::locally_update_ising_field(int t, int i)
    {
        this->m_ising_fields(t,i) = - this->m_ising_fields(t,i);
    }

    const double HubbardAttractiveU::get_acceptance_ratio(const DqmcCore& core, int t, int i) const
    {
        // the HS decomposition for attractive-U hubbard model preserves the spin rotation,
        // such that for spin-up/down fermions their Green's functions are identical.
        const refGreenFunc gftt = core.current_gftt();
        return exp(2*this->m_alpha*this->m_ising_fields(t,i))
            * std::pow(1+(1-gftt(i,i))*(exp(-2*this->m_alpha*this->m_ising_fields(t,i))-1),2);
        // TODO: speed up here by define delta = exp(-2*this->m_alpha*this->m_ising_fields(t,i))-1
    }

    void HubbardAttractiveU::update_green_function(DqmcCore& core, int t, int i)
    {
        // update the equal-time green's functions by supposing the ising field at (t,i) is flipped.
        // however the ising field at (t,i) remains unchanged.
        refGreenFunc gftt = core.current_gftt();
        const double factor = (exp(-2*this->m_alpha*this->m_ising_fields(t,i))-1)
            / (1+(1-gftt(i,i))*(exp(-2*this->m_alpha*this->m_ising_fields(t,i))-1));
        gftt -= factor * gftt.col(i) * (Eigen::VectorXd::Unit(this->m_ns,i).transpose()-gftt.row(i));
    }

    // --------------------------------------------------------------------------------------------------------
    // 
    //                       Multiplications of hopping kernel K and coupling kernel V
    //
    // --------------------------------------------------------------------------------------------------------

    void HubbardAttractiveU::multiply_expK_from_left(refGreenFunc gf) const {gf = this->m_expK * gf;}
    void HubbardAttractiveU::multiply_expK_from_right(refGreenFunc gf) const {gf = gf * this->m_expK;}
    void HubbardAttractiveU::multiply_inv_expK_from_left(refGreenFunc gf) const {gf = this->m_inv_expK * gf;}
    void HubbardAttractiveU::multiply_inv_expK_from_right(refGreenFunc gf) const {gf = gf * this->m_inv_expK;}
    void HubbardAttractiveU::multiply_adj_expK_from_left(refGreenFunc gf) const
    {
        // K is Hermitian matrix
        this->multiply_expK_from_left(gf);
    }

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
        this->multiply_expK_from_left(gf);
        this->multiply_expV_from_left(gf, t);
    }

    void HubbardAttractiveU::multiply_B_from_right(refGreenFunc gf, const int t) const
    {
        // Multiply the green's function, from the right by B(t)
        //      G  ->  G * B(t) = G * exp( -dt V(t) ) * exp( -dt K )
        // The green's function is changed in place.
        assert(t >= 0 && t < this->m_nt);
        this->multiply_expV_from_right(gf, t);
        this->multiply_expK_from_right(gf);
    }

    void HubbardAttractiveU::multiply_invB_from_left(refGreenFunc gf, const int t) const
    {
        // Multiply the green's function, from the left by B(t)^-1
        //      G  ->  B(t)^-1 * G = exp( +dt K ) * exp( +dt V(t) ) * G
        // The green's function is changed in place.
        assert(t >= 0 && t < this->m_nt);
        this->multiply_inv_expV_from_left(gf, t);
        this->multiply_inv_expK_from_left(gf);
    }

    void HubbardAttractiveU::multiply_invB_from_right(refGreenFunc gf, const int t) const
    {
        // Multiply the green's function, from the right by B(t)^-1
        //      G  ->  G * B(t)^-1 = G * exp( +dt K ) * exp( +dt V(t) )
        // The green's function is changed in place.
        assert(t >= 0 && t < this->m_nt);
        this->multiply_inv_expK_from_right(gf);
        this->multiply_inv_expV_from_right(gf, t);
    }

    void HubbardAttractiveU::multiply_adjB_from_left(refGreenFunc gf, const int t) const
    {
        // Multiply the green's function, from the left by B(t)^+
        //      G  ->  B(t)^+ * G = exp( -dt K )^+ * exp( -dt V(t) )^+ * G
        // The green's function is changed in place.
        assert(t >= 0 && t < this->m_nt);
        this->multiply_adj_expV_from_left(gf, t);
        this->multiply_adj_expK_from_left(gf);
    }
}