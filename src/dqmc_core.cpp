/*
 *   dqmc_core.cpp
 * 
 *     Created on: May 8, 2024
 *         Author: Jeffery Wang
 * 
 */

#include "dqmc_core.h"
#include "dqmc_params.hpp"
#include "hubbard.h"
#include "random.h"
#include "measure_handle.h"
#include "utils/stable_numerics.hpp"

namespace DQMC {
    using BStacks = Eigen::MatrixXd;
    using GF = Eigen::MatrixXd;

    void Core::initialize(const DqmcParams& params, const MeasureHandle& meas_handle)
    {
        this->m_nl = params.nl;
        this->m_ns = params.ns;
        this->m_ng = params.ng;
        this->m_nt = params.nt;
        this->m_dt = params.dt;
        this->m_stabilization_pace = params.stabilization_pace;
        this->m_is_equaltime = meas_handle.isEqualTime();
        this->m_is_dynamic = meas_handle.isDynamic();
    }

    void Core::allocate_svdstacks()
    {
        // release the pointers if assigned before
        if (this->m_svdstack_left) { this->m_svdstack_left.reset(); }
        if (this->m_svdstack_right) { this->m_svdstack_right.reset(); }
        
        // allocate memory for SvdStack objects
        this->m_svdstack_left  = std::make_unique<svdStack>(this->m_ng, this->m_nt);
        this->m_svdstack_right = std::make_unique<svdStack>(this->m_ng, this->m_nt);
    }

    void Core::allocate_green_functions()
    {
        // release the pointers if assigned before
        if (this->m_gftt) { this->m_gftt.reset(); }
        if (this->m_gft0) { this->m_gft0.reset(); }
        if (this->m_gf0t ) { this->m_gf0t.reset(); }
        if (this->m_vecgftt) { this->m_vecgftt.reset(); }
        if (this->m_vecgft0) { this->m_vecgft0.reset(); }
        if (this->m_vecgf0t) { this->m_vecgf0t.reset(); }

        // allocate memories for green's functions
        this->m_gftt = std::make_unique<GF>(this->m_ng, this->m_ng);
        // NOTE: Equal-time Green's functions are also collected during sweeps for dynamic Green's functions.
        if (this->m_is_equaltime || this->m_is_dynamic) {
            this->m_vecgftt = std::make_unique<vecGF>(this->m_nt, GF::Zero(this->m_ng, this->m_ng));
        }
        if (this->m_is_dynamic) {
            this->m_gft0 = std::make_unique<GF>(this->m_ng, this->m_ng);
            this->m_gf0t = std::make_unique<GF>(this->m_ng, this->m_ng);
            this->m_vecgft0 = std::make_unique<vecGF>(this->m_nt, GF::Zero(this->m_ng, this->m_ng));
            this->m_vecgf0t = std::make_unique<vecGF>(this->m_nt, GF::Zero(this->m_ng, this->m_ng));
        }
    }

    void Core::initialize_svdstacks(const Hubbard& model)
    {
        // initialize svd stacks for sweeping usage
        // the sweep process will start upwards from t=0 to beta, hence we initialize svd_stack_right here.
        // stabilize the procedure every `stabilization_pace` steps

        // first allocate memory
        this->allocate_svdstacks();
    
        BStacks stacks = BStacks::Identity(this->m_ng, this->m_ng);
        for (auto t = this->m_nt; t >= 1; --t) {
            // directly multiplication of the B matrices
            model.multiply_adjB_from_left(stacks, t2index(t));

            // stabilize the procedure every 'stabilization_pace' steps using the svd decomposition
            if ((t-1) % this->m_stabilization_pace == 0) {
                this->m_svdstack_right->push(stacks);
                stacks = BStacks::Identity(this->m_ng, this->m_ng);
            }
        }
    }

    void Core::initialize_green_functions()
    {
        // allocate memories for green's functions
        this->allocate_green_functions();

        // compute the equal-time green's function at t = 0, which equals Gtt(t=beta) due to PBC.
        // the svd stacks should be initialized correctly in advance
        Utils::StableNumerics::compute_equaltime_gf(*this->m_svdstack_left, *this->m_svdstack_right, *this->m_gftt);
    }

    /*
     *  Local updates of ising fields at time slice t with Metropolis algorithm.
     */
    void Core::metropolis_update(Hubbard& model, int t)
    {
        assert(t >= 0 && t < this->m_nt); // t takes values from 0 to nt-1
        assert(t == this->m_current_t%this->m_nt);
        for (auto i = 0; i < this->m_ns; ++i) {
            // obtain the ratio of flipping the bosonic field at (i,l)
            const auto acceptance_ratio = model.get_acceptance_ratio(*this, t, i);

            if (std::bernoulli_distribution(std::min(1.0, std::abs(acceptance_ratio)))(Utils::Random::Engine))
            {   
                // if accepted, first update the equal-time green's function according to the proposed new config
                model.update_green_function(*this, t, i);

                // then update the ising field at (t,i)
                model.locally_update_ising_field(t, i);
            }
        }
    }

    // B(t) (0<=t<=nt-1) involves the ising field at index t.
    // specially, we define B(t=nt) = B(t=0), such that the range of t can be extended to [0,nt].
    //
    //     t=0        t=1       ...     t=nt-1        t=nt
    //     B(0)       B(1)      ...     B(nt-1)      B(nt)=B(0)
    //    Ising(0)   Ising(1)   ...    Ising(nt-1)   Ising(0)
    //
    
    const int Core::t2index(const int t) {
        // convert t (range in [0,nt]) to indices of ising field (range in [0,nt)), which is also the argument of B 
        assert(t>=0 && t<=this->m_nt);
        return t%this->m_nt;
    }

    /*
     *  Propagate the Green's functions from the current time t to t+1 according to
     * 
     *      G(t+1) = B(t+1) * G(t) * B(t+1)^-1
     * 
     *  The Green's function are changed in place.
     */
    void Core::wrap_from_0_to_beta(const Hubbard& model, int t)
    {
        assert(t >= 0 && t <= this->m_nt);
        model.multiply_B_from_left     (*this->m_gftt, t2index(t+1));
        model.multiply_invB_from_right (*this->m_gftt, t2index(t+1));
    }

    /*
     *  Propagate the Green's function from the current time t to t-1 according to
     * 
     *      G(t-1) = B(t)^-1 * G(t) * B(t)
     * 
     *  The Green's functions are changed in place.
     */
    void Core::wrap_from_beta_to_0(const Hubbard& model, int t)
    {
        assert(t >= 0 && t <= this->m_nt);
        model.multiply_B_from_right   (*this->m_gftt, t2index(t));
        model.multiply_invB_from_left (*this->m_gftt, t2index(t));
    }
    
    /*
     *  Monte Carlo sweep over the spatial-time lattice (from 0 to beta).
     *  Equal-time Green's functions are collected if needed.
     */
    void Core::sweep_from_0_to_beta(Hubbard& model)
    {
        const int stack_length = (this->m_nt % this->m_stabilization_pace == 0)? 
            this->m_nt/this->m_stabilization_pace : this->m_nt/this->m_stabilization_pace+1;
        
        // current_t takes value from 0 to nt (t=0 and t=nt are equivalent due to the PBC)
        assert(this->m_current_t == 0);
        assert(this->m_svdstack_left->empty());
        assert(this->m_svdstack_right->StackLength() == stack_length);

        // temporary stacks for stable multiplications of B matrices
        BStacks stacks = BStacks::Identity(this->m_ng, this->m_ng);

        // sweep upwards from 0 to beta
        for (auto t = 1; t <= this->m_nt; ++t) {
            this->m_current_t++;

            // wrap the Green's function to the current t
            this->wrap_from_0_to_beta(model, t-1);

            // update the ising fields at t and update green's function accordingly
            this->metropolis_update(model, t2index(t));

            model.multiply_B_from_left(stacks, t2index(t));
            
            // perform the stabilizations
            if (t % this->m_stabilization_pace == 0 || t == this->m_nt) {
                // update svd stacks
                this->m_svdstack_right->pop();
                this->m_svdstack_left->push(stacks);

                // collect the wrapping error
                GF temp_gftt = GF::Zero(this->m_ng, this->m_ng);
                double temp_wrap_error_tt = 0.0;

                // compute the Green's function from scratch every 'stabilization_pace' wrapping steps
                // and collect the wrapping error. The equal-time Green's function G(t,t):
                //
                //      G(t,t) = ( 1 + stack_left * stack_right^H )^-1
                //
                // where stack_left = B(t) * ... * B(1) and stack_right = B(t+1)^H * B(t+2)^H * ... * B(nt)^H .
                Utils::StableNumerics::compute_equaltime_gf(*this->m_svdstack_left, *this->m_svdstack_right, temp_gftt);

                // compute the wrapping errors
                Utils::StableNumerics::matrix_compare_error(temp_gftt, *this->m_gftt, temp_wrap_error_tt);
                this->m_equaltime_wrap_error = std::max(this->m_equaltime_wrap_error, temp_wrap_error_tt);

                *this->m_gftt= temp_gftt;

                stacks = BStacks::Identity(this->m_ng, this->m_ng);
            }

            // save the Green's functions to the vector
            //
            //   vecgftt = [ G(0,0), G(dt,dt), G(2dt,2dt), ... , G(beta-dt,beta-dt) ]
            //
            if (!this->m_is_thermalization && this->m_is_equaltime) {
                (*this->m_vecgftt)[t2index(t)] = *this->m_gftt;
            }
        }
    }

    /*
     *  Monte Carlo sweep over the spatial-time lattice (from beta to 0).
     *  Equal-time Green's functions are collected if needed.
     */
    void Core::sweep_from_beta_to_0(Hubbard& model)
    {
        const int stack_length = (this->m_nt % this->m_stabilization_pace == 0)? 
            this->m_nt/this->m_stabilization_pace : this->m_nt/this->m_stabilization_pace+1;
        
        // current_t takes value from 0 to nt (t=0 and t=nt are equivalent due to the PBC)
        assert(this->m_current_t == this->m_nt);
        assert(this->m_svdstack_right->empty());
        assert(this->m_svdstack_left->StackLength() == stack_length);

        // temporary stacks for stable multiplications of B matrices
        BStacks stacks = BStacks::Identity(this->m_ng, this->m_ng);

        // sweep downwards from beta to 0
        for (auto t = this->m_nt; t >= 1; --t) {
            // update the ising fields at t and update green's function accordingly
            this->metropolis_update(model, t2index(t));

            model.multiply_adjB_from_left(stacks, t2index(t));

            // wrap the Green's function to the next t
            this->wrap_from_beta_to_0(model, t);

            // perform the stabilizations
            if ((t-1) % this->m_stabilization_pace == 0) {
                // update svd stacks
                this->m_svdstack_left->pop();
                this->m_svdstack_right->push(stacks);

                // collect the wrapping error
                GF temp_gftt = GF::Zero(this->m_ng, this->m_ng);
                double temp_wrap_error_tt = 0.0;
                Utils::StableNumerics::compute_equaltime_gf(*this->m_svdstack_left, *this->m_svdstack_right, temp_gftt);
                Utils::StableNumerics::matrix_compare_error(temp_gftt, *this->m_gftt, temp_wrap_error_tt);
                this->m_equaltime_wrap_error = std::max(this->m_equaltime_wrap_error, temp_wrap_error_tt);

                *this->m_gftt = temp_gftt;

                stacks = BStacks::Identity(this->m_ng, this->m_ng);
            }

            // save the Green's functions to the vector
            //
            //   vecgftt = [ G(0,0), G(dt,dt), G(2dt,2dt), ... , G(beta-dt,beta-dt) ]
            //
            if (!this->m_is_thermalization && this->m_is_equaltime) {
                (*this->m_vecgftt)[t2index(t-1)] = *this->m_gftt;
            }
            
            this->m_current_t--;
        }
    }

    /*
     *  Sweep over the spatial-time lattice (from 0 to beta) to collect the dynamic Green's functions.
     *  Note that the ising fields remain unchanged during this routine,
     *  and the equal-time Green's functions are also collected for the current ising configs.
     */
    void Core::sweep_for_dynamic_green_functions(Hubbard& model)
    {
        if (this->m_is_dynamic) {
            const int stack_length = (this->m_nt % this->m_stabilization_pace == 0)? 
                this->m_nt/this->m_stabilization_pace : this->m_nt/this->m_stabilization_pace+1;
            
            assert(this->m_current_t == 0);
            assert(this->m_svdstack_left->empty());
            assert(this->m_svdstack_right->StackLength() == stack_length);

            // initialize the dynamic green's functions: for t -> 0, gft0(0) = gftt(0), gf0t = -(1-gftt(0))
            *this->m_gft0 = *this->m_gftt;
            *this->m_gf0t = *this->m_gftt - GF::Identity(this->m_ng, this->m_ng);

            // temporary stacks for stable multiplications of B matrices
            BStacks stacks = BStacks::Identity(this->m_ng, this->m_ng);

            for (auto t = 1; t <= this->m_nt; ++t) {
                this->m_current_t++;

                // compute the equal-time and dynamic Green's functions at current t
                this->wrap_from_0_to_beta(model, t-1);
                model.multiply_B_from_left(*this->m_gft0, t2index(t));
                model.multiply_invB_from_right(*this->m_gf0t, t2index(t));

                model.multiply_B_from_left(stacks, t2index(t));
            
                // perform the stabilizations
                if (t % this->m_stabilization_pace == 0 || t == this->m_nt) {
                    // update svd stacks
                    this->m_svdstack_right->pop();
                    this->m_svdstack_left->push(stacks);

                    // collect the wrapping errors
                    GF temp_gftt = GF::Zero(this->m_ng, this->m_ng);
                    GF temp_gft0 = GF::Zero(this->m_ng, this->m_ng);
                    GF temp_gf0t = GF::Zero(this->m_ng, this->m_ng);
                    double temp_wrap_error_tt = 0.0;
                    double temp_wrap_error_t0 = 0.0;
                    double temp_wrap_error_0t = 0.0;

                    // compute the equal-time/dynamic Green's function from scratch
                    // every 'stabilization_pace' wrapping steps and collect the wrapping error.
                    Utils::StableNumerics::compute_dynamic_gf(*this->m_svdstack_left, *this->m_svdstack_right, temp_gft0, temp_gf0t, temp_gftt);

                    // compute the wrapping errors
                    Utils::StableNumerics::matrix_compare_error(temp_gftt, *this->m_gftt, temp_wrap_error_tt);
                    Utils::StableNumerics::matrix_compare_error(temp_gft0, *this->m_gft0, temp_wrap_error_t0);
                    Utils::StableNumerics::matrix_compare_error(temp_gf0t, *this->m_gf0t, temp_wrap_error_0t);
                    this->m_equaltime_wrap_error = std::max(this->m_equaltime_wrap_error, temp_wrap_error_tt);
                    this->m_dynamic_wrap_error = std::max(this->m_dynamic_wrap_error, std::max(temp_wrap_error_t0, temp_wrap_error_0t));
                    *this->m_gftt = temp_gftt;
                    *this->m_gft0 = temp_gft0;
                    *this->m_gf0t = temp_gf0t;

                    stacks = BStacks::Identity(this->m_ng, this->m_ng);
                }

                // save the Green's functions to the vectors
                //
                //  vecgftt = [ G(0,0), G(dt,dt), G(2dt,2dt), ... , G(beta-dt,beta-dt) ]
                //  vecgft0 = [ G(0,0), G(dt,0), G(2dt,0), ... , G(beta-dt,0) ]
                //  vecgf0t = [ G(0,0), G(0,dt), G(0,2dt), ... , G(0,beta-dt) ]
                //
                // with time grids [ 0, dt, 2dt, ..., beta-dt ].

                // however, during our sweeping procedure, Green's functions are obtained
                // at time grids [ beta, dt, 2dt, ..., beta-dt ], sorted by the order of storage.
                // note that in practice, according to our wrapping rules,
                // the dynamic Green's functions G(beta,0)/G(0,beta) are calculated as,
                //
                //    G(beta,0) = G(beta,beta) * B(beta,0) = (1 + B(beta,0))^-1 * B(beta,0) = G(0,0) * (G(0,0)^-1 - 1) = 1 - G(0,0)
                //    G(0,beta) = -B(beta,0)^-1 * (1-G(0,0)) = -B(beta,0)^-1 * (1 - (1 + B(beta,0))^-1) = -G(0,0)
                //
                // for equal-time ones, G(beta,beta) = G(0,0) automatically.

                // recalling the anti-periodicity of Matsubara Green's functions,
                // it's now plain to see that the Matsubara Green's function $G(\tau)$ has a discontinuity at $\tau=0/beta$;
                // this is consistent with a long-distance behavior of $G(i\omega_n)\sim 1/|\omega_n|$.
                // with this point in mind, we can define
                //
                //      G(t->0,0) = G(0,0)
                //      G(0,t->0) = -(1 - G(0,0))
                //
                // such that both G(t,0) and G(0,t) are continuous in [0,beta).
                // as a consequence of our conventions here,
                // special attentions have to be paid when time-displaced observables are evaluated with Wick's contraction.

                (*this->m_vecgftt)[t2index(t)] = *this->m_gftt;
                if (t2index(t) != 0) {
                    (*this->m_vecgft0)[t2index(t)] = *this->m_gft0;
                    (*this->m_vecgf0t)[t2index(t)] = *this->m_gf0t;
                }
                else {
                    (*this->m_vecgft0)[t2index(t)] = *this->m_gftt;
                    (*this->m_vecgf0t)[t2index(t)] = *this->m_gftt - GF::Identity(this->m_ng, this->m_ng);
                }
            }
        }
    }
}