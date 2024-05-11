/*
 *   dqmc_handle.h
 * 
 *     Created on: Aug 3, 2023
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef DQMC_HANDLE_H
#define DQMC_HANDLE_H

#include <chrono>

namespace Model { class HubbardAttractiveU; }
namespace Lattice { class SquareLattice; }
namespace Measurement { class Handle; }

namespace DQMC {

    class Core;

    // ----------------------------------------------  DQMC::Handle  ---------------------------------------------------
    class Handle {
        public:
            using DqmcCore = Core;
            using MeasureHandle = ::Measurement::Handle;
            using Hubbard = ::Model::HubbardAttractiveU;
            using Lattice = ::Lattice::SquareLattice;

            // --------------------------------------  Useful tools  -----------------------------------------------
            // set up whether to show the process bar or not
            static void show_progress_bar(bool show_progress_bar);

            // set up the format of the progress bar
            static void progress_bar_format(unsigned int width, char complete, char incomplete);

            // set up the rate of refreshing the progress bar
            static void set_refresh_rate(unsigned int refresh_rate);
            
            // return the duration time of the QMC procedure, e.g thermalization or measurements
            static const double timer();

            // start the timer
            static void timer_begin();
            
            // end the timer
            static void timer_end();

            // ----------------------------------  Crucial QMC routines  -------------------------------------------
            // thermalization of the ising fields
            static void thermalize(DqmcCore& core, const MeasureHandle& meas_handle, Hubbard& model, const Lattice& lattice);

            // perform MC sweeps and do the measurments
            static void measure(DqmcCore& core, MeasureHandle& meas_handle, Hubbard& model, const Lattice& lattice);

            // analyse the MC samples
            static void analyse(MeasureHandle& meas_handle);

        private:
            // ------------------------------  Declarations of static members  -------------------------------------
            static bool m_show_progress_bar;
            static unsigned int m_progress_bar_width;
            static unsigned int m_refresh_rate;
            static char m_progress_bar_complete_char, m_progress_bar_incomplete_char;
            static std::chrono::steady_clock::time_point m_begin_time, m_end_time;
    };
}

#endif