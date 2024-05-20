/*
 *   main.cpp
 * 
 *     Created on: May 7, 2024
 *         Author: Jeffery Wang
 * 
 */

#include <string>
#include <iostream>
#include <fstream>

#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/local_time/local_time.hpp>
#include <boost/program_options.hpp>

#include "dqmc_handle.h"
#include "dqmc_core.h"
#include "measure_handle.h"
#include "hubbard.h"
#include "square_lattice.h"
#include "random.h"
#include "dqmc_parser.hpp"
#include "dqmc_io.hpp"
#include "utils/observable_gather.hpp"

int main(int argc, char* argv[]) {

    // -------------------------------------------------------------------------------------------------------------
    //                                         Initialize MPI environment
    // -------------------------------------------------------------------------------------------------------------
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const int master = 0;
    const int rank = world.rank();

    // -------------------------------------------------------------------------------------------------------------
    //                                               Program options
    // -------------------------------------------------------------------------------------------------------------
    std::string config;
    std::string output;
    std::string ising_fields;

    boost::program_options::options_description opts("Program options");
    boost::program_options::variables_map vm;

    opts.add_options()
        ("help,h", "display this information.")
        ("config,c",
         boost::program_options::value<std::string>(&config)->default_value("./example/config.toml"),
         "path of the toml configuration file, default: ./example/config.toml.")
        ("output,o",
         boost::program_options::value<std::string>(&output)->default_value("./example"),
         "path of the folder where the program output are saved, default: ./example.")
        ("fields,f",
         boost::program_options::value<std::string>(&ising_fields),
         "path of the file where initial configurations of the auxiliary Ising fields are saved. (if not assigned, the field configurations are going to be set up randomly.)");

    // parse the command line options
    try { boost::program_options::store(parse_command_line(argc, argv, opts), vm); }
    catch (...) {
        if (rank == master) {
            throw std::runtime_error("main(): invalid program options got from the command line.");
        }
    }
    boost::program_options::notify(vm);

    // show the helping messages
    if (rank == master && vm.count("help")) {
        std::cerr << argv[0] << "\n" << opts << std::endl;
        return 0;
    }
    
    // initialize the output folder, i.e. create it if not exist
    if (rank == master) {
        if (access(output.c_str(), 0) != 0) {
            const std::string command = "mkdir -p " + output;
            if (system(command.c_str()) != 0) {
                throw std::runtime_error(
                    boost::str(boost::format("main(): fail to creat folder at '%s'.") % output)
                );
            }
        }
    }

    // -------------------------------------------------------------------------------------------------------------
    //                                        Output current date and time
    // -------------------------------------------------------------------------------------------------------------
    if (rank == master) {
        const auto current_time = boost::posix_time::second_clock::local_time();
        std::cout << boost::format(">> Current time: %s\n") % current_time << std::endl;
    }

    // -------------------------------------------------------------------------------------------------------------
    //                                        Output MPI and hardware info
    // -------------------------------------------------------------------------------------------------------------
    if (rank == master) {
        // print the MPI and hardware information
        boost::format fmt_mpi(">> Distribute tasks to %s processors, with the master processor being %s.\n");
        std::cout << fmt_mpi % world.size() % env.processor_name() << std::endl;
    }

    // -------------------------------------------------------------------------------------------------------------
    //                                              DQMC simulation
    // -------------------------------------------------------------------------------------------------------------
    // set up random seeds for different processes
    Utils::Random::set_seed(std::time(nullptr) + rank);

    // -------------------------------------------  Initializations  -----------------------------------------------    
    // parse parmas from the configuation file
    DQMC::Params* params = new DQMC::Params();
    DQMC::Parser::parse_toml_config(config, world.size(), *params);

    // create modules
    DQMC::Core* core = new DQMC::Core();
    Measurement::Handle* meas_handle = new Measurement::Handle();
    Model::HubbardAttractiveU* model = new Model::HubbardAttractiveU();
    Lattice::SquareLattice* lattice = new Lattice::SquareLattice();

    // initialize the modules, NOTE: the orders are important.
    lattice->initialize(*params);
    model->initialize(*params, *lattice);
    meas_handle->initialize(*params, *lattice);
    core->initialize(*params, *meas_handle);

    // if field configurations provided, load the configurations from file.
    // otherwise, initialize the ising fields randomly.
    if (!ising_fields.empty()) {
        DQMC::IO::load_ising_fields_configs(ising_fields, *model);
        if (rank == master) { 
            std::cout << ">> Configurations of ising fields loaded from file.\n" << std::endl;
        }
    }
    else {
        model->set_ising_fields_to_random();
        if (rank == master) {
            std::cout << ">> Configurations of ising fields initialized randomly.\n" << std::endl;
        }
    }

    // initialize svdstacks and Green's functions
    core->initialize_svdstacks(*model);
    core->initialize_green_functions();

    if (rank == master) {
        std::cout << ">> Initialization finished.\n\n" 
                  << ">> The simulation is going to get started with parameters below:\n" << std::endl;
        // output the initialization info
        DQMC::IO::print_initialization_info(std::cout, *params, *meas_handle, world.size());
    }

    // set up progress bar
    DQMC::Handle::show_progress_bar((rank == master));
    DQMC::Handle::progress_bar_format(50, '=', ' ');
    DQMC::Handle::set_refresh_rate(10);

    // ----------------------------------  Crucial steps of the simulation  ----------------------------------------
    // DQMC simulation get started
    DQMC::Handle::timer_begin();
    DQMC::Handle::thermalize(*core, *meas_handle, *model, *lattice);
    DQMC::Handle::measure(*core, *meas_handle, *model, *lattice);

    // gather measurement results from other processors
    Utils::MPI::gather_measurement_results_from_processors(world, *meas_handle);

    // perform the statistical analysis
    DQMC::Handle::analyse(*meas_handle);

    // end the timer
    DQMC::Handle::timer_end();

    // output the ending info
    if (rank == master) {
        DQMC::IO::print_dqmc_summary(std::cout, *core, *meas_handle);
    }

    // ------------------------------------  Output measurement results  -------------------------------------------
    if (rank == master) {
        std::ofstream ofile;
        auto reopen = [](std::ofstream& ofile, std::string file, std::ios_base::openmode mode) {
            ofile.close();
            ofile.clear();
            ofile.open(file, mode);
        };

        if (meas_handle->is_found("FillingNumber")) {
            const auto filling_num = meas_handle->find("FillingNumber");
            reopen(ofile, std::string(output+"/filling.out"), std::ios::out|std::ios::trunc);
            DQMC::IO::print_observable(ofile, filling_num, false);
            reopen(ofile, std::string(output+"/filling.bins.out"), std::ios::out|std::ios::app);
            DQMC::IO::print_observable_data(ofile, filling_num, false);
            // // file output in the npy format
            // DQMC::IO::save_obsevable_data_to_file(filling_num, std::string(output+"/filling_num.npy"), "npy");
        }

        if (meas_handle->is_found("DoubleOccupation")) {
            const auto double_occu = meas_handle->find("DoubleOccupation");
            reopen(ofile, std::string(output+"/double_occu.out"), std::ios::out|std::ios::trunc);
            DQMC::IO::print_observable(ofile, double_occu, false);
            reopen(ofile, std::string(output+"/double_occu.bins.out"), std::ios::out|std::ios::app);
            DQMC::IO::print_observable_data(ofile, double_occu, false);
        }

        if (meas_handle->is_found("DynamicGreenFunctions")) {
            const auto green_func = meas_handle->find("DynamicGreenFunctions");
            reopen(ofile, std::string(output+"/gf.out"), std::ios::out|std::ios::trunc);
            DQMC::IO::print_observable(ofile, green_func, false);
            reopen(ofile, std::string(output+"/gf.bins.out"), std::ios::out|std::ios::app);
            DQMC::IO::print_observable_data(ofile, green_func, false);
        }

        if (meas_handle->is_found("LocalDensityOfStates")) {
            const auto ldos = meas_handle->find("LocalDensityOfStates");
            reopen(ofile, std::string(output+"/ldos.out"), std::ios::out|std::ios::trunc);
            DQMC::IO::print_observable(ofile, ldos, false);
            reopen(ofile, std::string(output+"/ldos.bins.out"), std::ios::out|std::ios::app);
            DQMC::IO::print_observable_data(ofile, ldos, false);
        }

        if (meas_handle->is_found("StaticSWavePairingCorrelation")) {
            const auto swave_pairing = meas_handle->find("StaticSWavePairingCorrelation");
            reopen(ofile, std::string(output+"/swave_corr.out"), std::ios::out|std::ios::trunc);
            DQMC::IO::print_observable(ofile, swave_pairing, false);
            reopen(ofile, std::string(output+"/swave_corr.bins.out"), std::ios::out|std::ios::app);
            DQMC::IO::print_observable_data(ofile, swave_pairing, false);
        }

        if (meas_handle->is_found("LocalDynamicTransverseSpinCorrelation")) {
            const auto spin_corr = meas_handle->find("LocalDynamicTransverseSpinCorrelation");
            reopen(ofile, std::string(output+"/spin_corr.out"), std::ios::out|std::ios::trunc);
            DQMC::IO::print_observable(ofile, spin_corr, false);
            reopen(ofile, std::string(output+"/spin_corr.bins.out"), std::ios::out|std::ios::app);
            DQMC::IO::print_observable_data(ofile, spin_corr, false);
        }

        // output ising field configs
        const std::string ising_fields_out = (ising_fields.empty())? output+"/ising.fields" : ising_fields;
        DQMC::IO::save_ising_fields_configs(ising_fields_out, *model);

        // output momentum list and imaginary-time grids
        reopen(ofile, std::string(output+"/momenta.out"), std::ios::out|std::ios::trunc);
        DQMC::IO::print_momentum_list(ofile, *meas_handle, *lattice);

        reopen(ofile, std::string(output+"/tgrids.out"), std::ios::out|std::ios::trunc);
        DQMC::IO::print_imaginary_time_grids(ofile, *params);

        std::cout << boost::format(">> See the results under folder '%s'.") % output << std::endl;
    }

    delete params;
    delete core;
    delete meas_handle;
    delete model;
    delete lattice;

    return 0;
}