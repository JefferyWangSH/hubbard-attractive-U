# hubbard-attractive-U
![workflow](https://github.com/JefferyWangSH/hubbard-attractive-U/actions/workflows/build.yml/badge.svg?branch=master)

Determinant quantum Monte Carlo of attractive-U Hubbard model on the square lattice.
A variant of this code is devoted to generating the DQMC results in [arXiv.2212.05737](https://arxiv.org/abs/2212.05737). 

## Dependences
  * g++ (C++17)
  * cmake (>=3.21)
  * boost (>=1.71)
  * [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) (>=3.4)
  * [xtensor](https://github.com/xtensor-stack/xtensor)
  * Intel oneAPI (MKL)
  * Intel MPI (included in oneAPI) or OpenMPI

## Usage
 * Build the project with cmake/make.
    ```shell
      mkdir build
      cd build
      cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
      make -j4
    ```
   when the building process finished, a binary file `hubbard-attractive-u` would be generated under the `build` folder.

 * The program receives a *toml file* (parsed using the open-sourced software [toml++](https://github.com/marzer/tomlplusplus)) as the input of program parameters.
See [`config.toml`](config.toml) and check the definitions of parameters therein.

 * Run the simulation with your customized `config.toml` file, and pass it to the program through the program option `-c,--config`. Furthermore, it also receives other options from the command line, e.g.,
   * `-o,--output:` assign the path of folder where the program output are saved.
   * `-f,--fields:` assign the path of file where initial configurations of the auxiliary Ising fields are saved. (if not assigned, the field configurations are going to be set up randomly.)
   * `-h,--help:` show the helping message.
 
 * Run the program from the command line as
   ```shell
   build/hubbard-attractive-u -c ${config} -o ${output} -f ${fields}
   ```
   and the status of the simulation will be displayed in real-time on the console.
   Once the simulation finished, check the output folder for the results.
   (e.g. see [`example`](example/) for the typical program output.)
 * To execute the program in parallel with MPI, just run it as
   ```shell
   mpirun -np 4 build/hubbard-attractive-u -c ${config} -o ${output} -f ${fields} > ${output}/log.out 2>&1
   ```
   This time, the status information, and also the possible error info, will be saved to `${output}/log.out`.

## License
This repository is open-sourced under the MIT License.