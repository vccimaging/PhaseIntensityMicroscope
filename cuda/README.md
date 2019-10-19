### Installation

The command line solver `cws` is written with C++ and CUDA, with code hierarchy managed by CMake. 

After successful compilation, you may call the MATLAB wrapper (`../matlab/cws_gpu_wrapper.m`) for this command line solver. See `../matlab/cpu_gpu_comparison.m` for a simple demo. Due to floating point errors, the CPU (`cws.m`) and GPU solvers (`cws`) do not produce exactly the same result, but the difference is small.

#### Prerequisites

- CUDA Toolkit 8.0 or above (We used CUDA 10.0);
- CMake 2.8 or above (We used CMake 3.13);
- OpenCV 3.0 or above (We used OpenCV 3.4.3).

Make sure you complete above dependencies before proceeding.

#### Linux

Additional prerequisites:

- GNU make;
- GNU Compiler Collection (GCC).

Simply run `setup.py` to get it done. You can also build it manually, for example:

```shell
cd <the folder contains this README.md>
mkdir build
cd build
cmake ..
make -j8
sudo make install
```

#### Windows

Here we only provide one compilation approach. You can do any modifications for your convenience at will.

It has been tested using Visual Studio 2017 and CUDA 10.0 on Windows 10.

Steps:

- Add OpenCV to the system `PATH` variable. The specific path depends on your installation option.

- Run `cmake-gui` and configure your Visual Studio Solution. If CMake failed to auto-detect the paths:

  - Manually find the path (depends on specific installation option), and modify `./CMakeLists.txt`;
  - Then, try to configure and generate the project again.

  If success, you will see `CWS.sln` be generated in your build folder.

- In Visual Studio 2017 (opened as Administrator), build the solution. To install, in Visual Studio:

  - Build -> Configuration Manager -> click on Build option for INSTALL;
  - Then build the solution.

  If success, you will see new folders `bin`, `include` and `lib` appear.

### Usage

Run `./cws --help` in command line prompt to see all parameters:

```
cws version 1.0 (Nov 2018) 
Simultaneous intensity and wavefront recovery GPU solver for the coded wavefront sensor. Solve for:

 min            || i(x+\nabla phi) - A i_0(x) ||_2^2            +
A,phi   alpha   || \nabla phi ||_1                              +
        beta  ( || \nabla phi ||_2^2 + || \nabla^2 phi ||_2^2 ) +
        gamma ( || \nabla A ||_1     + || \nabla^2 A ||_1 )     +
        tau   ( || \nabla A ||_2^2   + || \nabla^2 A ||_2^2 ).

Inputs : i_0 (reference), i (measure).
Outputs: A (intensity), phi (phase).

by Congli Wang, VCC Imaging @ KAUST.

Usage: cws [-v] [--help] [--version] [-p <double>]... [-i <int>]... [-m <double>] [-m <double>] [-t <double>] [-s <int>] [-s <int>] [-l <int>] [-l <int>] [-o <.flo>] [-f <files>] [-f <files>]
  --help                    display this help and exit  
  --version                 display version info and exit  
  -p, --priors=<double>     prior weights {alpha,beta,gamma,beta} (default {0.1,0.1,100,5})  
  -i, --iter=<int>          iteartions {total alternating iter, A-update iter, phi-update iter} (default {3,20,20})  
  -m, --mu=<double>         ADMM parameters {mu_A,mu_phi} (default {0.1,100})  
  -t, --tol_phi=<double>    phi-update tolerance stopping criteria (default 0.05)  
  -v, --verbose             verbose output (default 0)  
  -s, --size=<int>          output size {width,height} (default input size)  
  -l, --L=<int>             padding size {pad_width,pad_height} (default nearest power of 2 of out_size, each in range [2, 32])  
  -o, --output=<.flo>       save output file (intensity A & wavefront phi) as *.flo file (default "./out.flo")  
  -f, --files=<files>       input file names (reference & measurement) 
```
