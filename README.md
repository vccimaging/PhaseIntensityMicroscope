## Quantitative Phase and Intensity Microscope
This is the open source repository for our paper to appear in Scientific Reports:

[**Quantitative Phase and Intensity Microscopy Using Snapshot White Light Wavefront Sensing**](<https://vccimaging.org/Publications/Wang2019QPM/Wang2019QPM.pdf>)

[Congli Wang](https://congliwang.github.io), [Qiang Fu](http://vccimaging.org/People/fuq/), [Xiong Dun](http://vccimaging.org/People/dunx/), and [Wolfgang Heidrich](http://vccimaging.org/People/heidriw/)

King Abdullah University of Science and Technology (KAUST)

### Overview

This repository contains:

- An improved version of the wavefront solver in [1], implemented in MATLAB and CUDA:
  - For MATLAB code, simply plug & play.
  - For the CUDA solver, you need an NVIDIA graphics card with CUDA to compile and run. Also refer to [`./cuda/README.md`](./cuda/README.md) for how to compile the code.
- Other solvers [2, 3] (our implementation).
- Scripts and data for generating Figure 2, Figure 3, and Figure 4 in the paper.

Our solver is not multi-scale because microscopy wavefronts are small; however based on our solver it is simple to implement such a pyramid scheme.

### Related

- Sensor principle: [The Coded Wavefront Sensor](https://vccimaging.org/Publications/Wang2017CWS/>) (Optics Express 2017).
- An adaptive optics application using this sensor: [Megapixel Adaptive Optics](<https://vccimaging.org/Publications/Wang2018AdaptiveOptics/>) (SIGGRAPH 2018).
- Sensor simulation or old solvers, refer to repository <https:github.com/vccimaging/MegapixelAO>.


### Citation

```bibtex
@article{wang2019quantitative,
  title = {Quantitative Phase and Intensity Microscopy Using Snapshot White Light Wavefront Sensing},
  author = {Wang, Congli and Fu, Qiang and Dun, Xiong and Heidrich, Wolfgang},
  journal = {Scientific Reports},
  volume = {},
  pages = {},
  year = {2019},
  publisher = {Nature Publishing Group}
}
```

### Contact

We welcome any questions or comments. Please either open up an issue, or email to congli.wang@kaust.edu.sa.

### References

[1] Congli Wang, Qiang Fu, Xiong Dun, and Wolfgang Heidrich. "Ultra-high resolution coded wavefront sensor." *Optics Express* 25.12 (2017): 13736-13746.

[2] Pascal Berto, Hervé Rigneault, and Marc Guillon. "Wavefront sensing with a thin diffuser." *Optics Letters* 42.24 (2017): 5117-5120.

[3] Sebastien Berujon and Eric Ziegler. "Near-field speckle-scanning-based X-ray imaging." *Physical Review A* 92.1 (2015): 013837.