This is a Cuda demo.
# Building
## Python package
```
cd self_attention
bash build_python_package.sh
```

## C executable program
First, ensure that your CUDA compiler can be located properly.
Type `nvcc -V`

Shows like: 
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Oct_12_20:09:46_PDT_2020
Cuda compilation tools, release 11.1, V11.1.105
Build cuda_11.1.TC455_06.29190527_0
```

Modify your conda python env path and some other config in self_attention/CMakeLists.txt like below
```
set(CONDA_ENV_PATH "/home/felix/miniconda3/envs/local_mnr388")
set(PYBIND11_INCLUDE_DIR  "${CONDA_ENV_PATH}/lib/python3.8/site-packages/torch/include")
set(PYTHON_INCLUDE_DIR "${CONDA_ENV_PATH}/include/python3.8")
set(PYTHON_LIBRARY_DIR "${CONDA_ENV_PATH}/lib")

...
find_library(PYTHON_LIB python3.8 PATHS ${PYTHON_LIBRARY_DIR} NO_DEFAULT_PATH)
```
Run the build shell.
```
cd self_attention
bash build_pure_c_version.sh
```


# Running
## Python demo
```
cd self_attention
python use_demo.py
```

## C executable program
```
./use_demo
```

