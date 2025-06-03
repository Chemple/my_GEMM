# nvcc -ptx -O3 -std=c++20 -arch=sm_86 compare_ptx_gemm1.cu
# nvcc -ptx -O3 -std=c++20 -arch=sm_86 compare_ptx_gemm2.cu
nvcc -ptx -Xcompiler -fopenmp -O3 -std=c++20 -arch=sm_86 naiveSGEMM.cu -lcublas
