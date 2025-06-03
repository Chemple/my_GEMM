nvcc -Xcompiler -fopenmp -G -std=c++20 -arch=sm_86 naiveSGEMM.cu -lcublas -o naiveSGEMM_debug
nvcc -Xcompiler -fopenmp -O3 -std=c++20 -arch=sm_86 naiveSGEMM.cu -lcublas -o naiveSGEMM