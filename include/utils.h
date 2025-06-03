#pragma once
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda.h>
#include <iostream>
#include <type_traits>
#include <vector>

// NOTE(shiwen): row-major order storage.
#define OFFSET(row_idx, col_idx, col_num) ((row_idx) * (col_num) + (col_idx))

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(stat)                                                     \
  {                                                                            \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": "   \
                << stat << std::endl;                                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

template <typename data_type = float, int32_t M, uint32_t N, uint32_t K>
float get_cublas_gemm_time(data_type *h_A, data_type *h_B, data_type *h_C,
                           uint32_t repeat = 4) {
  data_type *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void **)&d_A, M * K * sizeof(data_type)));
  CUDA_CHECK(cudaMalloc((void **)&d_B, K * N * sizeof(data_type)));
  CUDA_CHECK(cudaMalloc((void **)&d_C, M * N * sizeof(data_type)));

  CUDA_CHECK(
      cudaMemcpy(d_A, h_A, M * K * sizeof(data_type), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B, h_B, K * N * sizeof(data_type), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  const data_type alpha = 1.0f;
  const data_type beta = 0.0f;

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start));
  for (uint32_t i = 0; i < repeat; ++i) {
    // C = A * B (A: MxK, B: KxN, C: MxN)
    if constexpr (std::is_same_v<data_type, float>) {
      // float -> cublasSgemm
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                               &alpha, d_B, N, d_A, K, &beta, d_C, N));
    } else if constexpr (std::is_same_v<data_type, double>) {
      // double -> cublasDgemm
      CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                               &alpha, d_B, N, d_A, K, &beta, d_C, N));
    } else {
      static_assert(sizeof(data_type) == 0,
                    "Unsupported data type for cuBLAS GEMM");
    }
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  float msec_total;
  CUDA_CHECK(cudaEventElapsedTime(&msec_total, start, end));
  float msec_avg = msec_total / repeat;

  CUDA_CHECK(
      cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return msec_avg;
}

// CPU ref
template <typename data_type, uint32_t M, uint32_t N, uint32_t K>
float get_cpu_gemm_time(data_type *A, data_type *B, data_type *C) {
  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2)
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      data_type sum = 0;
      for (uint32_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

template <typename T>
void compute_error(const std::vector<T> &result0, const std::vector<T> &result1,
                   T &max_error, T &avg_error) {
  size_t size = result0.size();
  T total = 0;
  max_error = 0;

  for (size_t i = 0; i < size; ++i) {
    T diff = std::abs(result0[i] - result1[i]);
    total += diff;
    if (diff > max_error) {
      max_error = diff;
    }
  }
  avg_error = total / size;
}

template <typename T>
void print_vector(const std::vector<T> &vec, const uint32_t &from = 0,
                  const uint32_t &to = 99) {
  assert(from <= to);
  assert(to < vec.size());
  for (uint32_t i = from; i <= to; i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << "\n";
}