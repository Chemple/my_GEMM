#include <cassert>
#include <cstdint>
#include <random>
#include <sys/types.h>
#include <vector>

#include "./include/utils.h"

// template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
//           size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
//           size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
//           size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
// __device__ void load_data_from_global_memory_to_shared_memory(
//     T const *A, size_t lda, T const *B, size_t ldb,
//     T A_thread_block_tile[BLOCK_TILE_SIZE_Y]
//                          [BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
//     T B_thread_block_tile[BLOCK_TILE_SIZE_K]
//                          [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
//     size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t
//     n, size_t k) {
//   // Load data from A on DRAM to A_thread_block_tile on shared memory.
// #pragma unroll
//   for (size_t load_idx{0U};
//        load_idx <
//        (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
//        NUM_THREADS;
//        ++load_idx) {
//     size_t const A_thread_block_tile_row_idx{
//         (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
//     size_t const A_thread_block_tile_col_idx{
//         (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
//     size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
//                            A_thread_block_tile_row_idx};
//     size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
//                            A_thread_block_tile_col_idx};

//     // These boundary checks might slow down the kernel to some extent.
//     // But they guarantee the correctness of the kernel for all
//     // different GEMM configurations.
//     T val{static_cast<T>(0)};
//     if (A_row_idx < m && A_col_idx < k) {
//       val = A[A_row_idx * lda + A_col_idx];
//     }
//     // This if will slow down the kernel.
//     // Add static asserts from the host code to guarantee this if is
//     // always true.
//     static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
//     // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
//     //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
//     // {
//     //     A_thread_block_tile[A_thread_block_tile_row_idx]
//     //                        [A_thread_block_tile_col_idx] = val;
//     // }
//     A_thread_block_tile[A_thread_block_tile_row_idx]
//                        [A_thread_block_tile_col_idx] = val;
//   }
// // Load data from B on DRAM to B_thread_block_tile on shared memory.
// #pragma unroll
//   for (size_t load_idx{0U};
//        load_idx <
//        (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
//        NUM_THREADS;
//        ++load_idx) {
//     size_t const B_thread_block_tile_row_idx{
//         (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
//     size_t const B_thread_block_tile_col_idx{
//         (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};
//     size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
//                            B_thread_block_tile_row_idx};
//     size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
//                            B_thread_block_tile_col_idx};

//     // These boundary checks might slow down the kernel to some extent.
//     // But they guarantee the correctness of the kernel for all
//     // different GEMM configurations.
//     T val{static_cast<T>(0)};
//     if (B_row_idx < k && B_col_idx < n) {
//       val = B[B_row_idx * ldb + B_col_idx];
//     }
//     // This if will slow down the kernel.
//     // Add static asserts from the host code to guarantee this if is
//     // always true.
//     static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
//     // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
//     //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
//     // {
//     //     B_thread_block_tile[B_thread_block_tile_row_idx]
//     //                        [B_thread_block_tile_col_idx] = val;
//     // }
//     B_thread_block_tile[B_thread_block_tile_row_idx]
//                        [B_thread_block_tile_col_idx] = val;
//   }
// }

// the naive impl of single precise GEMM
// C = AB
// A(M x K)
// B(K x N)
// C(M x N)
// use M x N threads, each thread is assigned to calculate one element of C.
template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K>
__global__ void naiveSGEMM(data_type *__restrict__ d_A,
                           data_type *__restrict__ d_B,
                           data_type *__restrict__ d_C) {
  uint32_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_idx < M && y_idx < N) {
    data_type sum = 0.0f;
#pragma unroll
    for (uint32_t k_idx = 0; k_idx < K; k_idx++) {
      sum += d_A[OFFSET(x_idx, k_idx, K)] * d_B[OFFSET(k_idx, y_idx, N)];
    }
    d_C[OFFSET(x_idx, y_idx, N)] = sum;
  }
}

template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K>
__global__ void naiveSGEMM1(data_type *__restrict__ d_A,
                            data_type *__restrict__ d_B,
                            data_type *__restrict__ d_C) {

  // NOTE(shiwen): the only difference.❕❕❕
  uint32_t y_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_idx < M && y_idx < N) {
    data_type sum = 0.0f;
#pragma unroll
    for (uint32_t k_idx = 0; k_idx < K; k_idx++) {
      sum += d_A[OFFSET(x_idx, k_idx, K)] * d_B[OFFSET(k_idx, y_idx, N)];
    }
    d_C[OFFSET(x_idx, y_idx, N)] = sum;
  }
}

template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K>
__global__ void naiveSGEMM2(data_type *__restrict__ d_A,
                            data_type *__restrict__ d_B,
                            data_type *__restrict__ d_C) {

  // NOTE(shiwen): the only difference.❕❕❕
  // 1d layout
  uint32_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t x_idx = global_thread_idx / N;
  uint32_t y_idx = global_thread_idx % N;

  // NOTE(shiwen): note this!
  if (global_thread_idx < M * N) {
    data_type sum = 0.0f;
#pragma unroll
    for (uint32_t k_idx = 0; k_idx < K; k_idx++) {
      sum += d_A[OFFSET(x_idx, k_idx, K)] * d_B[OFFSET(k_idx, y_idx, N)];
    }
    d_C[OFFSET(x_idx, y_idx, N)] = sum;
  }
}

#define SMEM(address, smem_col_shape, smem_row_offset, smem_col_offset)        \
  address[((smem_row_offset) * (smem_col_shape) + (smem_col_offset))]

#define GMEM(address, gmem_col_shape, gmem_row_offset, gmem_col_offset)        \
  address[((gmem_row_offset) * (gmem_col_shape) + (gmem_col_offset))]

// https://siboehm.com/assets/img/CUDA-MMM/cache-blocking.png
// NOTE(shiwen): for simplification the shape of block and the shape of shared
// memory is same, just be (block_len * block_len). so element or thread in
// A,B,C can be index by (row_thread_idx, col_thread_idx)
template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t block_len, uint32_t block_size>
__global__ void sgemm_shared_mem_block(const data_type *__restrict__ d_A,
                                       const data_type *__restrict__ d_B,
                                       data_type *__restrict__ d_C) {
  static_assert(block_len * block_len == block_size);

  uint32_t row_block_idx = blockIdx.y;
  uint32_t col_block_idx = blockIdx.x;

  const data_type *A_block = d_A + row_block_idx * block_len * K;
  const data_type *B_block = d_B + col_block_idx * block_len;
  data_type *C_block =
      d_C + row_block_idx * block_len * N + col_block_idx * block_len;

  data_type __shared__ A_block_smem[block_len * block_len];
  data_type __shared__ B_block_smem[block_len * block_len];

  // TODO(shiwen): check this to enable coalesced memory access.
  // NOTE(shiwen): block_dim is 2d
  uint32_t row_thread_idx = threadIdx.y;
  uint32_t col_thread_idx = threadIdx.x;

  data_type sum = 0;
  // tiling
  // NOTE(shiwen): the shape of A,B,C must be multiple of block_x_dim and
  // block_y_dim.
  // TODO(shiwen): check this.
  constexpr uint32_t num = K / block_len;
  for (uint32_t idx = 0; idx < num; idx++) {
    // each thread is assigned to load one element from global memory to shared
    // memory.
    SMEM(A_block_smem, block_len, row_thread_idx, col_thread_idx) =
        GMEM(A_block, K, row_thread_idx, col_thread_idx);
    SMEM(B_block_smem, block_len, row_thread_idx, col_thread_idx) =
        GMEM(B_block, N, row_thread_idx, col_thread_idx);

    A_block += block_len;
    B_block += block_len * N;

    __syncthreads();

    // different from load element to shared memory, each thread is assinged to
    // calculate its correspoding row,col
    for (uint32_t i = 0; i < block_len; i++) {
      sum += SMEM(A_block_smem, block_len, row_thread_idx, i) *
             SMEM(B_block_smem, block_len, i, col_thread_idx);
    }

    __syncthreads();
  }

  GMEM(C_block, N, row_thread_idx, col_thread_idx) = sum;
}

// https://siboehm.com/assets/img/CUDA-MMM/cache-blocking.png
// NOTE(shiwen): for simplification the shape of block and the shape of shared
// memory is same, just be (block_len * block_len). so element or thread in
// A,B,C can be index by (row_thread_idx, col_thread_idx)
template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BM, uint32_t BN, uint32_t BK, uint32_t block_size>
__global__ void sgemm_shared_mem_block_normal(const data_type *__restrict__ d_A,
                                              const data_type *__restrict__ d_B,
                                              data_type *__restrict__ d_C) {
  static_assert(BM * BN == block_size, "the shape is len * len");

  // TODO(shiwen): change this.
  uint32_t row_block_idx = blockIdx.y;
  uint32_t col_block_idx = blockIdx.x;

  const data_type *A_block = d_A + row_block_idx * BM * K;
  const data_type *B_block = d_B + col_block_idx * BN;
  data_type *C_block = d_C + row_block_idx * BM * N + col_block_idx * BN;

  data_type __shared__ A_block_smem[BM * BK];
  data_type __shared__ B_block_smem[BK * BN];

  // TODO(shiwen): check this to enable coalesced memory access.
  // NOTE(shiwen): block_dim is 2d
  uint32_t row_thread_idx = threadIdx.y;
  uint32_t col_thread_idx = threadIdx.x;

  data_type sum = 0;
  // tiling
  // NOTE(shiwen): the shape of A,B,C must be multiple of block_x_dim and
  // block_y_dim.
  // TODO(shiwen): check this.
  constexpr uint32_t num = K / BK;
  for (uint32_t i = 0; i < num; i++) {
    // load A block and B block from global memory to shared memory.

#pragma unroll
    // A block: BM * BK
    // NOTE(shiwen): a_row_idx = row_thread_idx
    for (uint32_t a_col_idx = col_thread_idx; a_col_idx < BK; a_col_idx += BN) {
      SMEM(A_block_smem, BK, row_thread_idx, a_col_idx) =
          GMEM(A_block, K, row_thread_idx, a_col_idx);
    }

    // B block: BK * BN
    // NOTE(shiwen): b_col_idx = col_thread_idx
    for (uint32_t b_row_idx = row_thread_idx; b_row_idx < BK; b_row_idx += BM) {
      SMEM(B_block_smem, BN, b_row_idx, col_thread_idx) =
          GMEM(B_block, N, b_row_idx, col_thread_idx);
    }

    A_block += BK;
    B_block += BK * N;

    __syncthreads();

    for (uint32_t k = 0; k < BK; k++) {
      sum += SMEM(A_block_smem, BK, row_thread_idx, k) *
             SMEM(B_block_smem, BN, k, col_thread_idx);
    }

    __syncthreads();
  }

  GMEM(C_block, N, row_thread_idx, col_thread_idx) = sum;
}

// NOTE(shiwen): ❗️❗️❗️❗️❗️❗️the shit code 💩💩💩, do not use
// ❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️
template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BM, uint32_t BN, uint32_t BK, uint32_t TM,
          uint32_t block_size>
__global__ void sgemm_block_tiling_1D(const data_type *__restrict__ d_A,
                                      const data_type *__restrict__ d_B,
                                      data_type *__restrict__ d_C) {
  // static_assert(0 == 1, "the shit code 💩💩💩, do not use");
  static_assert(BM * BN / TM == block_size, "the shape is len * len");
  static_assert(BM % TM == 0, "for shape correctness");

  // TODO(shiwen): change this.
  uint32_t row_block_idx = blockIdx.y;
  uint32_t col_block_idx = blockIdx.x;

  const data_type *A_block = d_A + row_block_idx * BM * K;
  const data_type *B_block = d_B + col_block_idx * BN;
  data_type *C_block = d_C + row_block_idx * BM * N + col_block_idx * BN;

  data_type __shared__ A_block_smem[BM * BK];
  data_type __shared__ B_block_smem[BK * BN];

  // TODO(shiwen): check this to enable coalesced memory access.
  // NOTE(shiwen): block_dim is 2d
  uint32_t row_thread_idx = threadIdx.y;
  uint32_t col_thread_idx = threadIdx.x;

  data_type thread_results[BM / TM] = {0.0};

  constexpr uint32_t num = K / BK;
  for (uint32_t i = 0; i < num; i++) {
    // load A block and B block from global memory to shared memory.
#pragma unroll
    // A block: BM * BK
    for (uint32_t a_row_idx = row_thread_idx; a_row_idx < BM; a_row_idx += TM) {
      for (uint32_t a_col_idx = col_thread_idx; a_col_idx < BK;
           a_col_idx += BN) {
        SMEM(A_block_smem, BK, a_row_idx, a_col_idx) =
            GMEM(A_block, K, a_row_idx, a_col_idx);
      }
    }

    // B block: BK * BN
    // NOTE(shiwen): b_col_idx = col_thread_idx
    for (uint32_t b_row_idx = row_thread_idx; b_row_idx < BK; b_row_idx += TM) {
      SMEM(B_block_smem, BN, b_row_idx, col_thread_idx) =
          GMEM(B_block, N, b_row_idx, col_thread_idx);
    }

    A_block += BK;
    B_block += BK * N;

    __syncthreads();

    for (uint32_t k = 0; k < BK; k++) {
      data_type register_b_elem = SMEM(B_block_smem, BN, k, col_thread_idx);
#pragma unroll
      // inner loop
      for (uint32_t block_tiling_row_idx = 0; block_tiling_row_idx < BM / TM;
           block_tiling_row_idx++) {
        thread_results[block_tiling_row_idx] +=
            register_b_elem * SMEM(A_block_smem, BK,
                                   block_tiling_row_idx * TM + row_thread_idx,
                                   k);
      }
    }

    __syncthreads();
  }

  for (uint32_t res_idx = 0; res_idx < BM / TM; res_idx++) {
    GMEM(C_block, N, row_thread_idx + res_idx * TM, col_thread_idx) =
        thread_results[res_idx];
  }
}

// https://siboehm.com/assets/img/CUDA-MMM/kernel_5_2D_blocktiling.png
template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BM, uint32_t BN, uint32_t BK, uint32_t TM, uint32_t TN,
          uint32_t block_size>
__global__ void sgemm_block_tiling_2D(const data_type *__restrict__ d_A,
                                      const data_type *__restrict__ d_B,
                                      data_type *__restrict__ d_C) {
  static_assert(BM / TM * BN / TN == block_size, "the shape is len * len");
  static_assert(BM % TM == 0, "for shape correctness");
  static_assert(BN % TN == 0, "for shape correctness");

  // TODO(shiwen): change this.
  uint32_t row_block_idx = blockIdx.y;
  uint32_t col_block_idx = blockIdx.x;

  const data_type *A_block = d_A + row_block_idx * BM * K;
  const data_type *B_block = d_B + col_block_idx * BN;
  data_type *C_block = d_C + row_block_idx * BM * N + col_block_idx * BN;

  data_type __shared__ A_block_smem[BM * BK];
  data_type __shared__ B_block_smem[BK * BN];

  // TODO(shiwen): check this to enable coalesced memory access.
  // NOTE(shiwen): block_dim is 2d
  uint32_t row_thread_idx = threadIdx.y;
  uint32_t col_thread_idx = threadIdx.x;

  data_type thread_results[TM][TN] = {0.0};
  data_type TM_register[TM];
  data_type TN_register[TN];

  constexpr uint32_t num = K / BK;
  for (uint32_t i = 0; i < num; i++) {
    // load A block and B block from global memory to shared memory.
#pragma unroll
    // A block: BM * BK
    for (uint32_t a_row_idx = row_thread_idx; a_row_idx < BM;
         a_row_idx += BM / TM) {
      for (uint32_t a_col_idx = col_thread_idx; a_col_idx < BK;
           a_col_idx += BN / TN) {
        SMEM(A_block_smem, BK, a_row_idx, a_col_idx) =
            GMEM(A_block, K, a_row_idx, a_col_idx);
      }
    }

    // B block: BK * BN
    // NOTE(shiwen): b_col_idx = col_thread_idx
    for (uint32_t b_row_idx = row_thread_idx; b_row_idx < BK;
         b_row_idx += BM / TM) {
      for (uint32_t b_col_idx = col_thread_idx; b_col_idx < BN;
           b_col_idx += BN / TN) {
        SMEM(B_block_smem, BN, b_row_idx, b_col_idx) =
            GMEM(B_block, N, b_row_idx, b_col_idx);
      }
    }

    A_block += BK;
    B_block += BK * N;

    __syncthreads();

    for (uint32_t dot_idx = 0; dot_idx < BK; dot_idx++) {
      for (uint32_t i = 0; i < TM; i++) {
        TM_register[i] =
            SMEM(A_block_smem, BK, row_thread_idx * TM + i, dot_idx);
      }
      for (uint32_t i = 0; i < TN; i++) {
        TN_register[i] =
            SMEM(B_block_smem, BN, dot_idx, col_thread_idx * TN + i);
      }

      for (uint32_t a_res_idx = 0; a_res_idx < TM; a_res_idx++) {
        for (uint32_t b_res_idx = 0; b_res_idx < TN; b_res_idx++) {
          thread_results[a_res_idx][b_res_idx] +=
              TM_register[a_res_idx] * TN_register[b_res_idx];
        }
      }
    }

    __syncthreads();
  }

  for (uint32_t a_res_idx = 0; a_res_idx < TM; a_res_idx++) {
    for (uint32_t b_res_idx = 0; b_res_idx < TN; b_res_idx++) {
      GMEM(C_block, N, row_thread_idx * TM + a_res_idx,
           col_thread_idx * TN + b_res_idx) =
          thread_results[a_res_idx][b_res_idx];
    }
  }
}

template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t block_size>
float get_naiveSGEMM_launch_meantime(data_type *h_A, data_type *h_B,
                                     data_type *h_C, uint32_t repeat = 4) {
  // each block has 1024 threads, 32 x 32
  constexpr uint32_t block_dim_x = 16;
  constexpr uint32_t block_dim_y = 16;
  static_assert(block_size == block_dim_x * block_dim_y);

  dim3 block_dim = dim3(block_dim_x, block_dim_y);
  dim3 grid_dim = dim3((M + block_dim_x - 1) / block_dim_x,
                       (N + block_dim_y - 1) / block_dim_y);

  data_type *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(data_type));
  cudaMalloc(&d_B, K * N * sizeof(data_type));
  cudaMalloc(&d_C, M * N * sizeof(data_type));

  cudaMemcpy(d_A, h_A, M * K * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(data_type), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    naiveSGEMM<data_type, M, N, K><<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec_avg;
  float msec_total;
  cudaEventElapsedTime(&msec_total, start, end);
  msec_avg = msec_total / repeat;

  // for check error
  cudaMemcpy(h_C, d_C, M * N * sizeof(data_type), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return msec_avg;
}

template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t block_size>
float get_naiveSGEMM1_launch_meantime(data_type *h_A, data_type *h_B,
                                      data_type *h_C, uint32_t repeat = 4) {
  // each block has 1024 threads, 32 x 32
  constexpr uint32_t block_dim_x = 16;
  constexpr uint32_t block_dim_y = 16;
  static_assert(block_size == block_dim_x * block_dim_y);

  dim3 block_dim = dim3(block_dim_x, block_dim_y);
  // NOTE(shiwen): ❕❕❕ 需要让线程覆盖C的所有的点
  dim3 grid_dim = dim3((N + block_dim_x - 1) / block_dim_x,
                       (M + block_dim_y - 1) / block_dim_y);

  data_type *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(data_type));
  cudaMalloc(&d_B, K * N * sizeof(data_type));
  cudaMalloc(&d_C, M * N * sizeof(data_type));

  cudaMemcpy(d_A, h_A, M * K * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(data_type), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    naiveSGEMM1<data_type, M, N, K><<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec_avg;
  float msec_total;
  cudaEventElapsedTime(&msec_total, start, end);
  msec_avg = msec_total / repeat;

  // for check error
  cudaMemcpy(h_C, d_C, M * N * sizeof(data_type), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return msec_avg;
}

template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t block_size>
float get_naiveSGEMM2_launch_meantime(data_type *h_A, data_type *h_B,
                                      data_type *h_C, uint32_t repeat = 4) {

  dim3 block_dim = dim3(block_size);
  // NOTE(shiwen): ❕❕❕ 需要让线程覆盖C的所有的点
  dim3 grid_dim = dim3((M * N + block_size - 1) / block_size);

  data_type *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(data_type));
  cudaMalloc(&d_B, K * N * sizeof(data_type));
  cudaMalloc(&d_C, M * N * sizeof(data_type));

  cudaMemcpy(d_A, h_A, M * K * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(data_type), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    naiveSGEMM2<data_type, M, N, K><<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec_avg;
  float msec_total;
  cudaEventElapsedTime(&msec_total, start, end);
  msec_avg = msec_total / repeat;

  // for check error
  cudaMemcpy(h_C, d_C, M * N * sizeof(data_type), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return msec_avg;
}

template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t block_size>
float get_naiveSGEMM_SMEM_launch_meantime(data_type *h_A, data_type *h_B,
                                          data_type *h_C, uint32_t repeat = 4) {

  constexpr uint32_t block_len = 32;
  static_assert(M % block_len == 0, "must be multiple of block_len");
  static_assert(N % block_len == 0, "must be multiple of block_len");
  dim3 block_dim = dim3(block_len, block_len);
  // NOTE(shiwen): ❕❕❕ 需要让线程覆盖C的所有的点
  dim3 grid_dim = dim3(N / block_len, M / block_len);

  data_type *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(data_type));
  cudaMalloc(&d_B, K * N * sizeof(data_type));
  cudaMalloc(&d_C, M * N * sizeof(data_type));

  cudaMemcpy(d_A, h_A, M * K * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(data_type), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    sgemm_shared_mem_block<data_type, M, N, K, block_len, block_len * block_len>
        <<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec_avg;
  float msec_total;
  cudaEventElapsedTime(&msec_total, start, end);
  msec_avg = msec_total / repeat;

  // for check error
  cudaMemcpy(h_C, d_C, M * N * sizeof(data_type), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return msec_avg;
}

template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t block_size>
float get_naiveSGEMM_SMEM_normal_launch_meantime(data_type *h_A, data_type *h_B,
                                                 data_type *h_C,
                                                 uint32_t repeat = 4) {
  constexpr uint32_t BM = 16;
  constexpr uint32_t BN = 16;
  constexpr uint32_t BK = 64;
  static_assert(BM * BN == block_size);
  static_assert(M % BM == 0);
  static_assert(N % BN == 0);
  static_assert(K % BK == 0);

  dim3 block_dim = dim3(BN, BM);
  // NOTE(shiwen): ❕❕❕ 需要让线程覆盖C的所有的点
  dim3 grid_dim = dim3(N / BN, M / BM);

  data_type *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(data_type));
  cudaMalloc(&d_B, K * N * sizeof(data_type));
  cudaMalloc(&d_C, M * N * sizeof(data_type));

  cudaMemcpy(d_A, h_A, M * K * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(data_type), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    sgemm_shared_mem_block_normal<data_type, M, N, K, BM, BN, BK, BM * BN>
        <<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec_avg;
  float msec_total;
  cudaEventElapsedTime(&msec_total, start, end);
  msec_avg = msec_total / repeat;

  // for check error
  cudaMemcpy(h_C, d_C, M * N * sizeof(data_type), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return msec_avg;
}

template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t block_size>
float get_gemm_blocktiling_1D_launch_meantime(data_type *h_A, data_type *h_B,
                                              data_type *h_C,
                                              uint32_t repeat = 4) {
  constexpr uint32_t BM = 64;
  constexpr uint32_t BN = 16;
  constexpr uint32_t BK = 64;

  constexpr uint32_t TM = 4;

  static_assert(BM / TM * BN == block_size);

  static_assert(M % BM == 0);
  static_assert(N % BN == 0);
  static_assert(K % BK == 0);

  static_assert(BM > TM && BM % TM == 0);

  dim3 block_dim = dim3(BN, TM);
  // NOTE(shiwen): ❕❕❕ 需要让线程覆盖C的所有的点
  dim3 grid_dim = dim3(N / BN, M / BM);

  data_type *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(data_type));
  cudaMalloc(&d_B, K * N * sizeof(data_type));
  cudaMalloc(&d_C, M * N * sizeof(data_type));

  cudaMemcpy(d_A, h_A, M * K * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(data_type), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    sgemm_block_tiling_1D<data_type, M, N, K, BM, BN, BK, TM, BM * BN / TM>
        <<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec_avg;
  float msec_total;
  cudaEventElapsedTime(&msec_total, start, end);
  msec_avg = msec_total / repeat;

  // for check error
  cudaMemcpy(h_C, d_C, M * N * sizeof(data_type), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return msec_avg;
}

template <typename data_type = float, uint32_t M, uint32_t N, uint32_t K,
          uint32_t block_size>
float get_gemm_blocktiling_2D_launch_meantime(data_type *h_A, data_type *h_B,
                                              data_type *h_C,
                                              uint32_t repeat = 4) {
  constexpr uint32_t BM = 64;
  constexpr uint32_t BN = 64;
  constexpr uint32_t BK = 16;

  constexpr uint32_t TM = 4;
  constexpr uint32_t TN = 4;

  static_assert(BM / TM * BN / TN == block_size);

  static_assert(M % BM == 0);
  static_assert(N % BN == 0);
  static_assert(K % BK == 0);

  static_assert(BM > TM && BM % TM == 0);
  static_assert(BN > TN && BN % TN == 0);

  dim3 block_dim = dim3(BN / TN, BM / TM);
  dim3 grid_dim = dim3(N / BN, M / BM);

  data_type *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(data_type));
  cudaMalloc(&d_B, K * N * sizeof(data_type));
  cudaMalloc(&d_C, M * N * sizeof(data_type));

  cudaMemcpy(d_A, h_A, M * K * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(data_type), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    sgemm_block_tiling_2D<data_type, M, N, K, BM, BN, BK, TM, TN,
                          BM / TM * BN / TN>
        <<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec_avg;
  float msec_total;
  cudaEventElapsedTime(&msec_total, start, end);
  msec_avg = msec_total / repeat;

  // for check error
  cudaMemcpy(h_C, d_C, M * N * sizeof(data_type), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return msec_avg;
}

int main() {
  constexpr uint32_t M = 1 << 16;
  constexpr uint32_t N = 1 << 14;
  constexpr uint32_t K = 1 << 12;
  constexpr uint32_t block_size = 256;
  using test_dat_type = float;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<test_dat_type> dist(-1.0, 1.0);

  std::vector<test_dat_type> A(M * K);
  std::vector<test_dat_type> B(K * N);

  std::vector<test_dat_type> C_gpu_cublas(M * N, 0.0);
  std::vector<test_dat_type> C_gpu_gemm(M * N, 0.0);
  std::vector<test_dat_type> C_gpu_gemm1(M * N, 0.0);
  std::vector<test_dat_type> C_gpu_gemm2(M * N, 0.0);
  std::vector<test_dat_type> C_gpu_gemm_shared_memory(M * N, 0.0);
  std::vector<test_dat_type> C_gpu_smem_normal(M * N, 0.0);
  // std::vector<test_dat_type> C_gpu_blocktiling_1D(M * N, 0.0);
  std::vector<test_dat_type> C_gpu_blocktiling_2D(M * N, 0.0);
  // std::vector<test_dat_type> C_cpu(M * N, 0.0f);

  for (auto &val : A) {
    val = dist(gen);
  }
  for (auto &val : B) {
    val = dist(gen);
  }

  auto naive_gpu_time =
      get_naiveSGEMM_launch_meantime<test_dat_type, M, N, K, block_size>(
          A.data(), B.data(), C_gpu_gemm.data());

  auto naive_gpu_time1 =
      get_naiveSGEMM1_launch_meantime<test_dat_type, M, N, K, block_size>(
          A.data(), B.data(), C_gpu_gemm1.data());

  auto naive_gpu_time2 =
      get_naiveSGEMM2_launch_meantime<test_dat_type, M, N, K, block_size>(
          A.data(), B.data(), C_gpu_gemm2.data());

  auto naive_gpu_shared_memory_time =
      get_naiveSGEMM_SMEM_launch_meantime<test_dat_type, M, N, K, block_size>(
          A.data(), B.data(), C_gpu_gemm_shared_memory.data());

  auto naive_smem_normal_time =
      get_naiveSGEMM_SMEM_normal_launch_meantime<test_dat_type, M, N, K,
                                                 block_size>(
          A.data(), B.data(), C_gpu_smem_normal.data());
  // auto sgemm_block_tiling_1D_time =
  //     get_gemm_blocktiling_1D_launch_meantime<test_dat_type, M, N, K,
  //                                             block_size>(
  //         A.data(), B.data(), C_gpu_blocktiling_1D.data());

  auto sgemm_block_tiling_2D_time =
      get_gemm_blocktiling_2D_launch_meantime<test_dat_type, M, N, K,
                                              block_size>(
          A.data(), B.data(), C_gpu_blocktiling_2D.data());

  // auto cpu_time =
  //     get_cpu_gemm_time<float, M, N, K>(A.data(), B.data(), C_cpu.data());

  auto cublas_time = get_cublas_gemm_time<test_dat_type, M, N, K>(
      A.data(), B.data(), C_gpu_cublas.data());

  test_dat_type max_error = 0.0f;
  test_dat_type avg_error = 0.0f;
  compute_error(C_gpu_cublas, C_gpu_gemm, max_error, avg_error);

  test_dat_type max_error1 = 0.0f;
  test_dat_type avg_error1 = 0.0f;
  compute_error(C_gpu_cublas, C_gpu_gemm1, max_error1, avg_error1);

  test_dat_type max_error2 = 0.0f;
  test_dat_type avg_error2 = 0.0f;
  compute_error(C_gpu_cublas, C_gpu_gemm2, max_error2, avg_error2);

  test_dat_type max_error_shared_memory = 0.0f;
  test_dat_type avg_error_shared_memory = 0.0f;
  compute_error(C_gpu_cublas, C_gpu_gemm_shared_memory, max_error_shared_memory,
                avg_error_shared_memory);

  test_dat_type max_error_smem_normal = 0.0f;
  test_dat_type avg_error_smem_normal = 0.0f;
  compute_error(C_gpu_cublas, C_gpu_smem_normal, max_error_smem_normal,
                avg_error_smem_normal);

  // test_dat_type max_error_smem_blocktiling_1D = 0.0f;
  // test_dat_type avg_error_smem_blocktiling_1D = 0.0f;
  // compute_error(C_gpu_cublas, C_gpu_blocktiling_1D,
  //               max_error_smem_blocktiling_1D,
  //               avg_error_smem_blocktiling_1D);

  test_dat_type max_error_smem_blocktiling_2D = 0.0f;
  test_dat_type avg_error_smem_blocktiling_2D = 0.0f;
  compute_error(C_gpu_cublas, C_gpu_blocktiling_2D,
                max_error_smem_blocktiling_2D, avg_error_smem_blocktiling_2D);

  std::cout << "Naive GEMM Time: " << naive_gpu_time << " ms\n";
  std::cout << "Naive GEMM1 Time: " << naive_gpu_time1 << " ms\n";
  std::cout << "Naive GEMM2 Time: " << naive_gpu_time2 << " ms\n";
  std::cout << "Naive shared memory GEMM Time: " << naive_gpu_shared_memory_time
            << " ms\n";
  std::cout << "Naive shared memory (normal) GEMM Time: "
            << naive_smem_normal_time << " ms\n";
  // std::cout << "GEMM blocktiling1D: " << sgemm_block_tiling_1D_time << "
  // ms\n";
  std::cout << "GEMM blocktiling2D: " << sgemm_block_tiling_2D_time << " ms\n";
  std::cout << "CUBLAS GPU Time: " << cublas_time << " ms\n";
  // std::cout << "Speedup: " << cpu_time / gpu_time << "x\n";
  std::cout << "Max Error: " << max_error << "\n";
  std::cout << "Avg Error: " << avg_error << "\n";
  std::cout << "Max Error gemm1: " << max_error1 << "\n";
  std::cout << "Avg Error gemm1: " << avg_error1 << "\n";
  std::cout << "Max Error gemm2: " << max_error2 << "\n";
  std::cout << "Avg Error gemm2: " << avg_error2 << "\n";
  std::cout << "Max Error shared memory gemm: " << max_error_shared_memory
            << "\n";
  std::cout << "Avg Error shared memory gemm: " << avg_error_shared_memory
            << "\n";
  std::cout << "Max Error shared memory (normal) gemm: "
            << max_error_smem_normal << "\n";
  std::cout << "Avg Error shared memory (normal) gemm: "
            << avg_error_smem_normal << "\n";
  // std::cout << "Max Error 1D tiling: " << max_error_smem_blocktiling_1D <<
  // "\n"; std::cout << "Avg Error 1D tiling: " << avg_error_smem_blocktiling_1D
  // << "\n";
  std::cout << "Max Error 2D tiling: " << max_error_smem_blocktiling_2D << "\n";
  std::cout << "Avg Error 2D tiling: " << avg_error_smem_blocktiling_2D << "\n";
  std::cout << "print 2D block tiling: "
            << "\n";

  std::cout << "print C_gpu_cublas: "
            << "\n";
  print_vector(C_gpu_cublas, M * N - 100, M * N - 1);
  std::cout << "print C_gpu_gemm: "
            << "\n";
  print_vector(C_gpu_gemm, M * N - 100, M * N - 1);
  std::cout << "print C_gpu_gemm1: "
            << "\n";
  print_vector(C_gpu_gemm1, M * N - 100, M * N - 1);
  std::cout << "print C_gpu_gemm2: "
            << "\n";
  print_vector(C_gpu_gemm2, M * N - 100, M * N - 1);
  std::cout << "print C_gpu_shared_memory: "
            << "\n";
  print_vector(C_gpu_gemm_shared_memory, M * N - 100, M * N - 1);
  std::cout << "print C_gpu_smem_normal: "
            << "\n";
  print_vector(C_gpu_smem_normal, M * N - 100, M * N - 1);
  // std::cout << "print 1D block tiling: "
  //           << "\n";
  // print_vector(C_gpu_blocktiling_1D, M * N - 100, M * N - 1);
  std::cout << "print 2D block tiling: "
            << "\n";
  print_vector(C_gpu_blocktiling_2D, M * N - 100, M * N - 1);

  return 0;
}