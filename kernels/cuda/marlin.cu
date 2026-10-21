// Marlin W4A16 GEMM CUDA kernel for sm_75 to sm_90 architectures.
#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include <torch/headeronly/util/Exception.h>

namespace vllm {

class ScalarType {
 public:
  enum NanRepr : uint8_t {
    NAN_NONE = 0,
    NAN_IEEE_754 = 1,
    NAN_EXTD_RANGE_MAX_MIN = 2,

    NAN_REPR_ID_MAX
  };

  constexpr ScalarType(uint8_t exponent, uint8_t mantissa, bool signed_,
                       int32_t bias, bool finite_values_only = false,
                       NanRepr nan_repr = NAN_IEEE_754)
      : exponent(exponent),
        mantissa(mantissa),
        signed_(signed_),
        bias(bias),
        finite_values_only(finite_values_only),
        nan_repr(nan_repr) {};

  static constexpr ScalarType int_(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits - 1, true, bias);
  }

  static constexpr ScalarType uint(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits, false, bias);
  }

  static constexpr ScalarType float_IEEE754(uint8_t exponent,
                                            uint8_t mantissa) {
    STD_TORCH_CHECK(mantissa > 0 && exponent > 0);
    return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
  }

  static constexpr ScalarType float_(uint8_t exponent, uint8_t mantissa,
                                     bool finite_values_only,
                                     NanRepr nan_repr) {
    STD_TORCH_CHECK(nan_repr < NAN_REPR_ID_MAX, "Invalid NanRepr");
    STD_TORCH_CHECK(mantissa > 0 && exponent > 0);
    STD_TORCH_CHECK(
        nan_repr != NAN_IEEE_754,
        "use `float_IEEE754` constructor for floating point types that "
        "follow IEEE 754 conventions");
    return ScalarType(exponent, mantissa, true, 0, finite_values_only,
                      nan_repr);
  }

  uint8_t const exponent;
  uint8_t const mantissa;

  bool const signed_;

  int32_t const bias;

  bool const finite_values_only;
  NanRepr const nan_repr;

  using Id = int64_t;

 private:

  template <typename T_>
  static constexpr size_t member_id_field_width() {
    using T = std::decay_t<T_>;
    return std::is_same_v<T, bool> ? 1 : sizeof(T) * 8;
  }

  template <typename Fn, typename Init, typename Member, typename... Rest>
  static constexpr auto reduce_members_helper(Fn f, Init val, Member member,
                                              Rest... rest) {
    auto new_val = f(val, member);
    if constexpr (sizeof...(rest) > 0) {
      return reduce_members_helper(f, new_val, rest...);
    } else {
      return new_val;
    };
  }

  template <typename Fn, typename Init>
  constexpr auto reduce_members(Fn f, Init init) const {

    return reduce_members_helper(f, init, exponent, mantissa, signed_, bias,
                                 finite_values_only, nan_repr);
  };

  template <typename Fn, typename Init>
  static constexpr auto reduce_member_types(Fn f, Init init) {
    constexpr auto dummy_type = ScalarType(0, 0, false, 0, false, NAN_NONE);
    return dummy_type.reduce_members(f, init);
  };

  static constexpr auto id_size_bits() {
    return reduce_member_types(
        [](int acc, auto member) -> int {
          return acc + member_id_field_width<decltype(member)>();
        },
        0);
  }

 public:

  constexpr Id id() const {
    static_assert(id_size_bits() <= sizeof(Id) * 8,
                  "ScalarType id is too large to be stored");

    auto or_and_advance = [](std::pair<Id, uint32_t> result,
                             auto member) -> std::pair<Id, uint32_t> {
      auto [id, bit_offset] = result;
      auto constexpr bits = member_id_field_width<decltype(member)>();
      return {id | (int64_t(member) & ((uint64_t(1) << bits) - 1))
                       << bit_offset,
              bit_offset + bits};
    };
    return reduce_members(or_and_advance, std::pair<Id, uint32_t>{}).first;
  }

  static constexpr ScalarType from_id(Id id) {
    auto extract_and_advance = [id](auto result, auto member) {
      using T = decltype(member);
      auto [tuple, bit_offset] = result;
      auto constexpr bits = member_id_field_width<T>();
      auto extracted_val = static_cast<T>((int64_t(id) >> bit_offset) &
                                          ((uint64_t(1) << bits) - 1));
      auto new_tuple = std::tuple_cat(tuple, std::make_tuple(extracted_val));
      return std::pair<decltype(new_tuple), int>{new_tuple, bit_offset + bits};
    };

    auto [tuple_args, _] = reduce_member_types(extract_and_advance,
                                               std::pair<std::tuple<>, int>{});
    return std::apply([](auto... args) { return ScalarType(args...); },
                      tuple_args);
  }

  constexpr int64_t size_bits() const {
    return mantissa + exponent + is_signed();
  }
  constexpr bool is_signed() const { return signed_; }
  constexpr bool is_integer() const { return exponent == 0; }
  constexpr bool is_floating_point() const { return exponent > 0; }
  constexpr bool is_ieee_754() const {
    return is_floating_point() && finite_values_only == false &&
           nan_repr == NAN_IEEE_754;
  }
  constexpr bool has_nans() const {
    return is_floating_point() && nan_repr != NAN_NONE;
  }
  constexpr bool has_infs() const {
    return is_floating_point() && finite_values_only == false;
  }
  constexpr bool has_bias() const { return bias != 0; }

 private:
  double _floating_point_max() const {
    STD_TORCH_CHECK(mantissa <= 52 && exponent <= 11,
                    "Cannot represent max/min as a double for type ", str());

    uint64_t max_mantissa = (uint64_t(1) << mantissa) - 1;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN) {
      max_mantissa -= 1;
    }

    uint64_t max_exponent = (uint64_t(1) << exponent) - 2;
    if (nan_repr == NAN_EXTD_RANGE_MAX_MIN || nan_repr == NAN_NONE) {
      STD_TORCH_CHECK(exponent < 11,
                      "Cannot represent max/min as a double for type ", str());
      max_exponent += 1;
    }

    uint64_t exponent_bias = (uint64_t(1) << (exponent - 1)) - 1;
    uint64_t exponent_bias_double = (uint64_t(1) << 10) - 1;

    uint64_t max_exponent_double =
        max_exponent - exponent_bias + exponent_bias_double;

    uint64_t double_raw =
        (max_mantissa << (52 - mantissa)) | (max_exponent_double << 52);

    return *reinterpret_cast<double*>(&double_raw);
  }

  constexpr std::variant<int64_t, double> _raw_max() const {
    if (is_floating_point()) {
      return {_floating_point_max()};
    } else {
      STD_TORCH_CHECK(size_bits() < 64 || size_bits() == 64 && is_signed(),
                      "Cannot represent max as a int64_t");
      return {(int64_t(1) << mantissa) - 1};
    }
  }

  constexpr std::variant<int64_t, double> _raw_min() const {
    if (is_floating_point()) {
      STD_TORCH_CHECK(
          is_signed(),
          "We currently assume all floating point types are signed");
      constexpr uint64_t sign_bit_double = (uint64_t(1) << 63);

      double max = _floating_point_max();
      uint64_t max_raw = *reinterpret_cast<uint64_t*>(&max);
      uint64_t min_raw = max_raw | sign_bit_double;
      return {*reinterpret_cast<double*>(&min_raw)};
    } else {
      STD_TORCH_CHECK(!is_signed() || size_bits() <= 64,
                      "Cannot represent min as a int64_t");
      if (is_signed()) {

        return {INT64_MIN >> (64 - size_bits())};
      } else {
        return {int64_t(0)};
      }
    }
  }

 public:

  constexpr std::variant<int64_t, double> max() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_max());
  }

  constexpr std::variant<int64_t, double> min() const {
    return std::visit(
        [this](auto x) -> std::variant<int64_t, double> { return {x - bias}; },
        _raw_min());
  }

  std::string str() const {

    if (is_floating_point()) {
      auto ret = "float" + std::to_string(size_bits()) + "_e" +
                 std::to_string(exponent) + "m" + std::to_string(mantissa);
      if (!is_ieee_754()) {
        if (finite_values_only) {
          ret += "f";
        }
        if (nan_repr != NAN_NONE) {
          ret += "n";
        }
      }
      return ret;
    } else {
      auto ret = ((is_signed()) ? "int" : "uint") + std::to_string(size_bits());
      if (has_bias()) {
        ret += "b" + std::to_string(bias);
      }
      return ret;
    }
  }

  constexpr bool operator==(ScalarType const& other) const {
    return mantissa == other.mantissa && exponent == other.exponent &&
           bias == other.bias && signed_ == other.signed_ &&
           finite_values_only == other.finite_values_only &&
           nan_repr == other.nan_repr;
  }
};

using ScalarTypeId = ScalarType::Id;

static inline constexpr auto kS4 = ScalarType::int_(4);
static inline constexpr auto kU4 = ScalarType::uint(4);
static inline constexpr auto kU4B8 = ScalarType::uint(4, 8);
static inline constexpr auto kS8 = ScalarType::int_(8);
static inline constexpr auto kU8 = ScalarType::uint(8);
static inline constexpr auto kU8B128 = ScalarType::uint(8, 128);

static inline constexpr auto kFE2M1f =
    ScalarType::float_(2, 1, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE3M2f =
    ScalarType::float_(3, 2, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE2M3f =
    ScalarType::float_(2, 3, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE4M3fn =
    ScalarType::float_(4, 3, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M0fnu =
    ScalarType(8, 0, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE5M2 = ScalarType::float_IEEE754(5, 2);
static inline constexpr auto kFE8M7 = ScalarType::float_IEEE754(8, 7);
static inline constexpr auto kFE5M10 = ScalarType::float_IEEE754(5, 10);

static inline constexpr auto kInt4 = kS4;
static inline constexpr auto kUint4 = kU4;
static inline constexpr auto kUint4b8 = kU4B8;
static inline constexpr auto kInt8 = kS8;
static inline constexpr auto kUint8 = kU8;
static inline constexpr auto kUint8b128 = kU8B128;

static inline constexpr auto kFloat4_e2m1f = kFE2M1f;
static inline constexpr auto kFloat6_e3m2f = kFE3M2f;
static inline constexpr auto kFloat6_e2m3f = kFE2M3f;
static inline constexpr auto kFloat8_e4m3fn = kFE4M3fn;
static inline constexpr auto kFloat8_e5m2 = kFE5M2;
static inline constexpr auto kFloat16_e8m7 = kFE8M7;
static inline constexpr auto kFloat16_e5m10 = kFE5M10;

static inline constexpr auto kHalf = kFE5M10;
static inline constexpr auto kFloat16 = kHalf;
static inline constexpr auto kBFloat16 = kFE8M7;

static inline constexpr auto kFloat16Id = kFloat16.id();
};

#pragma once

#ifndef _marlin_cuh
  #define _marlin_cuh
  #include <cuda.h>
  #include <cuda_fp16.h>
  #include <cuda_runtime.h>
  #include <iostream>

  #ifndef MARLIN_NAMESPACE_NAME
    #define MARLIN_NAMESPACE_NAME marlin
  #endif

namespace MARLIN_NAMESPACE_NAME {

static constexpr int default_threads = 256;

static constexpr int pipe_stages =
    4;

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

__device__ inline void cp_async1_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    reinterpret_cast<int32_t*>(smem_ptr)[0] =
        reinterpret_cast<const int32_t*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    reinterpret_cast<int64_t*>(smem_ptr)[0] =
        reinterpret_cast<const int64_t*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  if (pred) {
    reinterpret_cast<int4*>(smem_ptr)[0] =
        reinterpret_cast<const int4*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  reinterpret_cast<int4*>(smem_ptr)[0] =
      reinterpret_cast<const int4*>(glob_ptr)[0];
}

__device__ inline void cp_async_fence() {}

template <int n>
__device__ inline void cp_async_wait() {}

  #else

__device__ inline void cp_async1_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  const int BYTES = 8;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

  #endif

}

#endif

#ifndef _data_types_cuh
#define _data_types_cuh
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

template <long scalar_type_id>
class MarlinScalarType {};

template <>
class MarlinScalarType<vllm::kFloat16.id()> {
 public:
  using scalar_t = half;
  using scalar_t2 = half2;
  using scalar_t4 = half2;
  using scalar_32bit_t = half2;

  using FragA = Vec<half2, 4>;
  using FragB = Vec<half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<half2, 1>;
  using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
  using FragZP = Vec<half2, 4>;

  static __device__ float inline num2float(const half x) {
    return __half2float(x);
  }

  static __device__ half2 inline num2num2(const half x) {
    return __half2half2(x);
  }

  static __device__ half2 inline nums2num2(const half x1, const half x2) {
    return __halves2half2(x1, x2);
  }

  static __host__ __device__ half inline float2num(const float x) {
    return __float2half(x);
  }

  static __host__ __device__ float2 inline num22float2(const half2 x) {
    return __half22float2(x);
  }
};

template <>
class MarlinScalarType<vllm::kBFloat16.id()> {
 public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;
  using scalar_t4 = nv_bfloat162;
  using scalar_32bit_t = nv_bfloat162;

  using FragA = Vec<nv_bfloat162, 4>;
  using FragB = Vec<nv_bfloat162, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<nv_bfloat162, 1>;
  using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
  using FragZP = Vec<nv_bfloat162, 4>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  static __device__ float inline num2float(const nv_bfloat16 x) {
    return __bfloat162float(x);
  }

  static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
  }

  static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1,
                                                  const nv_bfloat16 x2) {
    return __halves2bfloat162(x1, x2);
  }

  static __host__ __device__ nv_bfloat16 inline float2num(const float x) {
    return __float2bfloat16(x);
  }

  static __host__ __device__ float2 inline num22float2(const nv_bfloat162 x) {
    return __bfloat1622float2(x);
  }
#endif
};

template <>
class MarlinScalarType<vllm::kFE4M3fn.id()> {
 public:
  using scalar_t = __nv_fp8_e4m3;
  using scalar_t2 = __nv_fp8x2_e4m3;
  using scalar_t4 = __nv_fp8x4_e4m3;
  using scalar_32bit_t = __nv_fp8x4_e4m3;

  using FragA = Vec<__nv_fp8x4_e4m3, 4>;
  using FragB = Vec<__nv_fp8x4_e4m3, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<__nv_fp8x2_e4m3, 4>;

  static __host__ __device__
      float2 inline num22float2(const __nv_fp8x2_e4m3 x) {
    return (float2)x;
  }
};

template <>
class MarlinScalarType<vllm::kS8.id()> {
 public:
  using scalar_t = int8_t;
  using scalar_t2 = int16_t;
  using scalar_t4 = int32_t;
  using scalar_32bit_t = int32_t;

  using FragA = Vec<int32_t, 4>;
  using FragB = Vec<int32_t, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<int16_t, 4>;
};

template <typename scalar_t>
class MarlinScalarType2 {};

template <>
class MarlinScalarType2<half> : public MarlinScalarType<vllm::kFloat16.id()> {};

template <>
class MarlinScalarType2<nv_bfloat16>
    : public MarlinScalarType<vllm::kBFloat16.id()> {};

template <>
class MarlinScalarType2<__nv_fp8_e4m3>
    : public MarlinScalarType<vllm::kFE4M3fn.id()> {};

template <>
class MarlinScalarType2<int8_t> : public MarlinScalarType<vllm::kS8.id()> {};

}

#endif

namespace MARLIN_NAMESPACE_NAME {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <typename scalar_t2, vllm::ScalarTypeId w_type_id,
          bool skip_flop = false>
__device__ inline void dequant(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant<half2, vllm::kU4B8.id(), true>(int q,
                                                              half2* frag_b) {
  const int MASK = 0x000f000f;
  const int EX = 0x64006400;

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, vllm::kU4B8.id(), false>(int q,
                                                               half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<half2, vllm::kU4.id(), true>(int q,
                                                            half2* frag_b) {
  dequant<half2, vllm::kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, vllm::kU4.id(), false>(int q,
                                                             half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4B8.id(), true>(
    int q, nv_bfloat162* frag_b) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  frag_b[0] = *reinterpret_cast<nv_bfloat162*>(&lo);
  frag_b[1] = *reinterpret_cast<nv_bfloat162*>(&hi);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4B8.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kU4B8.id(), true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43084308;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4.id(), true>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kU4.id(), true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43004300;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

template <>
__device__ inline void dequant<half2, vllm::kU8B128.id(), true>(int q,
                                                                half2* frag_b) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, vllm::kU8B128.id(), false>(
    int q, half2* frag_b) {
  dequant<half2, vllm::kU8B128.id(), true>(q, frag_b);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  frag_b[0] = __hsub2(frag_b[0],
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(frag_b[1],
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<half2, vllm::kU8.id(), true>(int q,
                                                            half2* frag_b) {
  dequant<half2, vllm::kU8B128.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, vllm::kU8.id(), false>(int q,
                                                             half2* frag_b) {
  dequant<half2, vllm::kU8.id(), true>(q, frag_b);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;
  frag_b[0] = __hsub2(frag_b[0],
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(frag_b[1],
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU8B128.id(), false>(
    int q, nv_bfloat162* frag_b) {
  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388736.f;
  fp32_intermediates[1] -= 8388736.f;
  fp32_intermediates[2] -= 8388736.f;
  fp32_intermediates[3] -= 8388736.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU8.id(), false>(
    int q, nv_bfloat162* frag_b) {
  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388608.f;
  fp32_intermediates[1] -= 8388608.f;
  fp32_intermediates[2] -= 8388608.f;
  fp32_intermediates[3] -= 8388608.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);
}

template <>
__device__ inline void dequant<half2, vllm::kFE4M3fn.id(), true>(
    int q, half2* frag_b) {

  constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

template <>
__device__ inline void dequant<half2, vllm::kFE4M3fn.id(), false>(
    int q, half2* frag_b) {
  dequant<half2, vllm::kFE4M3fn.id(), true>(q, frag_b);

  constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;

  constexpr int BIAS_OFFSET =
      (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE4M3fn.id(), true>(
    int q, nv_bfloat162* frag_b) {

  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;

  constexpr int MASK = 0x7F007F00;

  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE4M3fn.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kFE4M3fn.id(), true>(q, frag_b);

  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;

  constexpr int BIAS_OFFSET =
      (1 << (BF16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));

  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg =
      __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<half2, vllm::kFE2M1f.id(), true>(int q,
                                                                half2* frag_b) {

  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;
  constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
}

template <>
__device__ inline void dequant<half2, vllm::kFE2M1f.id(), false>(
    int q, half2* frag_b) {
  dequant<half2, vllm::kFE2M1f.id(), true>(q, frag_b);

  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;

  constexpr int BIAS_OFFSET =
      (1 << (FP16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
    int q, nv_bfloat162* frag_b) {

  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kFE2M1f.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(q, frag_b);

  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;

  constexpr int BIAS_OFFSET =
      (1 << (BF16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));

  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg =
      __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<__nv_fp8x4_e4m3, vllm::kFE2M1f.id(), true>(
    int q, __nv_fp8x4_e4m3* frag_b) {

  constexpr int FP4_EXPONENT = 2, FP8_EXPONENT = 4;
  constexpr int RIGHT_SHIFT = FP8_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70707070;

  int Out1 = (q & 0x80808080) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80808080) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
  frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

template <>
__device__ inline void dequant<int32_t, vllm::kU4B8.id(), true>(
    int q, int32_t* frag_b) {
  constexpr int repeated_zp = 0x08080808;
  constexpr int MASK = 0x80808080;

  frag_b[0] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
  q >>= 4;
  frag_b[1] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
}

template <>
__device__ inline void dequant<__nv_fp8x4_e4m3, vllm::kU4B8.id(), true>(
    int q, __nv_fp8x4_e4m3* frag_b) {
  int s = q & 0x08080808;
  int Out1 = ((q & 0x07070707) | (s << 4)) + (s >> 3);
  q >>= 4;
  s = q & 0x08080808;
  int Out2 = ((q & 0x07070707) | (s << 4)) + (s >> 3);

  frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
  frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

template <typename scalar_t2, vllm::ScalarTypeId s_type_id>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2, vllm::kFE4M3fn.id()>(
    int q, half2* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  ;
  q <<= 8;
  int Out2 = (q & 0xFF00FF00) >> 1;

  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
};

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, vllm::kFE4M3fn.id()>(
    int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
    int q, nv_bfloat162* frag_b) {

  int Out1 = (q & 0xFF00FF00) >> 1;
  q <<= 7;
  int Out2 = q & 0x7F807F80;

  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
};

template <typename scalar_t2, vllm::ScalarTypeId w_type_id,
          bool skip_flop = false>
__device__ inline void sub_zp_and_dequant(int q, scalar_t2* frag_b, int zp);

template <>
__device__ inline void sub_zp_and_dequant<int32_t, vllm::kU4.id(), true>(
    int q, int32_t* frag_b, int zp) {

  int repeated_zp = 0x01010101 * zp;
  int MASK = 0x80808080;

  frag_b[0] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
  q >>= 4;
  frag_b[1] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
}

template <>
__device__ inline void sub_zp_and_dequant<__nv_fp8x4_e4m3, vllm::kU4.id(),
                                          true>(int q, __nv_fp8x4_e4m3* frag_b,
                                                int zp) {

  uint32_t u_q = *reinterpret_cast<uint32_t*>(&q);
  uint32_t u_zp = *reinterpret_cast<uint32_t*>(&zp);
  uint32_t u_zp1 = u_zp + 1;
  uint32_t repeated_zp = 0x01010101 * u_zp;

  uint32_t q0, s;
  q0 = (u_q & 0x0F0F0F0F) | 0x70707070;
  s = (q0 + repeated_zp) & 0x80808080;
  uint32_t Out1 = (q0 + (s >> 7) * u_zp1) & 0x0F0F0F0F | s;

  u_q >>= 4;
  q0 = (u_q & 0x0F0F0F0F) | 0x70707070;
  s = (q0 + repeated_zp) & 0x80808080;
  uint32_t Out2 = (q0 + (s >> 7) * u_zp1) & 0x0F0F0F0F | s;

  frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
  frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

#endif

}

namespace MARLIN_NAMESPACE_NAME {

template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::FragC& frag_c, int idx = 0) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  if constexpr (!std::is_same<scalar_t, half>::value || k_size != 16) {
    static_assert(!use_fp16_accum);
  }

  if constexpr (k_size == 16) {
    if constexpr (std::is_same<scalar_t, half>::value && !use_fp16_accum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(b[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[2]), "r"(a[3]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
#else
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
    } else if constexpr (std::is_same<scalar_t, half>::value &&
                         use_fp16_accum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[2]), "r"(a[3]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
#else
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "r"(c[0]), "r"(c[1]));
#endif
    } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[idx * 2]), "r"(a[idx * 2 + 1]), "r"(b[idx]), "f"(c[0]),
            "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(a[idx * 2]), "r"(a[idx * 2 + 1]), "r"(b[idx]), "r"(c[0]),
            "r"(c[1]), "r"(c[2]), "r"(c[3]));
    }
  } else if (k_size == 32) {
    if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[0]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(a[1]), "r"(b[0]), "r"(c[2]), "r"(c[3]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[2]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(a[3]), "r"(b[1]), "r"(c[2]), "r"(c[3]));
#else
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#endif
    }
  }
}

template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma_trans(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    const typename MarlinScalarType<type_id>::FragB& frag_b2,
    typename MarlinScalarType<type_id>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* b2 = reinterpret_cast<const uint32_t*>(&frag_b2);
  float* c = reinterpret_cast<float*>(&frag_c);
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  if constexpr (!std::is_same<scalar_t, half>::value || k_size != 16) {
    static_assert(!use_fp16_accum);
  }

  if constexpr (k_size == 16) {
    if constexpr (std::is_same<scalar_t, half>::value && !use_fp16_accum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[1]), "r"(b2[1]), "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
#else
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
    } else if constexpr (std::is_same<scalar_t, half>::value &&
                         use_fp16_accum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[1]), "r"(b2[1]), "r"(a[1]), "r"(c[0]), "r"(c[1]));
#else
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "r"(c[0]), "r"(c[1]));
#endif
    } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]),
            "r"(c[3]));
    }
  } else {
    if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(a[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(b2[1]), "r"(a[0]), "r"(c[2]), "r"(c[3]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(a[1]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(b2[1]), "r"(a[1]), "r"(c[2]), "r"(c[3]));
#else
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#endif
    }
  }
}

}

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace MARLIN_NAMESPACE_NAME {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

template <typename scalar_t,
          const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,

          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,

          const int stages,

          const int group_blocks,

          const bool is_zp_float
          >
__global__ void Marlin(
    const int4* __restrict__ A,
    const int4* __restrict__ B,
    int4* __restrict__ C,
    int4* __restrict__ C_tmp,
    const int4* __restrict__ scales_ptr,

    int prob_m,
    int prob_n,
    int prob_k,
    int* locks,
    bool use_fp32_reduce
) {}

}

#else

template <int count, vllm::ScalarTypeId type_id>
__device__ inline void ldsm(typename MarlinScalarType<type_id>::FragA& frag_a,
                            const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(a[0])
                 : "r"(smem));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

template <vllm::ScalarTypeId type_id>
__device__ inline void scale(typename MarlinScalarType<type_id>::FragB& frag_b,
                             typename MarlinScalarType<type_id>::FragS& frag_s,
                             int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <vllm::ScalarTypeId type_id>
__device__ inline void scale_and_sub(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t s,
    typename MarlinScalarType<type_id>::scalar_t zp) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s2 = MarlinScalarType<type_id>::num2num2(s);
  scalar_t2 zp2 = MarlinScalarType<type_id>::num2num2(zp);
  frag_b[0] = __hfma2(frag_b[0], s2, __hneg2(zp2));
  frag_b[1] = __hfma2(frag_b[1], s2, __hneg2(zp2));
}

template <vllm::ScalarTypeId type_id>
__device__ inline void sub_zp(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t2& frag_zp, int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 zp = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

template <vllm::ScalarTypeId type_id>
__device__ inline void scale_float(
    float* c, typename MarlinScalarType<type_id>::FragS& s) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], MarlinScalarType<type_id>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], MarlinScalarType<type_id>::num2float(s_ptr[1]));
}

__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do

      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;

    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

__device__ inline void wait_negative_and_add(int* lock) {
  if (threadIdx.x == 0) {
    int state = 0;
    do

      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

template <const vllm::ScalarTypeId a_type_id,
          const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,

          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,

          const int stages,

          const int group_blocks,

          const bool is_zp_float
          >
__global__ void Marlin(
    const int4* __restrict__ A0,
    const int4* __restrict__ B,
    int4* __restrict__ C0,
    int4* __restrict__ C_tmp,
    const int4* __restrict__ b_bias_ptr,

    const float* __restrict__ a_scales_ptr,

    const int4* __restrict__ scales_ptr,

    const float* __restrict__ global_scale_ptr,

    const int4* __restrict__ zp_ptr,
    int prob_m,
    int prob_n,
    int prob_k,
    int lda,
    int* locks,
    bool has_bias,
    bool use_atomic_add,
    bool use_fp32_reduce,
    int max_shared_mem) {

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890

  if constexpr (a_type_id == vllm::kFE4M3fn.id()) return;
  #endif

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750

  if constexpr (a_type_id != vllm::kFloat16.id() && a_type_id != vllm::kS8.id())
    return;
  #endif

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  constexpr auto num_bits = vllm::ScalarType::from_id(b_type_id).size_bits();

  constexpr bool use_fp16_accum =
      a_type_id == vllm::kFloat16.id() &&
      (!(b_type_id == vllm::kFE2M1f.id() && s_type_id == vllm::kFE4M3fn.id()) &&
       !(group_blocks == -1 && num_bits == 4));
  #else
  constexpr bool use_fp16_accum = false;
  #endif
  using Adtype = MarlinScalarType<a_type_id>;
  using Cdtype = MarlinScalarType<c_type_id>;
  const int4* A = A0;
  int4* C = C0;

  using scalar_t = typename MarlinScalarType<a_type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<a_type_id>::scalar_t2;
  using scalar_32bit_t = typename MarlinScalarType<a_type_id>::scalar_32bit_t;

  using c_scalar_t = typename MarlinScalarType<c_type_id>::scalar_t;
  using c_scalar_t2 = typename MarlinScalarType<c_type_id>::scalar_t2;

  using FragA = typename MarlinScalarType<a_type_id>::FragA;
  using FragB = typename MarlinScalarType<a_type_id>::FragB;
  using FragC = typename MarlinScalarType<a_type_id>::FragC;
  using FragS = typename MarlinScalarType<c_type_id>::FragS;
  using FragZP = typename MarlinScalarType<c_type_id>::FragZP;

  static constexpr auto a_type = vllm::ScalarType::from_id(a_type_id);
  static constexpr auto b_type = vllm::ScalarType::from_id(b_type_id);
  static constexpr auto c_type = vllm::ScalarType::from_id(c_type_id);
  static constexpr auto s_type = vllm::ScalarType::from_id(s_type_id);
  if constexpr (b_type == vllm::kFE2M1f) {
    static_assert(s_type == vllm::kFE4M3fn && group_blocks == 1 ||
                  s_type == vllm::kFE8M0fnu && group_blocks == 2);
  } else if constexpr (s_type == vllm::kFE8M0fnu) {

    static_assert(b_type == vllm::kFE4M3fn && group_blocks == 2);
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    static_assert(s_type == vllm::kBFloat16);
  } else if constexpr (std::is_same<scalar_t, half>::value) {
    static_assert(s_type == vllm::kFloat16);
  }

  constexpr bool is_a_8bit = a_type.size_bits() == 8;
  constexpr bool is_8bit_scale = s_type.size_bits() == 8;
  if constexpr (!is_a_8bit) {
    static_assert(std::is_same<scalar_t, c_scalar_t>::value);
  }
  constexpr bool has_zp = b_type == vllm::kU4 || b_type == vllm::kU8;
  constexpr bool is_int_type = b_type == vllm::kU4 || b_type == vllm::kU8 ||
                               b_type == vllm::kS4 || b_type == vllm::kS8 ||
                               b_type == vllm::kU4B8 || b_type == vllm::kU8B128;

  constexpr bool dequant_skip_flop =
      is_a_8bit || (b_type == vllm::kFE4M3fn && !(s_type == vllm::kFE8M0fnu)) ||
      b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn ||
      has_zp && !is_zp_float && !std::is_same<scalar_t, nv_bfloat16>::value ||
      has_zp && !is_zp_float && !(b_type == vllm::kU8);

  float global_scale_f32 = 1.0f;

  if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
    global_scale_f32 = global_scale_ptr[0];
  }

  static_assert(group_blocks != 0,
                "runtime activation ordering is not supported");
  constexpr int m_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);

  extern __shared__ int4 sh[];
  float* sh_a_s = reinterpret_cast<float*>(sh);
  int4* sh_new = sh + (is_a_8bit ? (4 * thread_m_blocks) : 0);
  constexpr int pack_factor = 32 / b_type.size_bits();
  static_assert(thread_m_blocks == 1 || !m_block_size_8);

  int parallel = 1;
  if (prob_m > m_block_size) {
    parallel = prob_m / m_block_size;
    prob_m = m_block_size;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;

  int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  int part1_mn_iters = 0;
  bool in_part2 = false;

  if (global_mn_tiles > gridDim.x) {
    part2_mn_tiles = global_mn_tiles % gridDim.x;
    if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
    part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / gridDim.x;
  }

  int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);

  if constexpr (group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {

      iters = (group_blocks / thread_k_blocks) *
              div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }

  int slice_row = 0;
  int slice_col_par = blockIdx.x;
  int slice_col;
  int slice_iters =
      k_tiles;

  int slice_count = 1;

  int slice_idx = 0;

  int par_id = 0;
  int locks_off = 0;

  if (part2_mn_tiles >= gridDim.x) {

    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

  bool first_init = true;
  auto init_part2_slice = [&]() {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= part2_mn_tiles) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (part2_mn_tiles >= gridDim.x) {
      if (slice_count > 1 && slice_idx == slice_count - 1) {
        locks_off++;
      }
    } else {
      locks_off++;
    }

    if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
      constexpr int threads_per_m = 16 * thread_n_blocks / 8;
      int m_per_thread =
          div_ceil(thread_m_blocks * 16, threads / threads_per_m);
      if (m_block_size_8) m_per_thread = div_ceil(8, threads / threads_per_m);
      for (int i = 0; i < m_per_thread; i++) {
        int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
        if (row < prob_m) {
          int col = slice_col * 16 * thread_n_blocks / 8 +
                    threadIdx.x % threads_per_m;
          C[row * prob_n / 8 + col] = {0, 0, 0, 0};
        }
      }

      __syncthreads();
      if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
    }

    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * lda / (is_a_8bit ? 16 : 8);
      C += 16 * thread_m_blocks * prob_n / 8;
      slice_col = 0;
      par_id++;
    }
    if (is_a_8bit && (first_init || slice_col == 0)) {
      __syncthreads();
      int a_s_gl_rd = par_id * 16 * thread_m_blocks + threadIdx.x;
      cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[a_s_gl_rd],
                        threadIdx.x < prob_m);
    }
  };

  auto init_part1_slice = [&]() {
    if (part1_mn_iters) {
      part1_mn_iters--;
      par_id = slice_col_par / n_tiles;
      slice_col = slice_col_par % n_tiles;
      slice_iters = k_tiles;
      A = A0 + 16 * thread_m_blocks / (is_a_8bit ? 16 : 8) * par_id * lda;
      C = C0 + 16 * thread_m_blocks / 8 * par_id * prob_n;
      if (is_a_8bit) {
        __syncthreads();
        int a_s_gl_rd = par_id * 16 * thread_m_blocks + threadIdx.x;
        cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[a_s_gl_rd],
                          threadIdx.x < prob_m);
      }
    }
  };

  auto init_slice = [&]() {
    if (!in_part2 && !part1_mn_iters) {
      in_part2 = true;
      slice_col_par = (iters * blockIdx.x) / k_tiles;
      slice_row = (iters * blockIdx.x) % k_tiles;
      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;
      A = A0 + 16 * thread_m_blocks / (is_a_8bit ? 16 : 8) * par_id * lda;
      C = C0 + 16 * thread_m_blocks / 8 * par_id * prob_n;
    }
    if (!in_part2) {
      init_part1_slice();
    } else {
      init_part2_slice();
      first_init = false;
    }
  };

  init_slice();

  int a_gl_stride = lda / (is_a_8bit ? 16 : 8);

  constexpr int a_sh_stride = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);

  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);

  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);

  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);

  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;

  constexpr int a_sh_stage = a_sh_stride * m_block_size;

  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  int b_gl_stride = 16 * prob_n / (pack_factor * (is_a_8bit ? 2 : 4));
  constexpr int b_sh_stride =
      ((thread_n_blocks * 16) * 16 / pack_factor) / (is_a_8bit ? 2 : 4);
  constexpr int b_thread_vecs = b_type.size_bits() == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage =
      b_sh_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  int s_gl_stride = prob_n / (is_8bit_scale ? 16 : 8);
  constexpr int s_sh_stride = 16 * thread_n_blocks / (is_8bit_scale ? 16 : 8);
  constexpr int s_tb_groups =
      group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks
          : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  constexpr int tb_n_warps = thread_n_blocks / (is_a_8bit ? 2 : 4);

  int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = is_zp_float
                                   ? 16 * thread_n_blocks / 8
                                   : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;

  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);

  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % (16 / (m_block_size_8 ? 2 : 1))) +
      (threadIdx.x % 32) / (16 / (m_block_size_8 ? 2 : 1));
  a_sh_rd += 2 * ((threadIdx.x / 32) / tb_n_warps) * b_sh_wr_iters;

  int b_gl_rd;
  if (threads <= b_sh_stride) {
    b_gl_rd = threadIdx.x;
  } else {
    b_gl_rd =
        b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  }

  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_rd = threadIdx.x * b_thread_vecs;
  b_sh_rd += b_sh_rd / b_sh_stride * (b_sh_stride * (b_sh_wr_iters - 1));

  int s_gl_rd;
  if constexpr (group_blocks == -1) {
    s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
  } else if constexpr (group_blocks >= thread_k_blocks) {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + threadIdx.x;
  } else {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                             threadIdx.x / s_sh_stride) +
              s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
  }
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stage;

  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else if constexpr (group_blocks >= thread_k_blocks) {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                 zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                 threadIdx.x / zp_sh_stride) +
                 zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
    }
  }
  auto zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = zp_sh_stage > 0 && threadIdx.x < zp_sh_stage;

  int s_sh_rd;
  if constexpr (is_a_8bit) {
    s_sh_rd = 4 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 4);
  } else if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
  else if constexpr (group_blocks == -1 &&
                     (m_block_size_8 || (has_zp && !dequant_skip_flop)))
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) % 4;

  int bias_sh_rd;
  if constexpr (m_block_size_8) {
    bias_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  } else {
    bias_sh_rd = (is_a_8bit ? 4 : 8) * ((threadIdx.x / 32) % tb_n_warps) +
                 (threadIdx.x % 32) % 4;
  }

  int bias_sh_wr = threadIdx.x;
  int bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;

  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    if constexpr (is_zp_float) {
      if constexpr (group_blocks != -1) {
        zp_sh_rd =
            8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
      }
    } else if (is_a_8bit) {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % tb_n_warps / 2) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    } else {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % tb_n_warps) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    }
  }

  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };

  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(2 * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  constexpr int sh_b_size = stages * b_sh_stage;
  int4* sh_b = sh_new;
  int4* sh_red = sh_new;
  constexpr int sh_size_b_red_min =
      (sh_red_size < sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_size_b_red_max =
      (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_bias_size = (thread_n_blocks * 16 / 8);
  constexpr int sh_b_red_bias_size =
      sh_size_b_red_max > (sh_size_b_red_min + sh_bias_size)
          ? sh_size_b_red_max
          : (sh_size_b_red_min + sh_bias_size);

  int4* sh_bias = sh_new + sh_size_b_red_min;
  int4* sh_zp = sh_new + sh_b_red_bias_size;
  constexpr int sh_s_size = stages * s_sh_stage;
  int4* sh_s = sh_zp + (stages * zp_sh_stage);
  int4* sh_a = sh_s + sh_s_size;

  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  FragC frag_c_tmp[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  FragS frag_s[2][4];
  FragS frag_bias[2][4];
  int frag_qzp[2][num_ints_per_thread];
  FragZP frag_zp;
  FragZP frag_zpf[2];

  if constexpr (is_a_8bit) {
  #pragma unroll
    for (int j = 0; j < 2; j++) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][0][g] = 0.0f;
        }

  #pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][1][g] = 0.0f;
        }
      }
    }
  }

  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {
        constexpr int count = div_ceil(b_sh_stride, threads);
        int b_gl_idx =
            b_gl_rd + (i % count) * threads +
            b_gl_stride * (i / count) * div_ceil(threads, b_sh_stride);

        cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
      }

      b_gl_rd += b_gl_rd_delta_o;

      if constexpr (group_blocks != -1) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;

        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          s_gl_rd += s_gl_rd_delta * s_tb_groups;
        }
      }

      if constexpr (has_zp && group_blocks != -1) {
        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if (zp_sh_wr_pred) {
            cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
          }
          zp_gl_rd += zp_gl_rd_delta * zp_tb_groups;
        }
      }
    }

    cp_async_fence();
  };

  auto fetch_col_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  auto fetch_col_scale_to_shared = [&]() {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
  };

  auto wait_for_stage = [&]() {

    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<m_block_size_8 ? 2 : 4, a_type_id>(
          frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;
    using IT1 = typename std::conditional_t<is_a_8bit, int2, int4>;
    using IT0 = typename std::conditional_t<is_a_8bit, int, int2>;
    constexpr int group_blocks2 = div_ceil(group_blocks, is_a_8bit ? 2 : 1);

    if constexpr (group_blocks == -1) {

      if (k == 0 && full_pipe == 0 && dequant_skip_flop) {
        reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd];
        reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
      }
    } else if constexpr (group_blocks >= thread_k_blocks) {
      constexpr int g = group_blocks / thread_k_blocks;
      if (pipe % g == 0) {
        if (k % b_sh_wr_iters == 0) {
          int4* sh_s_stage = sh_s + s_sh_stage * (g * (pipe / g));
          reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
        } else {
          reinterpret_cast<int4*>(&frag_s[1])[0] =
              reinterpret_cast<int4*>(&frag_s[0])[0];
        }
      }
    } else if (group_blocks2 < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
      auto warp_id = threadIdx.x / 32;
      int warp_row = warp_id / tb_n_warps;

      int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
      int cur_group_id = k_blocks / group_blocks2;

      int4* sh_s_stage = sh_s + s_sh_stage * pipe;

      if constexpr (!is_8bit_scale) {
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
            sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
      } else {
        reinterpret_cast<int2*>(&frag_s[k % 2])[0] = reinterpret_cast<int2*>(
            sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
      }
    } else if (group_blocks >= b_sh_wr_iters) {
      if constexpr (!is_8bit_scale) {
        reinterpret_cast<int4*>(&frag_s[1])[0] =
            reinterpret_cast<int4*>(&frag_s[0])[0];
      } else {
        reinterpret_cast<int2*>(&frag_s[1])[0] =
            reinterpret_cast<int2*>(&frag_s[0])[0];
      }
    }
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    static_assert(!has_zp || group_blocks != 0);

    if constexpr (has_zp && !is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {

        if (k == 0 && full_pipe == 0 || is_a_8bit) {
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
          }
        }
      } else if constexpr (group_blocks >= thread_k_blocks) {
        constexpr int g = group_blocks / thread_k_blocks;
        if (pipe % g == 0 && k % b_sh_wr_iters == 0 || is_a_8bit) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] =
                (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
          }
        }
      } else {
        auto warp_id = threadIdx.x / 32;

        int warp_row = warp_id / tb_n_warps;

        int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
        int cur_group_id = k_blocks / div_ceil(group_blocks, is_a_8bit ? 2 : 1);

        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        sh_zp_stage += cur_group_id * zp_sh_stride;

  #pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    }

    else if constexpr (has_zp && is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          constexpr int g = group_blocks / thread_k_blocks;
          if (pipe % g == 0 && k % b_sh_wr_iters == 0) {
            int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
            reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
                sh_zp_stage[zp_sh_rd];
          }
        } else if (group_blocks < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
          auto warp_id = threadIdx.x / 32;

          int warp_row = warp_id / tb_n_warps;
          int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
          int cur_group_id = k_blocks / group_blocks;

          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
              sh_zp_stage[zp_sh_rd + cur_group_id * zp_sh_stride];
        }
      }
    }
  };

  auto dequant_data = [&](int q, scalar_32bit_t* frag_b_ptr, int zp = 0) {
    if constexpr (a_type.size_bits() != b_type.size_bits()) {
      if constexpr (is_a_8bit && has_zp) {
        sub_zp_and_dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(
            q, frag_b_ptr, zp);
      } else {
        dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(q, frag_b_ptr);
      }
    }
  };

  bool is_first_matmul_in_slice = true;
  auto matmul = [&](int k, int pipe) {
    if (is_a_8bit) return;
    int k2 = k % 2;
    constexpr int g =
        group_blocks > 0 ? div_ceil(group_blocks, thread_k_blocks) : 1;
    const bool is_new_zp =
        (group_blocks > 0 && (group_blocks < b_sh_wr_iters || k == 0)) &&
            (pipe % g == 0) ||
        (group_blocks == -1 && is_first_matmul_in_slice);
    if constexpr (has_zp && !is_zp_float) {
      if (is_new_zp) {
        if constexpr (group_blocks == -1) is_first_matmul_in_slice = false;
        int zp_quant_0, zp_quant_1;

        if constexpr (b_type.size_bits() == 4) {
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = zp_quant_0 >> 8;
        } else {
          static_assert(b_type.size_bits() == 8);
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = frag_qzp[k2][1];
        }

        dequant_data(zp_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_zp));
        dequant_data(zp_quant_1,
                     reinterpret_cast<scalar_32bit_t*>(&frag_zp) + 2);
      }
    }
    if constexpr (!dequant_skip_flop && has_zp && is_zp_float) {
      if (is_new_zp) {
        reinterpret_cast<int4*>(&frag_zp)[0] =
            reinterpret_cast<int4*>(&frag_zpf[k2])[0];
      }
    }

    if constexpr (s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu) {
      int s_quant_0 = reinterpret_cast<int*>(frag_s[k2])[0];
      int s_quant_1 = reinterpret_cast<int*>(frag_s[k2])[1];

      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_0, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]));
      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_1, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]) + 2);
    }

  #pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      int b_quant_0, b_quant_1;

      if constexpr (b_type_id == vllm::kFE2M1f.id()) {
        b_quant_1 = frag_b_quant[k2][0][j];
        b_quant_0 = b_quant_1 << 8;
      } else if constexpr (b_type.size_bits() == 4) {
        b_quant_0 = frag_b_quant[k2][0][j];
        b_quant_1 = b_quant_0 >> 8;
      } else {
        static_assert(b_type.size_bits() == 8);
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k2]);
        b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
      }

      dequant_data(b_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_b0));
      dequant_data(b_quant_1, reinterpret_cast<scalar_32bit_t*>(&frag_b1));

      if constexpr (dequant_skip_flop && has_zp && !is_zp_float && !is_a_8bit) {
        sub_zp<a_type_id>(frag_b0, frag_zp[j], 0);
        sub_zp<a_type_id>(frag_b1, frag_zp[j], 1);
      }

      if constexpr (!dequant_skip_flop && has_zp && !is_zp_float &&
                    group_blocks == -1 && !is_a_8bit) {
        int idx = (threadIdx.x / 4) % 2;
        scalar_t2 s2 = Adtype::nums2num2(
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 0])[idx],
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 1])[idx]);
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], s2);
        scale_and_sub<a_type_id>(frag_b0, s2.x, frag_zp[j].x);
        scale_and_sub<a_type_id>(frag_b1, s2.y, frag_zp[j].y);
      } else if constexpr (!dequant_skip_flop && has_zp && group_blocks != -1 &&
                           !is_a_8bit) {
        if (is_new_zp)
          frag_zp[j] = __hmul2(frag_zp[j],
                               *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        scale_and_sub<a_type_id>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
        scale_and_sub<a_type_id>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);
      } else if constexpr (group_blocks != -1 && !is_a_8bit) {
        scale<a_type_id>(frag_b0, frag_s[k2][j], 0);
        scale<a_type_id>(frag_b1, frag_s[k2][j], 1);
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          mma_trans<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0, frag_b1,
                                               frag_c[i][j][0]);
        } else {
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0,
                                         frag_c[i][j][0]);
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b1,
                                         frag_c[i][j][1]);
        }
      }
    }
  };

  auto matmul_a8 = [&](int k) {
    int k2 = k % 2;
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      FragB frag_b[2];

      if (is_a_8bit && b_type.size_bits() == 4 && !has_zp) {
        dequant_data(frag_b_quant[k2][0][j * 2],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b));
        dequant_data(frag_b_quant[k2][0][j * 2 + 1],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2);
      } else if (is_a_8bit && b_type.size_bits() == 4 && has_zp) {
        int off = (threadIdx.x / 32) % 2 * 2 + j;
        int zp = (frag_qzp[k2][0] >> (off * 8)) & 0xF;
        dequant_data(frag_b_quant[k2][0][j * 2],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b), zp);
        zp = (frag_qzp[k2][0] >> (off * 8 + 4)) & 0xF;
        dequant_data(frag_b_quant[k2][0][j * 2 + 1],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2, zp);
      } else {
        reinterpret_cast<int2*>(&frag_b)[0] =
            reinterpret_cast<int2*>(&frag_b_quant[k2][j])[0];
        reinterpret_cast<int2*>(&frag_b)[1] =
            reinterpret_cast<int2*>(&frag_b_quant[k2][j])[1];
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma<a_type_id, false, 32>(
            frag_a[k2][i], frag_b[0],
            (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][0]);
        mma<a_type_id, false, 32>(
            frag_a[k2][i], frag_b[1],
            (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][1]);
      }

      if constexpr (group_blocks != -1) {
        if (group_blocks == 2 || k == 1) {
          if constexpr (a_type == vllm::kS8) {
            int2 s_vals[2];
            s_vals[0] = {
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[0],
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[1]};
            s_vals[1] = {
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[0],
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[1]};

  #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
              for (int g = 0; g < 4; g++) {
                int scale = reinterpret_cast<int*>(&s_vals[0])[g % 2];
                *reinterpret_cast<int32_t*>(&frag_c[i][j][0][g]) +=
                    *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][0][g]) *
                    scale;
                frag_c_tmp[i][j][0][g] = 0.0f;
              }

  #pragma unroll
              for (int g = 0; g < 4; g++) {
                int scale = reinterpret_cast<int*>(&s_vals[1])[g % 2];
                *reinterpret_cast<int32_t*>(&frag_c[i][j][1][g]) +=
                    *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][1][g]) *
                    scale;
                frag_c_tmp[i][j][1][g] = 0.0f;
              }
            }
          } else {
            float2 s_vals[2];
            if constexpr (s_type_id != vllm::kFE8M0fnu.id()) {
              static_assert(a_type.size_bits() == 16 ||
                            s_type.size_bits() == 16);
              s_vals[0] = Cdtype::num22float2(frag_s[k2][j * 2][0]);
              s_vals[1] = Cdtype::num22float2(frag_s[k2][j * 2 + 1][0]);
            } else {
              int32_t* s_vals_int = reinterpret_cast<int32_t*>(&s_vals[0]);
              int32_t s_vals_e8m0 =
                  *reinterpret_cast<int32_t*>(&frag_s[k2][j][0]);

              s_vals_int[0] = (s_vals_e8m0 & 0xFF) << 23;
              s_vals_int[1] = (s_vals_e8m0 & 0xFF00) << 15;
              s_vals_int[2] = (s_vals_e8m0 & 0xFF0000) << 7;
              s_vals_int[3] = (s_vals_e8m0 & 0xFF000000) >> 1;
            }

  #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
              for (int g = 0; g < 4; g++) {
                float scale = reinterpret_cast<float*>(&s_vals[0])[g % 2];
                frag_c[i][j][0][g] += frag_c_tmp[i][j][0][g] * scale;
                frag_c_tmp[i][j][0][g] = 0.0f;
              }

  #pragma unroll
              for (int g = 0; g < 4; g++) {
                float scale = reinterpret_cast<float*>(&s_vals[1])[g % 2];
                frag_c[i][j][1][g] += frag_c_tmp[i][j][1][g] * scale;
                frag_c_tmp[i][j][1][g] = 0.0f;
              }
            }
          }
        }
      }
    }
  };

  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride =
          b_sh_stride_threads * (is_a_8bit ? 2 : 4) * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);

  #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
  #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
  #pragma unroll
            for (int j = 0; j < (is_a_8bit ? 2 : 4) * 2;
                 j += (m_block_size_8 ? 2 : 1)) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(
                    &sh_red[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
  #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(
                      frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh_red[red_sh_wr] = reinterpret_cast<int4*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
  #pragma unroll
          for (int i = 0; i < (is_a_8bit ? 2 : 4) * 2;
               i += (m_block_size_8 ? 2 : 1)) {
            float* c_rd =
                reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
  #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(
                  frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {

    constexpr int active_threads = 32 * tb_n_warps;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride * (is_a_8bit ? 2 : 1);
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr;
      if constexpr (m_block_size_8) {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 4) * 2) +
                  4 * (threadIdx.x / 32) + (threadIdx.x % 32) / 8;
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
      } else {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) * (is_a_8bit ? 2 : 1) +
                  4 * (threadIdx.x / 32) + threadIdx.x % 4;
        c_gl_wr += (2 * thread_n_blocks) * slice_col * (is_a_8bit ? 2 : 1);
      }
      constexpr int c_sh_wr_delta = active_threads;
      auto c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {

  #pragma unroll
        for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
          if constexpr (m_block_size_8) {
            cp_async4_pred(&sh_red[c_sh_wr + c_sh_wr_delta * i],
                           &C[c_gl_wr + i * c_gl_stride +
                              (threadIdx.x % 8) / 4 * c_gl_wr_delta_i],
                           (threadIdx.x % 4) * 2 + i < prob_m);
          } else if constexpr (is_a_8bit) {
            int2* sh_red_int2 = reinterpret_cast<int2*>(sh_red);
            int2* c_int2 = reinterpret_cast<int2*>(C);
            cp_async2_ca_pred(
                &sh_red_int2[c_sh_wr + c_sh_wr_delta * i],
                &c_int2[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                        c_gl_wr_delta_i * (i % 2)],
                i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
          } else {
            cp_async4_pred(
                &sh_red[c_sh_wr + c_sh_wr_delta * i],
                &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                   c_gl_wr_delta_i * (i % 2)],
                i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
          }
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

  #pragma unroll
      for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
        bool mask = (!m_block_size_8) && (i < (thread_m_blocks - 1) * 4 ||
                                          8 * (i / 2) + row < prob_m) ||
                    (m_block_size_8) && ((threadIdx.x % 4) * 2 + i < prob_m);
        if (mask) {
          if (!first) {
            c_scalar_t* c_red_f16;
            if constexpr (is_a_8bit) {
              int2 tmp =
                  reinterpret_cast<int2*>(sh_red)[c_sh_wr + i * c_sh_wr_delta];
              c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
            } else {
              int4 tmp = sh_red[c_sh_wr + i * c_sh_wr_delta];
              c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
            }
  #pragma unroll
            for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              reinterpret_cast<float*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j +
                           (i % 4) + delta] += Cdtype::num2float(c_red_f16[j]);
            }
          }
          if (!last) {
            c_scalar_t c_f16[is_a_8bit ? 4 : 8];
  #pragma unroll
            for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              c_f16[j] = Cdtype::float2num(reinterpret_cast<float*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j +
                           (i % 4) + delta]);
            }
            if constexpr (m_block_size_8) {
              C[c_gl_wr + i * c_gl_stride +
                (threadIdx.x % 8) / 4 * c_gl_wr_delta_i] =
                  *reinterpret_cast<int4*>(c_f16);
            } else if constexpr (is_a_8bit) {
              int2* c_int2 = reinterpret_cast<int2*>(C);
              c_int2[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                     c_gl_wr_delta_i * (i % 2)] =
                  *reinterpret_cast<int2*>(c_f16);
            } else {
              C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                c_gl_wr_delta_i * (i % 2)] = *reinterpret_cast<int4*>(c_f16);
            }
          }
        }
      }
    }
  };

  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;

    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;

    constexpr int active_threads = 32 * tb_n_warps;
    bool is_th_active = threadIdx.x < active_threads;

    constexpr int num_floats = thread_m_blocks * (is_a_8bit ? 2 : 4) * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;

    int c_cur_offset = locks_off * c_size;

    if (!is_th_active) {
      return;
    }

    if (!first) {
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        sh_red[threadIdx.x] =
            C_tmp[c_cur_offset + active_threads * k + threadIdx.x];

        float* sh_c_ptr = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) {
          frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
        }
      }
    }

    if (!last) {
      int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
      }
    }
  };

  auto write_result = [&](bool last) {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr;
    if constexpr (m_block_size_8) {
      c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) +
                (threadIdx.x % 32) / 4;
      c_sh_wr += 64 * (threadIdx.x / 32);
    } else {
      c_sh_wr =
          (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
      c_sh_wr += (is_a_8bit ? 16 : 32) * (threadIdx.x / 32);
    }

    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    auto write = [&](int idx, float c0, float c1, FragS& s, FragS& b_bias) {
      if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
        c0 *= global_scale_f32;
        c1 *= global_scale_f32;
      }
      c_scalar_t2 res =
          Cdtype::nums2num2(Cdtype::float2num(c0), Cdtype::float2num(c1));

      if constexpr (group_blocks == -1 && !is_a_8bit &&
                    b_type.size_bits() == 4 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        c_scalar_t2 tmp_scale = s[0];
        if constexpr (m_block_size_8) {
          tmp_scale = Cdtype::num2num2(
              reinterpret_cast<scalar_t*>(&s[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hmul2(res, tmp_scale);
      }
      if (has_bias && last) {
        c_scalar_t2 tmp_bias = b_bias[0];
        if constexpr (m_block_size_8) {
          tmp_bias = Cdtype::num2num2(
              reinterpret_cast<scalar_t*>(&b_bias[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hadd2(res, tmp_bias);
      }

      if constexpr (m_block_size_8) {
        ((c_scalar_t*)sh_red)[idx] = res.x;
        ((c_scalar_t*)sh_red)[idx + 8 * c_sh_stride] = res.y;
      } else {
        ((c_scalar_t2*)sh_red)[idx] = res;
      }
    };

    if (threadIdx.x / 32 < tb_n_warps) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int j = 0; j < (is_a_8bit ? 2 : 4); j++) {
          if constexpr (m_block_size_8) {
            int wr = c_sh_wr + 16 * j;
            write(wr, frag_c[i][j][0][0], frag_c[i][j][0][1],
                  frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + 8, frag_c[i][j][0][2], frag_c[i][j][0][3],
                  frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          } else {
            int wr = c_sh_wr + 8 * j;
            write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0],
                  frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2],
                  frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0],
                  frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
            write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2],
                  frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          }
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

  #pragma unroll
    for (int i = 0;
         i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
        if (use_atomic_add && slice_count > 1) {
          c_scalar_t2* C_half2 = reinterpret_cast<c_scalar_t2*>(&C[c_gl_wr]);
          c_scalar_t2* sh_red_half2 =
              reinterpret_cast<c_scalar_t2*>(&sh_red[c_sh_rd]);
  #pragma unroll
          for (int a = 0; a < 4; a++) {
            atomicAdd(&C_half2[a], sh_red_half2[a]);
          }
        } else {
          C[c_gl_wr] = sh_red[c_sh_rd];
        }
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  };

  auto start_pipes = [&]() {

  #pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      if constexpr (has_zp && !is_zp_float && group_blocks == -1) {
        if (i == 0) {
          fetch_col_zp_to_shared();
          if constexpr (!dequant_skip_flop) {
            fetch_col_scale_to_shared();
          }
        }
      }
      fetch_to_shared(i, i, i < slice_iters);
    }

    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  if (slice_iters) {
    start_pipes();
  }

  while (slice_iters) {

  #pragma unroll
    for (int pipe = 0; pipe < stages;) {
  #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        fetch_zp_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }

        if constexpr (!is_a_8bit) {
          matmul(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));
        } else {
          static_assert(group_blocks != 0 && group_blocks != 1);
          matmul_a8(k);
        }
      }
      slice_iters--;
      if (slice_iters == 0) {
        break;
      }
    }

    a_gl_rd += a_gl_rd_delta_o * stages;

    if (slice_iters == 0) {

      if constexpr (use_fp16_accum) {
  #pragma unroll
        for (int i = 0; i < (thread_m_blocks * (is_a_8bit ? 2 : 4) * 2); i++) {
          float* frag_c_part_float = reinterpret_cast<float*>(frag_c) + i * 4;
          scalar_t* frag_c_part_half =
              reinterpret_cast<scalar_t*>(frag_c_part_float);

  #pragma unroll
          for (int i = 3; i >= 0; i--) {
            frag_c_part_float[i] = Cdtype::num2float(frag_c_part_half[i]);
          }
        }
      }

      if constexpr (is_a_8bit) {
        float frag_a_s[2 * thread_m_blocks];

        for (int i = 0; i < 2 * thread_m_blocks; i++)
          frag_a_s[i] = sh_a_s[i * 8 + (threadIdx.x % 32) / 4];

  #pragma unroll
        for (int j = 0; j < 2; j++) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][0][g];

              if constexpr (a_type == vllm::kS8) {
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][0][g] = c_val * s_val;
            }
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][1][g];

              if constexpr (a_type == vllm::kS8) {
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][1][g] = c_val * s_val;
            }
          }
        }
      }

      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;

      if constexpr (group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (b_type.size_bits() == 8 || (last || use_atomic_add) || is_a_8bit) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      thread_block_reduce();

      if (has_bias && last) {
        __syncthreads();
        cp_async4_pred(&sh_bias[bias_sh_wr], &b_bias_ptr[bias_gl_rd],
                       threadIdx.x < 16 * thread_n_blocks / 8);
        cp_async_fence();
      }

      if constexpr (group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp || is_a_8bit)) {
        if constexpr (is_a_8bit) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
          }
        } else if (b_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            if constexpr (m_block_size_8) {
              int idx = (threadIdx.x / 4) % 2;
              c_scalar_t2* frag_s_half2 =
                  reinterpret_cast<c_scalar_t2*>(frag_s);
  #pragma unroll
              for (int i = 0; i < 8; i++) {
                frag_s_half2[i] = Cdtype::num2num2(
                    reinterpret_cast<c_scalar_t*>(&frag_s_half2[i])[idx]);
              }
            }
          }
        }
      }

      if constexpr (group_blocks == -1 && is_a_8bit) {
  #pragma unroll
        for (int j = 0; j < 2; j++) {
          float2 aa[2];
          aa[0] = Cdtype::num22float2(frag_s[0][j * 2][0]);
          aa[1] = Cdtype::num22float2(frag_s[0][j * 2 + 1][0]);

  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[0])[g % 2];
              frag_c[i][j][0][g] *= scale;
            }

  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[1])[g % 2];
              frag_c[i][j][1][g] *= scale;
            }
          }
        }
      } else if (group_blocks == -1 && b_type.size_bits() == 8 &&
                 (has_zp && dequant_skip_flop || !has_zp)) {
        if (threadIdx.x / 32 < tb_n_warps) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<c_type_id>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][0]),
                  frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<c_type_id>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][2]),
                  frag_s[j / 2][2 * (j % 2) + (m_block_size_8 ? 1 : 0)]);

              if constexpr (!m_block_size_8) {
                scale_float<c_type_id>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][0]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
                scale_float<c_type_id>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][2]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
              }
            }
          }
        }
      }

      if (slice_count > 1 && !use_atomic_add) {

        barrier_acquire(&locks[locks_off], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[locks_off], last);
      }

      if (has_bias && last) {
        cp_async_wait<0>();
        __syncthreads();
        reinterpret_cast<int4*>(&frag_bias)[0] = sh_bias[bias_sh_rd];
        if constexpr (!is_a_8bit)
          reinterpret_cast<int4*>(&frag_bias)[1] = sh_bias[bias_sh_rd + 4];
        __syncthreads();
      }

      if (use_atomic_add && slice_count > 1 && slice_idx != 0)
        wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)

        write_result(last);
      slice_row = 0;
      if (!in_part2) {
        slice_col_par += gridDim.x;
      } else {
        slice_col_par++;
        slice_col++;
      }
      is_first_matmul_in_slice = true;
      init_slice();

      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
        a_gl_rd += a_gl_rd_delta_o * slice_row;
        b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) +
                  (threadIdx.x % b_sh_stride);
        b_gl_rd += b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;

        bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
        if constexpr (group_blocks == -1) {
          s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
        } else if constexpr (group_blocks >= thread_k_blocks) {
          s_gl_rd =
              s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd =
              zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              zp_sh_stride * slice_col + threadIdx.x;
        } else {
          s_gl_rd =
              s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                             threadIdx.x / s_sh_stride) +
              s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
          zp_gl_rd =
              zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                              threadIdx.x / zp_sh_stride) +
              zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
        }
        start_pipes();
      }
    }
  }
}

}

#endif

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#define MARLIN_KERNEL_PARAMS                                               \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                  \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                      \
      const int4 *__restrict__ b_bias_ptr,                                 \
      const float *__restrict__ a_scales_ptr,                              \
      const int4 *__restrict__ scales_ptr,                                 \
      const float *__restrict__ global_scale_ptr,                          \
      const int4 *__restrict__ zp_ptr, int prob_m, int prob_n, int prob_k, \
      int lda, int *locks, bool has_bias, bool use_atomic_add,             \
      bool use_fp32_reduce, int max_shared_mem

namespace MARLIN_NAMESPACE_NAME {
template <const vllm::ScalarTypeId a_type_id,
          const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id,
          const vllm::ScalarTypeId s_type_id,
          const int threads,
          const int thread_m_blocks,

          const int thread_n_blocks,
          const int thread_k_blocks,
          const bool m_block_size_8,

          const int stages,

          const int group_blocks,

          const bool is_zp_float
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 1, 8, 8, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 8, 4, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 4, 8, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 2, 16, 4, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 2, 8, 4, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 2, 4, 8, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 3, 16, 4, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 3, 8, 4, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 3, 4, 8, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 4, 16, 4, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 4, 8, 4, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 4, 4, 8, false, 2, 1, false>( MARLIN_KERNEL_PARAMS );

}

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

}

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

}

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

}

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

}

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

}

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>( MARLIN_KERNEL_PARAMS );

}

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace marlin {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

}

at::Tensor marlin_gemm(
    at::Tensor& a, std::optional<at::Tensor> c_or_none,
    at::Tensor& b_q_weight,
    std::optional<at::Tensor> const& b_bias_or_none,
    at::Tensor& b_scales,
    std::optional<at::Tensor> const& a_scales_or_none,
    std::optional<at::Tensor> const& global_scale_or_none,
    std::optional<at::Tensor> const& b_zeros_or_none,
    at::Tensor& workspace, vllm::ScalarTypeId const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k, bool use_atomic_add,
    bool use_fp32_reduce, bool is_zp_float) {
  STD_TORCH_CHECK_NOT_IMPLEMENTED(false,
                                  "marlin_gemm(..) requires CUDA_ARCH >= 7.5");
  return torch::empty({1, 1});
}

#else

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {

    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {

    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          int stages) {
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  int tb_scales = tb_groups * tb_n * 2;
  return tb_scales * stages;
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, int has_zp, bool is_zp_float,
                          bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size = get_scales_cache_size(th_config, prob_m, prob_n, prob_k,
                                        num_bits, group_size, stages);
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, int has_zp, bool is_zp_float,
                     bool is_a_8bit, int stages, int max_shared_mem) {

  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  if (th_config.num_threads < 128) {
    return false;
  }

  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(const vllm::ScalarType a_type,
                                const vllm::ScalarType b_type,
                                const vllm::ScalarType c_type,
                                const vllm::ScalarType s_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_zp, int group_blocks, int threads,
                                bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU8B128 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE4M3fn && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 1, 8, 8, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 8, 4, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 4, 8, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 2, 16, 4, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 2, 8, 4, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 2, 4, 8, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 3, 16, 4, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 3, 8, 4, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 3, 4, 8, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 4, 16, 4, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 4, 8, 4, false, 2, 1, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kFE2M1f && c_type == vllm::kFloat16 && s_type == vllm::kFE4M3fn && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 1 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 128, 4, 4, 8, false, 2, 1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == -1 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, -1, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 2 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 2, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 4 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 4, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFE4M3fn)
  STD_TORCH_CHECK(false, "marlin kernel with fp8 activation is not built.");
  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_zp, bool is_zp_float, int is_a_8bit,
    int stages, int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_zp, is_zp_float, is_a_8bit,
                         stages, max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
        group_size, has_zp, is_zp_float, is_a_8bit, stages);

    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks,
        th_config.thread_n / 16, th_config.thread_k / 16, m_block_size_8,
        has_zp, group_blocks, th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, int prob_m,
               int prob_n, int prob_k, int lda, void* workspace,
               vllm::ScalarType const& a_type, vllm::ScalarType const& b_type,
               vllm::ScalarType const& c_type, vllm::ScalarType const& s_type,
               bool has_bias, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  bool is_a_8bit = a_type.size_bits() == 8;
  STD_TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [",
                  prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    STD_TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                    " is not divisible by group_blocks = ", group_blocks);
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  int* locks = (int*)workspace;

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  STD_TORCH_CHECK(max_shared_mem > 0);

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 75,
                  "marlin kernel only support Turing or newer GPUs.");

  int stages = 2;
  if (major_capability == 7 && minor_capability == 5) {
    STD_TORCH_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                    "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 89,
                    "FP8 only support Ada Lovelace or newer GPUs.");
    STD_TORCH_CHECK(
        major_capability * 10 + minor_capability == 89 ||
            major_capability == 12,
        "Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      STD_TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                      " is not divisible by thread_n = ", thread_n);
      STD_TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                      " is not divisible by thread_k = ", thread_k);
    } else {

      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_zp,
          is_zp_float, is_a_8bit, stages, max_shared_mem, sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size, has_zp,
                              is_zp_float, is_a_8bit, stages,
                              max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    STD_TORCH_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_zp, is_zp_float,
                        is_a_8bit, stages, max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
        is_zp_float, stages);

    if (kernel == MarlinDefault) {
      STD_TORCH_CHECK(
          false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n, ", ",
          prob_k, "]", ", num_groups = ", num_groups,
          ", group_size = ", group_size, ", prob_m_split = ", prob_m_split,
          ", thread_m_blocks = ", thread_m_blocks,
          ", thread_n_blocks = ", thread_n_blocks,
          ", thread_k_blocks = ", thread_k_blocks,
          ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);

    bool is_a_8bit = a_type.size_bits() == 8;
    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}

at::Tensor marlin_gemm(
    at::Tensor& a, std::optional<at::Tensor> c_or_none,
    at::Tensor& b_q_weight,
    std::optional<at::Tensor> const& b_bias_or_none,
    at::Tensor& b_scales,
    std::optional<at::Tensor> const& a_scales_or_none,
    std::optional<at::Tensor> const& global_scale_or_none,
    std::optional<at::Tensor> const& b_zeros_or_none,
    at::Tensor& workspace, vllm::ScalarTypeId const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k, bool use_atomic_add,
    bool use_fp32_reduce, bool is_zp_float) {
  vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

  auto c_scalar_type = a.scalar_type();
  if (a.scalar_type() == at::kHalf) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else if (a.scalar_type() == at::kBFloat16) {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  } else {
    c_scalar_type = b_scales.scalar_type();
    if (b_scales.scalar_type() == at::kHalf) {
      c_type_id = vllm::kFloat16.id();
    } else if (b_scales.scalar_type() ==
               at::kBFloat16) {
      c_type_id = vllm::kBFloat16.id();
    } else {
      c_type_id = vllm::kBFloat16.id();

      STD_TORCH_CHECK(c_or_none.has_value(), "c must be passed for W4A8-FP4");
      at::Tensor c = c_or_none.value();
      c_scalar_type = c.scalar_type();

      if (c.scalar_type() == at::kHalf) {
        c_type_id = vllm::kFloat16.id();
      } else if (c.scalar_type() == at::kBFloat16) {
        c_type_id = vllm::kBFloat16.id();
      } else {
        STD_TORCH_CHECK(false, "unsupported c dtype");
      }
    }

    if (a.scalar_type() == at::kFloat8_e4m3fn) {
      a_type_id = vllm::kFE4M3fn.id();
    } else if (a.scalar_type() == at::kChar) {
      a_type_id = vllm::kS8.id();
    } else {
      STD_TORCH_CHECK(false, "unsupported `a` scalar_type");
    }
  }

  s_type_id = c_type_id;
  if (b_type_id == vllm::kFE2M1f.id()) {
    if (b_scales.scalar_type() ==
        at::kFloat8_e4m3fn) {
      s_type_id = vllm::kFE4M3fn.id();
    } else if (b_scales.scalar_type() ==
               at::kFloat8_e8m0fnu) {
      s_type_id = vllm::kFE8M0fnu.id();
    } else {
      STD_TORCH_CHECK(
          false, "When b_type = float4_e2m1f, b_scale scalar type must be",
          "float8_e4m3fn (for NVFP4) or float8_e8m0fnu (for MXFP4).");
    }
  } else if (b_type_id == vllm::kFE4M3fn.id() &&
             b_scales.scalar_type() ==
                 at::kFloat8_e8m0fnu) {
    s_type_id = vllm::kFE8M0fnu.id();
  }

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

  int pack_factor = 32 / b_type.size_bits();

  STD_TORCH_CHECK(a.size(0) == size_m,
                  "Shape mismatch: a.size(0) = ", a.size(0),
                  ", size_m = ", size_m);
  STD_TORCH_CHECK(a.size(1) == size_k,
                  "Shape mismatch: a.size(1) = ", a.size(1),
                  ", size_k = ", size_k);

  STD_TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  STD_TORCH_CHECK(
      (size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(0),
      "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
      ", size_k = ", size_k,
      ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  STD_TORCH_CHECK(
      b_q_weight.size(1) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(1) = ", b_q_weight.size(1),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  STD_TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
                  ", actual_size_n = ", actual_size_n);

  STD_TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  STD_TORCH_CHECK(a.stride(1) == 1, "A.stride(1) is not 1");

  STD_TORCH_CHECK(a.stride(0) % 8 == 0, "A.stride(0) must divisible by 8");
  STD_TORCH_CHECK(((uint64_t)a.const_data_ptr()) % 16 == 0,
                  "A must aligned to 16 bytes");

  STD_TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  STD_TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  STD_TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  STD_TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  at::Tensor a_scales;
  const auto device = a.device();

  if (a_scales_or_none.has_value()) {
    a_scales = a_scales_or_none.value();
    STD_TORCH_CHECK(a_type.size_bits() == 8,
                    "a_scales can only be used for 8bit activation.");
  } else {
    a_scales = torch::empty({0}, torch::TensorOptions().device(device).dtype(at::kFloat));
    STD_TORCH_CHECK(
        a_type.size_bits() != 8,
        "the a_scales parameter must be passed for 8bit activation.");
  }

  int thread_k = -1;

  int thread_n = -1;

  int sms = -1;
  const int32_t device_index = a.get_device();
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_index);

  c10::cuda::CUDAGuard device_guard(device_index);
  at::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    STD_TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    STD_TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    STD_TORCH_CHECK(c.size(0) == size_m,
                    "Shape mismatch: c.size(0) = ", c.size(0),
                    ", size_m = ", size_m);
    STD_TORCH_CHECK(c.size(1) == size_n,
                    "Shape mismatch: c.size(1) = ", c.size(1),
                    ", size_n = ", size_n);
  } else {
    c = torch::empty({size_m, size_n},
                     torch::TensorOptions().device(device).dtype(c_scalar_type));
  }
  if (size_m == 0) return c;

  at::Tensor c_tmp;
  if (use_fp32_reduce) {
    int max_m_block_size = (size_m + 16 - 1) / 16 * 16;
    max_m_block_size = min(max_m_block_size, 64);
    int max_c_tmp_size =
        sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
    c_tmp = torch::empty({max_c_tmp_size},
                         torch::TensorOptions().device(device).dtype(at::kFloat));
  } else {
    c_tmp = torch::empty({0}, torch::TensorOptions().device(device).dtype(at::kFloat));
  }

  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.dim();
  STD_TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");
  STD_TORCH_CHECK(b_scales.size(1) == size_n,
                  "b_scales dim 1 = ", b_scales.size(1),
                  " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  if (num_groups > 1) {
    STD_TORCH_CHECK(
        size_k % num_groups == 0, "size_k = ", size_k,
        ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
    group_size = size_k / num_groups;
  } else {
    group_size = -1;
  }

  at::Tensor global_scale;
  if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    STD_TORCH_CHECK(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
                    "global_scale can only be used for nvfp4 format.");
  } else {
    global_scale = torch::empty({0},
                                torch::TensorOptions().device(device).dtype(at::kFloat));
    STD_TORCH_CHECK(
        !(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
        "the global_scale parameter must be passed for nvfp4 format.");
  }

  bool has_bias = b_bias_or_none.has_value();
  at::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    STD_TORCH_CHECK(b_bias.device().is_cuda(), "b_bias is not on GPU");
    STD_TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");
    STD_TORCH_CHECK(b_bias.size(0) == size_n, "b_bias.size(0) != size_n");
    STD_TORCH_CHECK(b_bias.stride(0) == 1, "b_bias.stride(0) != 1");
  } else {
    b_bias = torch::empty({0}, torch::TensorOptions().device(device).dtype(c_scalar_type));
  }

  at::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    STD_TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    STD_TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = torch::empty({0}, torch::TensorOptions().device(device).dtype(c_scalar_type));
  }
  bool has_zp = b_zeros.size(-1) > 0;
  if (has_zp) {
    STD_TORCH_CHECK(
        b_type == vllm::kU4 || b_type == vllm::kU8,
        "b_type must be u4 or u8 when has_zp = True. Got = ", b_type.str());
  } else {
    STD_TORCH_CHECK(b_type == vllm::kU4B8 || b_type == vllm::kU8B128 ||
                        b_type == vllm::kS4 || b_type == vllm::kS8 ||
                        b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
                    "b_type must be uint4b8, uint8b128, int4, int8, "
                    "float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
                    b_type.str());
  }

  if (has_zp && is_zp_float) {
    STD_TORCH_CHECK(
        a.scalar_type() == at::kHalf,
        "Computation type must be float16 (half) when using float zero "
        "points.");
  }

  if (has_zp) {
    int rank = b_zeros.dim();
    STD_TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");
    if (is_zp_float) {
      STD_TORCH_CHECK(b_zeros.size(1) == size_n,
                      "b_zeros dim 1 = ", b_zeros.size(1),
                      " is not size_n = ", size_n);
      STD_TORCH_CHECK(num_groups == b_zeros.size(0),
                      "b_zeros dim 0 = ", b_zeros.size(0),
                      " is not num_groups = ", num_groups);
      STD_TORCH_CHECK(num_groups != -1, "num_groups must be != -1");
    } else {
      STD_TORCH_CHECK(b_zeros.size(0) == num_groups,
                      "b_zeros dim 0 = ", b_zeros.size(0),
                      " is not num_groups = ", num_groups);
      STD_TORCH_CHECK(b_zeros.size(1) == size_n / pack_factor,
                      "b_zeros dim 1 = ", b_zeros.size(1),
                      " is not size_n / pack_factor = ", size_n / pack_factor);
    }
  }

  STD_TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
                  "size_n = ", size_n, ", is not divisible by min_thread_n = ",
                  MARLIN_NAMESPACE_NAME::min_thread_n);

  int min_workspace_size = sms;
  STD_TORCH_CHECK(workspace.numel() >= min_workspace_size,
                  "workspace.numel = ", workspace.numel(),
                  " is below min_workspace_size = ", min_workspace_size);

  STD_TORCH_CHECK(
      a_scales.scalar_type() == at::kFloat,
      "scalar type of a_scales must be float");
  STD_TORCH_CHECK(
      global_scale.scalar_type() == at::kFloat,
      "scalar type of global_scale must be float");
  if (a_type.size_bits() == 16) {
    STD_TORCH_CHECK(
        a.scalar_type() == c.scalar_type(),
        "scalar type of a must be the same with c for 16 bit activation");
  }

  marlin::marlin_mm(
      a.const_data_ptr(), b_q_weight.const_data_ptr(), c.mutable_data_ptr(),
      c_tmp.mutable_data_ptr(), b_bias.mutable_data_ptr(),
      a_scales.mutable_data_ptr(), b_scales.mutable_data_ptr(),
      global_scale.mutable_data_ptr(), b_zeros.mutable_data_ptr(), size_m,
      size_n, size_k, a.stride(0), workspace.mutable_data_ptr(), a_type, b_type,
      c_type, s_type, has_bias, has_zp, num_groups, group_size, device_index,
      at::cuda::getCurrentCUDAStream(device_index), thread_k, thread_n, sms,
      use_atomic_add, use_fp32_reduce, is_zp_float);

  return c;
}

#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &marlin_gemm, py::arg("a"), py::arg("c"),
        py::arg("b_q_weight"), py::arg("b_bias"), py::arg("b_scales"),
        py::arg("a_scales"), py::arg("global_scale"), py::arg("b_zeros"),
        py::arg("workspace"), py::arg("b_type_id"), py::arg("size_m"),
        py::arg("size_n"), py::arg("size_k"), py::arg("use_atomic_add") = false,
        py::arg("use_fp32_reduce") = false, py::arg("is_zp_float") = false,
        py::doc("Marlin quantized GEMM (fp16/bf16 activation, u4 weight)"));
  m.def("k_u4", []() { return (int64_t)vllm::kU4.id(); },
        py::doc("ScalarType id for u4 with zero point"));
  m.def("k_u4b8", []() { return (int64_t)vllm::kU4B8.id(); },
        py::doc("ScalarType id for u4b8 (no zero point)"));
}
