#pragma once
#include <array>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <type_traits>

namespace tensor{

// TENSOR DIMENSIONS PRODUCT
// templatized dimension list --> product (for Tensor underlying array size)
template <size_t...Ds>
struct TensorDimsProduct;

// base case
template<>
struct TensorDimsProduct<> {
    static constexpr size_t value = 1;
};

// recursive case
template<size_t First, size_t... Rest>
struct TensorDimsProduct<First, Rest...> {
    static constexpr size_t value = First * TensorDimsProduct<Rest...>::value;
};


// GET DIMENSION FROM TENSOR DIMENSIONS LIST
template <size_t N, size_t... Ds>
struct SizeTemplateGet;

// base case: first element
template<size_t First, size_t... Rest>
struct SizeTemplateGet<0, First, Rest...> {
    static constexpr size_t value = First;
};

// recursive case
template<size_t N, size_t First, size_t... Rest>
struct SizeTemplateGet<N, First, Rest...> {
    // peel off, ditch first, until N = 0
    static constexpr size_t value = SizeTemplateGet<N - 1, Rest...>::value;
};


// STRIDES: for N-Rank indexing in flat vector, we need stride vector
// stride[0] = TensorDimsProduct<A, B, ..., N>, stride[-2] = N, stride[-1] = 1
template<size_t... Ds>
struct ComputeStrides;

// base case: no dim
template<>
struct ComputeStrides<> {
    static constexpr std::array<size_t, 0> value = {};
};

// base case: one dim
template<size_t D>
struct ComputeStrides<D> {
    static constexpr std::array<size_t, 1> value = {1};
};

// recursive case: stride[0] is product of remaining dims, then recurse
template<size_t First, size_t Second, size_t... Rest>
struct ComputeStrides<First, Second, Rest...> {
    static constexpr auto tail = ComputeStrides<Second, Rest...>::value;
    static constexpr size_t N = 1 + tail.size();

    static constexpr std::array<size_t, N> compute() {
        std::array<size_t, N> result{};
        // stride[0] = Second * Rest... = product of everything after first
        result[0] = TensorDimsProduct<Second, Rest...>::value;
        for (size_t i = 0; i < tail.size(); i++) {
            result[i + 1] = tail[i];
        }
        return result;
    }

    static constexpr auto value = compute();
};

//<f=3,s=2,r=>
//  <2,>--tail: [1]
//      <>--tail: []
//      N = 1; result = [1] (from product base case)
//      value = [1]
//  N = 2; result = [2,1]
//  value = [2,1]


// TENSOR
template<size_t... Dims>
class Tensor {
public:
    // rank = number of dimensions
    static constexpr size_t Rank = sizeof...(Dims);
    // size = total number of values (size of array)
    static constexpr size_t Size = TensorDimsProduct<Dims...>::value;
    // shape = array of dims
    static constexpr std::array<size_t, Rank> Shape = {Dims...};
    // strides = used for seamless indexing
    static constexpr std::array<size_t, Rank> Strides = ComputeStrides<Dims...>::value;
private:
    std::array<float, Size> data_{};
public:
    Tensor() = default;

    // construct from initializer list
    Tensor(std::initializer_list<float> init) {
        size_t i = 0;
        for (auto v : init) {
            if (i < Size){
                data_[i++] = v;
            }
        }
    }

    // fill with a value
    void fill(float v) { data_.fill(v); }

    // get underlying array
    float* data() {return data_.data();}
    [[nodiscard]] const float* data() const {return data_.data();}

    // flat indexing
    float& flat(size_t idx) {return data_[idx];}
    [[nodiscard]] float flat(size_t idx) const {return data_[idx];}

    // map: apply f(float) -> float element-wise, return new tensor
    template<typename F>
    Tensor map(F f) const {
        Tensor out;
        for (size_t i = 0; i < Size; ++i) out.data_[i] = f(data_[i]);
        return out;
    }

    // zip: apply f(float, float) -> float element-wise with another tensor, return new tensor
    template<typename F>
    Tensor zip(const Tensor& other, F f) const {
        Tensor out;
        for (size_t i = 0; i < Size; ++i) out.data_[i] = f(data_[i], other.data_[i]);
        return out;
    }

    // apply: mutate each element in-place with f(float&)
    template<typename F>
    void apply(F f) {
        for (size_t i = 0; i < Size; ++i) f(data_[i]);
    }

    // zip_apply: mutate each element in-place using corresponding element from other with f(float&, float)
    template<typename F>
    void zip_apply(const Tensor& other, F f) {
        for (size_t i = 0; i < Size; ++i) f(data_[i], other.data_[i]);
    }

#define ACCESS_IMPL {                                                                   \
    static_assert(sizeof...(idxs) == Rank,"Number of indices must match tensor rank");  \
    const size_t idx_arr[] = {static_cast<size_t>(idxs)...};                            \
    /* flat index is simply a dot product between requested indices and strides ! */    \
    size_t flat_index = 0;                                                              \
    for (size_t i = 0; i < Rank; i++) {                                                 \
        flat_index += idx_arr[i] * Strides[i];                                          \
    }                                                                                   \
    return data_[flat_index];                                                           \
    }
    // proper dimensional indexing
    template<typename... Indices>
    float& operator()(Indices... idxs) ACCESS_IMPL
    template<typename... Indices>
    float operator()(Indices... idxs) const ACCESS_IMPL
#undef ACCESS_IMPL


    void print(const char* name = nullptr) const {
        if (name) std::cout << name << " ";
        std::cout << "Tensor<";
        for (size_t i = 0; i < Rank; ++i) {
            if (i > 0) std::cout << ",";
            std::cout << Shape[i];
        }
        std::cout << "> strides=(";
        for (size_t i = 0; i < Rank; ++i) {
            if (i > 0) std::cout << ",";
            std::cout << Strides[i];
        }
        std::cout << ")\n";
    }

};


// EINSUM: generalized tensor contraction
// contract index I from Tensor A and index J from Tensor B
// Example:
//      multiplication of two Rank-0 Tensors: no contraction, no dimensions, just multiply
//      dot product of two Rank-1 Tensors: contract non-1 dimension of each (user must specify...for two row vectors it would be einsum<1,1>)
//      cross product of two Rank-1 Tensors: contract 1 dimensions of each (for two row vectors it would be einsum<0,0>)
//      matmul two Rank-2 Tensors: typically einsum<1,0>
//  Helpers:

// ConcatTensors: concatenate two tensor shapes into one
template<typename T1, typename T2>
struct TensorConcat;

template<size_t...Ds1, size_t...Ds2>
struct TensorConcat<Tensor<Ds1...>, Tensor<Ds2...>> {
    using type = Tensor<Ds1..., Ds2...>;
};
// thin wrapper to present as new Tensor type
// so when we need to remove axis and suture, we'll get the type this way



// now need beautiful helper: constexpr array --> Tensor
template<typename KeptIdxs, typename Iota>
struct ArrayToTensor;

template<typename KeptIdxs, size_t... Iota>
struct ArrayToTensor<KeptIdxs, std::index_sequence<Iota...>> {
    static constexpr auto arr = KeptIdxs::value;
    using type = Tensor<arr[Iota]...>;
};


// Build a Tensor type from selected indices of a Shape array
// Given a pack Dims... and an axis to skip, produce Tensor<remaining dims...>
// wrap kept dimensions array in a type so it can be passed to the above as template!
template <size_t Skip, size_t... Dims>
struct KeptDimsHolder {
    static constexpr auto value = [] {
        constexpr std::array<size_t, sizeof...(Dims)> all = {Dims...};
        std::array<size_t, sizeof...(Dims) - 1> result{};
        size_t out = 0;
        for (size_t i = 0; i < sizeof...(Dims); ++i) {
            if (i != Skip) {
                result[out++] = all[i];
            }
        }
        return result;
    }();
};


// Finally; RemoveAxis. this returns the new Tensor type
template<size_t Skip, size_t... Dims>
struct RemoveAxis {
    using type = typename ArrayToTensor<
        // kept indices (actual values)
        KeptDimsHolder<Skip, Dims...>,
        // iota of the above to pattern-match and grab them into new Tensor type!
        std::make_index_sequence<sizeof...(Dims) - 1>
    >::type;
};



// now ready for Einsum

template<size_t I, size_t J, typename TA, typename TB>
struct EinsumResultType;

template<size_t I, size_t J, size_t... ADims, size_t... BDims>
struct EinsumResultType<I, J, Tensor<ADims...>, Tensor<BDims...>> {
    static_assert(SizeTemplateGet<I, ADims...>::value == SizeTemplateGet<J, BDims...>::value,
        "axis I from A and axis J from B must be same size!");

    using A_Reduced = typename RemoveAxis<I, ADims...>::type;
    using B_Reduced = typename RemoveAxis<J, BDims...>::type;
    using type = typename TensorConcat<A_Reduced, B_Reduced>::type;
};


// EINSUM FUNCTION
// einsum<I, J>(A, B) contracts axis I from A with axis J from B
//
// we iterate over free indices in EinsumResultType, and sum over contracted indices

template<size_t I, size_t J, typename TA, typename TB>
auto Einsum(){};

template<size_t I, size_t J, size_t... ADims, size_t... BDims>
auto Einsum(const Tensor<ADims...>& A, const Tensor<BDims...>& B) {
    using Result = typename EinsumResultType<I, J, Tensor<ADims...>, Tensor<BDims...>>::type;

    constexpr size_t A_Rank = Tensor<ADims...>::Rank;
    constexpr size_t B_Rank = Tensor<BDims...>::Rank;
    constexpr size_t ContractDimSize = SizeTemplateGet<I, ADims...>::value;

    constexpr auto A_Strides = Tensor<ADims...>::Strides;
    constexpr auto B_Strides = Tensor<BDims...>::Strides;

    Result C;
    C.fill(0.0f);

    constexpr size_t C_Rank = Result::Rank;
    constexpr auto C_Strides = Result::Strides;

    // loop over raw C
    //      to do this, we must decompose a C-array index into a multi C index
    //      map the first (A_Rank - 1) components to A's free indices
    //      map the next/last (B_Rank - 1) components to B's free indices
    //      loop over contracted dimension

#define GET_FREE_AXES(idx, tensor_rank) [] {        \
    std::array<size_t, tensor_rank - 1> result{};   \
    size_t out = 0;                                 \
    for (size_t i = 0; i < tensor_rank; i++) {      \
        if (i != idx) {                             \
            result[out++] = i;                      \
        }                                           \
    }                                               \
    return result;                                  \
}();

    constexpr auto a_free_axes = GET_FREE_AXES(I, A_Rank)
    constexpr auto b_free_axes = GET_FREE_AXES(J, B_Rank)
#undef GET_FREE_AXES


    // main loop
    for (size_t c_flat = 0; c_flat < Result::Size; c_flat++) {
        // decompose c_flat into multi-index
        std::array<size_t, C_Rank> c_multi_index{};
        // starting at the flat index
        size_t temp = c_flat;
        // for each dimension (whose strides bumped us to c_flat!)
        for (size_t dimension = 0; dimension < C_Rank; dimension++) {
            // reverse the multiplication it contributed, leaving the strided
            c_multi_index[dimension] = temp / C_Strides[dimension];
            temp %= C_Strides[dimension];
        }
        /*
         * Example: Tensor<3,3> x;
         * x(2, 1) = 3 * 2 + 1 = 7
         * ---the array is 0-8 inclusive, [2][1] is second to last item!
         *
         * Example: Tensor<3,3,3> x;
         * x(0, 1, 2) = 9 * 0 + 3 * 1 + 2 = 5
         * ---harder to picture array, but this is first row, second column, third matrix.
         * x(2, 2, 1) = 9 * 2 + 3 * 2 + 1 = 25
         * ---array is 0-26 inclusive, [2][2][1] is second to last item!
         */

        // build A's and B's base flat indices (which will get us to the right zone to iterate over ContractDim)
        // a_base brings us to A[contracted_index][0]
        size_t a_base = 0;
        // for each free axis in A
        for (size_t freeA = 0; freeA < A_Rank - 1; freeA++) {
            // get that axis from C's multi-index and adjust with proper A stride
            // a_free_axes[] maps from free-space to A_Rank space
            a_base += c_multi_index[freeA] * A_Strides[a_free_axes[freeA]];
        }
        size_t b_base = 0;
        for (size_t freeB = 0; freeB < B_Rank - 1; freeB++) {
            // need to adjust for dimensions already taken by A
            b_base += c_multi_index[A_Rank - 1 + freeB] * B_Strides[b_free_axes[freeB]];
        }

        // CONTRACTION
        float sum = 0.0f;
        for (size_t k = 0; k < ContractDimSize; k++) {
            // go to base starting point, add dimension-adjusted stride for k-th item in that array
            const float a_val = A.flat(a_base + k * A_Strides[I]);
            const float b_val = B.flat(b_base + k * B_Strides[J]);
            sum += a_val * b_val;
        }

        C.flat(c_flat) = sum;
    }
    return C;
}


template<size_t... Dims>
Tensor<Dims...> operator+(const Tensor<Dims...>& a, const Tensor<Dims...>& b) {
    return a.zip(b, [](float x, float y) { return x + y; });
}
template<size_t... Dims>
Tensor<Dims...> operator-(const Tensor<Dims...>& a, const Tensor<Dims...>& b) {
    return a.zip(b, [](float x, float y) { return x - y; });
}
template<size_t... Dims>
Tensor<Dims...>& operator+=(Tensor<Dims...>& a, const Tensor<Dims...>& b) {
    a.zip_apply(b, [](float& x, float y) { x += y; });
    return a;
}
template<size_t... Dims>
Tensor<Dims...> operator*(const Tensor<Dims...>& a, float s) {
    return a.map([s](float x) { return x * s; });
}
template<size_t... Dims>
Tensor<Dims...> operator*(float s, const Tensor<Dims...>& a) { return a * s; }


// outer product (no contraction) specialization
template<size_t... ADims, size_t... BDims>
auto Einsum(const Tensor<ADims...>& A, const Tensor<BDims...>& B) {
    Tensor<ADims..., BDims...> C;
    // for each value of A
    for (size_t i = 0; i < Tensor<ADims...>::Size; i++) {
        // for each value of B
        for (size_t j = 0; j < Tensor<BDims...>::Size; j++) {
            C.flat(i * Tensor<BDims...>::Size + j) = A.flat(i) * B.flat(j);
            /*
             * Example: Tensor<3, 1> A; Tensor<1, 2> B;
             * [1][1] --> 1 * 2 + 1 = 3
             *
             * essentially, every new dimension of A (in the array) is separated by a full subarray 'from B', scaled by A[i]
             */
        }
    }
    return C;
}
}