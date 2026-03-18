#pragma once
#include "Tensor.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <numeric>
#include <ranges>
#include <algorithm>

using namespace tensor;

static constexpr float EPS = 1e-8f;
static constexpr float ADAM_BETA_1 = 0.9f;
static constexpr float ADAM_BETA_2 = 0.999f;

enum class ActivationFunction { Input, Sigmoid, ReLU, Softmax, Tanh };

// activation function
template<size_t N>
Tensor<N> Activate(const Tensor<N>& z, const ActivationFunction act) {
    switch (act) {
        case ActivationFunction::ReLU:
            return z.map([](const float x) {
                return x > 0.f ? x : 0.f;
            });
        case ActivationFunction::Sigmoid:
            return z.map([](const float x) {
                return 1.f / (1.f + std::exp(-x));
            });
        case ActivationFunction::Tanh: return z.map([](const float x) { return std::tanh(x); });
        case ActivationFunction::Softmax: {
            // subtract max for numerical stability
            float maxV = z.flat(0);
            for (size_t i = 1; i < N; ++i) if (z.flat(i) > maxV) maxV = z.flat(i);
            auto a = z.map([maxV](const float x) { return std::exp(x - maxV); });
            float sum = 0.f;
            for (size_t i = 0; i < N; ++i) sum += a.flat(i);
            a.apply([sum](float& x) { x /= sum; });
            return a;
        }
        default: return z;
    }
}

// given upstream gradient (dL/da) and post-activation a --> dL/dz
template<size_t N>
Tensor<N> ActivatePrime(const Tensor<N>& grad, const Tensor<N>& a, const ActivationFunction act) {
    switch (act) {
        case ActivationFunction::ReLU:
            return grad.zip(a, [](const float g, const float ai) { return g * (ai > 0.f ? 1.f : 0.f); });
        case ActivationFunction::Sigmoid:
            return grad.zip(a, [](const float g, const float ai) { return g * ai * (1.f - ai); });
        case ActivationFunction::Tanh:
            return grad.zip(a, [](const float g, const float ai) { return g * (1.f - ai * ai); });
        case ActivationFunction::Softmax: {
            float dot = 0.f;
            for (size_t i = 0; i < N; ++i) dot += a.flat(i) * grad.flat(i);
            return a.zip(grad, [dot](const float ai, const float gi) { return ai * (gi - dot); });
        }
        default: return grad;
    }
}

// CROSS ENTROPY LOSS
template<size_t N>
float CrossEntropyLoss(const Tensor<N>& output, const Tensor<N>& target) {
    auto indices = std::views::iota(size_t{0}, N);

    return std::accumulate(indices.begin(), indices.end(), 0.0f,
        [&target, &output](float current_loss, size_t i) {
            return current_loss - target.flat(i) * std::log(std::max(output.flat(i), EPS));
        }
    );
}

// xavier init for controlled init variance
template<size_t In, size_t Out>
void XavierInit(Tensor<Out, In>& W) {
    static std::mt19937 rng{std::random_device{}()};
    const float limit = std::sqrt(6.f / static_cast<float>(In + Out));
    std::uniform_real_distribution<float> dist{-limit, limit};
    W.apply([&dist](float& x) { x = dist(rng); });
}

// LAYER STRUCT
// weights, bias, activation, Adam moments
template<size_t In, size_t Out>
struct Layer {
    Tensor<Out, In> W;
    Tensor<Out> b{};          
    ActivationFunction actFunc = ActivationFunction::Input;

    // Adam first and second moments (0 init)
    Tensor<Out, In> mW{}, vW{};
    Tensor<Out> mb{}, vb{};
    Layer() = default;
    explicit Layer(const ActivationFunction a) : actFunc(a) { XavierInit(W); }

    // forward pass
    // uses Einsum to contract 2nd and 1st dimensions from W and x, respectively
    // then calls activation
    Tensor<Out> Forward(const Tensor<In>& x) const {
        // MATVEC contracts matrix's columns with column-vec's rows --> Tensor<NumRowsInWeight>
        auto z = Einsum<1, 0>(W, x) + b;
        return Activate(z, actFunc);
    }

    // backward pass
    // take in downstream gradient and take outer product with upstream activation: this is dL/dW
    Tensor<In> Backward(const Tensor<Out>& delta, const Tensor<In>& a_prev, float lr, float mCorr, float vCorr) {
        // dW = OUTER(delta, a_prev)
        // delta: Tensor<Out>, a_prev: Tensor<In>....outer prod --> Tensor<Out,In>, same dim as W

        // then apply Adam update to W and b
        AdamUpdate(Einsum(delta, a_prev), delta, lr, mCorr, vCorr);

        // then pass gradient upstream, defining dL/dA_prev:
        //      W: Tensor<Out,In>, delta: Tensor<Out>...contract first axis of each --> Tensor<In>, same dim as a_prev
        // (same thing as DOT(W.Transpose(), delta), but Einsum obviates Transpose!)
        return Einsum<0, 0>(W, delta);
    }

    // Adam update.
    // mCorr and vCorr are precomputed by NN. at the beginning, (low mT), corrections amplify
    // moments from 0-bias, but eventually corrections approach 1
    void AdamUpdate(const Tensor<Out, In>& dW, const Tensor<Out>& db, float lr, float mCorr, float vCorr) {
        // for each Weight and Bias, subtract LR * adjusted_First_Moment / sqrt(adjusted_Second_Moment)
        //      first moment approximates consistency of direction of update
        //      second moment approximates inverse of smoothness of local terrain on loss landscape
        for (size_t i = 0; i < Out * In; ++i) {
            const float g  = dW.flat(i);
            mW.flat(i) = ADAM_BETA_1 * mW.flat(i) + (1.f - ADAM_BETA_1) * g;
            vW.flat(i) = ADAM_BETA_2 * vW.flat(i) + (1.f - ADAM_BETA_2) * g * g;
            W.flat(i) -= lr * (mW.flat(i) * mCorr) / (std::sqrt(vW.flat(i) * vCorr) + EPS);
        }
        for (size_t i = 0; i < Out; ++i) {
            const float g  = db.flat(i);
            mb.flat(i) = ADAM_BETA_1 * mb.flat(i) + (1.f - ADAM_BETA_1) * g;
            vb.flat(i) = ADAM_BETA_2 * vb.flat(i) + (1.f - ADAM_BETA_2) * g * g;
            b.flat(i) -= lr * (mb.flat(i) * mCorr) / (std::sqrt(vb.flat(i) * vCorr) + EPS);
        }
    }

    void Save(std::ofstream& f) const {
        const auto a = static_cast<uint8_t>(actFunc);
        f.write(reinterpret_cast<const char*>(&a), 1);
        f.write(reinterpret_cast<const char*>(W.data()), Out * In * sizeof(float));
        f.write(reinterpret_cast<const char*>(b.data()), Out * sizeof(float));
    }
    void Load(std::ifstream& f) {
        uint8_t a; f.read(reinterpret_cast<char*>(&a), 1);
        actFunc = static_cast<ActivationFunction>(a);
        f.read(reinterpret_cast<char*>(W.data()), Out * In * sizeof(float));
        f.read(reinterpret_cast<char*>(b.data()), Out      * sizeof(float));
    }
};


// NEURAL NETWORK CLASS
// templatized by layer sizes
//      sizes[0] = input size
//      sizes[1...] = layer output sizes
// network is essentially a Tuple of Layer<In,Out> objects, where mLayers[i - 1]::Out == mLayers[i]::In

template<size_t... Sizes>
class NeuralNetwork {
    static_assert(sizeof...(Sizes) >= 2, "Need at least input size + one output size");

    // LAYER TUPLE TOOLS
    // necessary unpacker to make tuple of Layers from list of activation sizes
    // remember, these are all type operations!
    // we just need LayerTupleBuilder to give us the full type of the layer tuple

    // generic
    template<size_t... LayerSizes>
    struct LayerTupleBuilder;

    // base case: two left --> final layer
    template<size_t In, size_t Out>
    struct LayerTupleBuilder<In, Out> {
        using type = std::tuple<Layer<In, Out>>;
    };

    // recursive case
    template<size_t In, size_t Mid, size_t... Rest>
    struct LayerTupleBuilder<In, Mid, Rest...> {
        // get type of this tuple_cat monstrosity
        using type = decltype(std::tuple_cat(
            // declval allows us to peek around at types in an unevaluated context
            // we need the type of this tuple AS A VALUE for tuple_cat

            // this layer fully pops top of template list, peeks second-to-top, to create a Layer
            std::declval<std::tuple<Layer<In, Mid>>>(),
            // once again, extract type of recursive result !
            // recurse on the rest and forget about In
            std::declval<typename LayerTupleBuilder<Mid, Rest...>::type>()
        ));
    };
    /*
     * DECLVAL example from C++ ref:
     *
    *   #include <iostream>
        #include <utility>
        struct Default
        {
            int foo() const { return 1; }
        };

        struct NonDefault
        {
            NonDefault() = delete;
            int foo() const { return 1; }
        };

        int main()
        {
            decltype(Default().foo())                   n1 = 1;     // type of n1 is int
            decltype(std::declval<Default>().foo())     n2 = 1;     // same

            decltype(NonDefault().foo())               n3 = n1;     // error: no default constructor
            decltype(std::declval<NonDefault>().foo()) n3 = n1;     // type of n3 is int

            std::cout << "n1 = " << n1 << '\n'
                      << "n2 = " << n2 << '\n'
                      << "n3 = " << n3 << '\n';
        }
     *
     *
     */

    static constexpr size_t NumLayers = sizeof...(Sizes) - 1;

    // now we extract our hard-won std::tuple<Layer<>...> which is our network
    using LayerTuple = LayerTupleBuilder<Sizes...>::type;
    LayerTuple mLayers;
    int mT = 0;
    float mCorr = 1.0f;
    float vCorr = 1.0f;

public:
    static constexpr size_t InSize = SizeTemplateGet<0, Sizes...>::value;
    static constexpr size_t OutSize = SizeTemplateGet<sizeof...(Sizes) - 1, Sizes...>::value;

    using InputTensor = Tensor<InSize>;
    using OutputTensor = Tensor<OutSize>;

    // tuple of Tensors representing intermediate network activations
    using ActivationsTuple = std::tuple<Tensor<Sizes>...>;


    NeuralNetwork() = default;

    NeuralNetwork(std::initializer_list<ActivationFunction> acts) {
        std::array<ActivationFunction, NumLayers> actArr{};
        std::copy(acts.begin(), acts.end(), actArr.begin());

        auto attachActivationFunction = [&]<size_t... Is>(std::index_sequence<Is...>) {
            // use C++ lambda unfolding to assign .actFunc member to each Layer
            ((std::get<Is>(mLayers).actFunc = actArr[Is]), ...);
            // and to xavier initialize each
            (XavierInit(std::get<Is>(mLayers).W), ...);
        };
        attachActivationFunction(std::make_index_sequence<NumLayers>{});
    }


    [[nodiscard]] ActivationsTuple ForwardAll(const InputTensor& x) const {
        // declare tuple of activation Tensors
        ActivationsTuple A;
        // assign InputTensor to first activation Tensor
        std::get<0>(A) = x;
        // populate tuple with each layer
        forward_all_impl(A);
        // return activation Tensors
        return A;
    }

    [[nodiscard]] OutputTensor Forward(const InputTensor& x) const {
        // call forwardAll and grab last activation Tensor
        return std::get<NumLayers>(ForwardAll(x));
    }

    // special use TrainStep: pass in gradient wrt final layer logits
    // grad should be dL/dz; da/dz should already have been computed and multiplied into grad
    InputTensor TrainStepLogits(const InputTensor& x, const OutputTensor& grad, float lr) {
        const auto A = ForwardAll(x);
        tick_adam();
        return backward_update_impl<NumLayers>(A, grad, lr);
    }

    // raw full train step: pass in gradient wrt final layer output
    // grad should be dL/da
    void TrainStep(const InputTensor& x, const OutputTensor& grad, float lr) {
        const auto A = ForwardAll(x);
        // peel off activation, finding dL/dz, then call helper
        const auto lastDelta = ActivatePrime(grad, std::get<NumLayers>(A),
                                             std::get<NumLayers-1>(mLayers).actFunc);
        tick_adam();
        backward_update_impl<NumLayers>(A, lastDelta, lr);
    }


    void Save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("Cannot write: " + path);
        }
        // for each layer, call Layer.Save()
        auto writeLayers = [&]<size_t... Is>(std::index_sequence<Is...>) {
            (std::get<Is>(mLayers).Save(f), ...);
        };
        writeLayers(std::make_index_sequence<NumLayers>{});
    }
    void Load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("Cannot read: " + path);
        }
        auto readLayers = [&]<size_t... Is>(std::index_sequence<Is...>) {
            (std::get<Is>(mLayers).Load(f), ...);
        };
        readLayers(std::make_index_sequence<NumLayers>{});
    }

private:
    // increment Adam ticker and update moment corrections
    void tick_adam() {
        ++mT;
        mCorr = 1.0f / (1.0f - std::pow(ADAM_BETA_1, static_cast<float>(mT)));
        vCorr = 1.0f / (1.0f - std::pow(ADAM_BETA_2, static_cast<float>(mT)));
    }

    // internal recursive implementation of forward pass
    // uses Layer.Forward to populate ActivationsTuple with activation Tensors
    template<size_t I = 0>
    void forward_all_impl(ActivationsTuple& A) const {
        if constexpr (I < NumLayers) {
            // next activation Tensor = Layer[i].Forward(prev activation Tensor)
            // recall there are Sizes activation Tensors but NumLayers actual param layers
            std::get<I+1>(A) = std::get<I>(mLayers).Forward(std::get<I>(A));
            // recurse forward
            forward_all_impl<I+1>(A);
        }
        // base case is just termination because we have no return
    }

    // recursively backpropagate gradient and return InputTensor-sized/typed gradient
    // calls Layer.Backward --> AdamUpdate!

    // first template param is actually used, second is just for delta Tensor<_> that we take in...it's effectively free
    // 'I' starts as NumLayers, so it starts by peeling off last layer (that's how backprop works)
    template<size_t I, size_t DeltaSize>
    InputTensor backward_update_impl(const ActivationsTuple& A, const Tensor<DeltaSize>& delta, float lr) {
        // use Layer.Backward to update params + get gradient
        //                      last Layer's grad........delta, last Layer's activation Tensor
        const auto grad = std::get<I-1>(mLayers).Backward(delta, std::get<I-1>(A), lr, mCorr, vCorr);
        if constexpr (I > 1) {
            // Layer.Backward is raw outer product; we need to peel off ActivationFunc derivative here
            //                             gradient wrt A[I-1], A[I-1], ActivationFunction[I-2] which produces former <-
            const auto prev_delta = ActivatePrime(grad, std::get<I-1>(A), std::get<I-2>(mLayers).actFunc);
            // now recurse on prev layer with raw delta (ActivationFuncPrime) already peeled off
            return backward_update_impl<I-1>(A, prev_delta, lr);
        } 
        // because we have a if-constexpr (compile time if), we must pair it with an else.
        // even when I > 1, this code (if not else-wrapped) would run, causing type errors!
        else
        {
            // base case: gradient is already wrt Input
            return grad;
        }

    }
};
