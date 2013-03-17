#ifndef NEURAL_LAYER_HPP
#define NEURAL_LAYER_HPP

#include "vect.hpp"
#include <iostream>
#include <cmath>
#include <type_traits>
using namespace std;
using namespace math;

namespace ml {

enum activation { LINEAR, SIGMOID };
enum learning { ONLINE, BATCH };
enum type { HIDDEN, OUTPUT };
enum shared { SHARED, NOSHARED };

inline static double sigmoid( const double& value, const double lambda = 1.0 ) {
  return 1.0 / ( 1.0 + exp( -lambda * value ) );
}

template<size_t N, size_t I, activation A, learning L, type S, class T, shared V, bool = false>
class neural_layer {
protected:
  typedef vect<N, T> outType;
  typedef vect<I, T> inType;
  typedef vect<I + 1, T> richInType;

  typedef typename conditional<V == SHARED, shared_matrix<N, I + 1, T>, matrix<N, I + 1, T> >::type M;
  M __weights;
  outType     __output;   // last output
  richInType  __input;    // last input
  double eta;

  // add the fictitious input and save it in the member variable
  const richInType& setInput( const inType& input ) {
    __input[I] = -1.0;
    copy( input.data(), input.data() + I, __input.data() );

    return __input;
  }

public:
  typedef M weightsType;

  neural_layer ( Random& gen ) {
    init( gen );
  }

  neural_layer& init ( Random& gen ) {
    if ( A == LINEAR ) eta = 0.1;
    if ( A == SIGMOID ) eta = 0.8;

    for ( size_t i = 0; i < N; ++i )
      for ( size_t j = 0; j < I + 1; ++j )
        __weights(i,j) = 0.5 * gen.realnegative();
    return *this;
  }

  const outType& output() const {
    return __output;
  }

  inline static constexpr size_t size ( ) {
    return N * (I + 1);
  }

  inline outType& activation ( const outType& net ) {
    for ( size_t i = 0; i < N && A == SIGMOID; ++i ) {
      this->__output[i] = sigmoid( net[i] );
    }

    if ( A == LINEAR )
      this->__output = net;

    return this->__output;
  }

  // compute the component-wise delta calculation (in-place)
  inline outType computedelta ( outType& v ) {
    for ( size_t k = 0; k < N && A == SIGMOID; ++k )
      v[k] = v[k] * (1.0 - this->__output[k]) * this->__output[k];
    return v;
  }

  // compute the output of the network given an input
  const outType& compute ( const inType& input ) {
    // the calculation is basically a matrix-vector multiplication where
    // the input has been added a ficticious value, -1, so there are two possibilities:
    //   - copy the input vector in a bigger vector and perform the multiplication
    //   - split the multiplication as follows:
    //     out = -W0 + W*in
    // i choose to copy the input, it's O(n) instead of O(n^2)
    this->setInput( input );

    return this->activation( __weights * this->__input );
  }

  // the back-propagation (GD) algorithm
  inType backprop ( outType& error, M& dw ) {
    outType delta = this->computedelta( error );
    inType out;

    // compute the deltas for the previous layer, meaningful only for the output
    if ( S == OUTPUT ) {
      richInType propagate = __weights.transpose() * delta;
      for ( size_t i = 0; i < I; ++i ) out[i] = propagate[i];
    }

    // update the weights, the delta rule
    if ( L == ONLINE )
      __weights += this->eta * delta * this->__input.transpose();
    if ( L == BATCH )
      dw += this->eta * delta * this->__input.transpose();

    return out;
  }

  neural_layer& update ( M& dw ) {
    __weights += dw;
    return *this;
  }
};

template<size_t N, size_t I, activation A, learning L, type S, class T>
class neural_layer<N, I, A, L, S, T, SHARED, true> : neural_layer<N, I, A, L, S, T, SHARED, false> {
  typedef neural_layer<N, I, A, L, S, T, SHARED, false> base;
public:
  neural_layer ( Random& gen, T* data ) {
    this->__weights = base::weightsType( data );
    this->init( gen );
  }
};

} // namespace ml

#endif // NEURAL_LAYER_HPP
