#ifndef NEURAL_LAYER_HPP
#define NEURAL_LAYER_HPP

#include "vect.hpp"
#include <iostream>
#include <cmath>
using namespace std;
using namespace math;

namespace ml {

enum activation { LINEAR, SIGMOID };
enum learning { ONLINE, BATCH };
enum type { HIDDEN, OUTPUT };

inline static double sigmoid( const double& value, const double lambda = 1.0 ) {
  return 1.0 / ( 1.0 + exp( -lambda * value ) );
}

template<size_t N, size_t I, activation A, learning L, type S, class T>
class layer_base {
protected:
  typedef vect<N, T> outType;
  typedef vect<I, T> inType;
  typedef vect<I + 1, T> richInType;

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
  layer_base ( ) {
    if ( A == LINEAR ) eta = 0.1;
    if ( A == SIGMOID ) eta = 0.8;
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
  template<class M>
  const outType& compute_base ( const inType& input, const M& weights ) {
    // the calculation is basically a matrix-vector multiplication where
    // the input has been added a ficticious value, -1, so there are two possibilities:
    //   - copy the input vector in a bigger vector and perform the multiplication
    //   - split the multiplication as follows:
    //     out = -W0 + W*in
    // i choose to copy the input, it's O(n) instead of O(n^2)
    this->setInput( input );

    return this->activation( weights * this->__input );
  }

  // the back-propagation (GD) algorithm
  template<class M>
  inType backprop_base ( outType& error, M& weights, M& dw ) {
    outType delta = this->computedelta( error );
    inType out;

    // compute the deltas for the previous layer, meaningful only for the output
    if ( S == OUTPUT ) {
      richInType propagate = weights.transpose() * delta;
      for ( size_t i = 0; i < I; ++i ) out[i] = propagate[i];
    }

    // update the weights, the delta rule
    if ( L == ONLINE )
      weights += this->eta * delta * this->__input.transpose();
    if ( L == BATCH )
      dw += this->eta * delta * this->__input.transpose();

    return out;
  }
};

template<size_t N, size_t I, activation A, learning L, type S, class T>
class neural_layer : public layer_base<N, I, A, L, S, T> {
  typedef layer_base<N, I, A, L, S, T> base;
  typedef typename base::outType outType;
  typedef typename base::inType inType;
  matrix<N, I + 1, T> __weights;
public:
  typedef matrix<N, I + 1, T> weightsType;

  neural_layer ( Random& gen ) : base( ) {
    for ( size_t i = 0; i < N; ++i )
      for ( size_t j = 0; j < I + 1; ++j )
        __weights(i,j) = 0.5 * gen.realnegative();
  }

  // redefine compute and backprop using the member weights
  const outType& compute ( const inType& input ) {
    return this->compute_base( input, __weights );
  }

  template<class M>
  inType backprop ( outType& error, M& dw ) {
    return this->backprop_base( error, __weights, dw );
  }

  template<class M>
  neural_layer& update ( M& dw ) {
    __weights += dw;
    return *this;
  }
};

} // namespace ml

#endif // NEURAL_LAYER_HPP
