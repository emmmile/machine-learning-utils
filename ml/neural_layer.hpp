#ifndef NEURAL_LAYER_HPP
#define NEURAL_LAYER_HPP

#include "vect.hpp"
#include <iostream>
using namespace std;
using namespace math;

template<size_t N, size_t I, class T = double>
class neural_layer {
  typedef shared_matrix<N, I + 1, T> weightsType;
  typedef vect<N, T> outType;
  typedef vect<I, T> inType;
  typedef vect<I + 1, T> richInType;

  weightsType __weights;
  outType     __output;   // last output
  richInType  __input;    // last input


  static constexpr double eta = 0.8;

  inline static double sigmoid( const double& value, const double lambda = 1.0 ) {
    return 1.0 / ( 1.0 + exp( -lambda * value ) );
  }

  // compute the component-wise delta calculation (in-place)
  inline outType& computedelta ( outType& v ) {
    for ( size_t k = 0; k < N; ++k )
      v[k] = v[k] * (1.0 - __output[k]) * __output[k];
    return v;
  }

  // the activation is component-wise too
  inline outType& activation ( const outType& net ) {
    for ( size_t i = 0; i < N; ++i ) __output[i] = sigmoid( net[i] );
    return __output;
  }

  // add the fictitious input and save it in the member variable
  const richInType& setInput( const inType& input ) {
    __input[I] = -1.0;
    copy( input.data(), input.data() + I, __input.data() );
    return __input;
  }

public:
  neural_layer ( T* data ) : __weights( data ) {
  }

  // compute the output of the network given an input
  const outType& compute ( const inType& input ) {
    // the calculation is basically a matrix-vector multiplication where
    // the input has been added a ficticious value, -1, so there are two possibilities:
    //   - copy the input vector in a bigger vector and perform the multiplication
    //   - split the multiplication as follows:
    //     out = -W0 + W*in
    // i choose to copy the input, it's O(n) instead of O(n^2)
    setInput( input );

    return activation( __weights * __input );
  }

  inType backprop ( outType& error, bool first = false ) {
    outType delta = computedelta( error );
    inType out;

    // update the weights, the delta rule
    __weights += eta * delta * __input.transpose();

    // compute the deltas for the previous layer
    if ( !first ) {
      richInType propagate = __weights.transpose() * delta;
      for ( size_t i = 0; i < I; ++i ) out[i] = propagate[i];
    }

    return out;
  }

  const weightsType& weights() const {
    return __weights;
  }

  const outType& output() const {
    return __output;
  }

  inline static constexpr size_t size ( ) {
    return N * (I + 1);
  }

  friend ostream& operator<< ( ostream & os, const neural_layer& l ) {
    //os.precision( 2 );
    return os << "weights in the layer:\n" << l.weights();
  }
};

#endif // NEURAL_LAYER_HPP
