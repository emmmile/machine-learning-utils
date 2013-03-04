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
  outType __output;  // last output
  richInType __input; // last input


  static constexpr double eta = 1.0;

  inline static double sigmoid( const double& value, const double lambda = 1.0 ) {
      double k = exp( lambda * value );
      return k / ( 1.0 + k );
  }

  inline outType computedelta ( const outType& v ) {
    outType d;
    for ( size_t k = 0; k < N; ++k )
      d[k] = v[k] * (1.0 - __output[k]) * __output[k];
    return d;
  }

  inline outType& activation ( const outType& net ) {
    for ( size_t i = 0; i < N; ++i ) __output[i] = sigmoid( net[i] );
    return __output;
  }

public:
  neural_layer ( T* data, Random& gen ) : __weights( data ) {
    // initialization to small, zero-centered weights
    for ( size_t i = 0; i < N; ++i )
      for ( size_t j = 0; j < I + 1; ++j )
        __weights(i,j) = 0.1 * gen.realnegative();
  }

  inline static constexpr size_t size ( ) {
    return N * (I + 1);
  }

  inline static richInType addFictitiusInput( const inType& input ) {
    richInType richinput;
    richinput[I] = -1.0;
    copy( input.data(), input.data() + I, richinput.data() );
    return richinput;
  }

  // compute the output of the network given an input
  const outType& compute ( const inType& input ) {
    // the calculation is basically a matrix-vector multiplication where
    // the input has been added a ficticious value, -1, so there are two possibilities:
    //   - copy the input vector in a bigger vector and perform the multiplication
    //   - split the multiplication as follows:
    //     out = -W0 + W*in
    // i choose to copy the input, it's O(n) instead of O(n^2)
    __input = addFictitiusInput( input );

    return activation( __weights * __input );
  }

  inType backprop ( const outType& error, bool first = false ) {
    outType delta = computedelta( error );
    inType out;

    // update the weights, the delta rule
    cout << "error = " << error << endl;
    cout << "delta = " << delta << endl;
    __weights += eta * delta * __input.transpose();
    cout << __weights << endl;
    getchar();

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

  friend ostream& operator<< ( ostream & os, const neural_layer& l ) {
    //os.precision( 2 );
    return os << "weights in the layer:\n" << l.weights();
  }
};

#endif // NEURAL_LAYER_HPP
