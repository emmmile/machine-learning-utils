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

template<size_t N, size_t I, activation A, learning L, class T>
class layer_base {
protected:
  typedef shared_matrix<N, I + 1, T> weightsType;
  typedef vect<N, T> outType;
  typedef vect<I, T> inType;
  typedef vect<I + 1, T> richInType;

  weightsType __weights;
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

  layer_base ( T* data ) : __weights( data ) {
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

  friend ostream& operator<< ( ostream & os, const layer_base& l ) {
    //os.precision( 2 );
    return os << "weights in the layer:\n" << l.weights();
  }
};


// specializations for the activation function
template<size_t N, size_t I, activation A, learning L, class T>
class layer_activation : public layer_base<N,I,A,L,T> {
  typedef layer_base<N,I,A,L,T> base;
public:
  // available in gcc 4.8
  //using neural_layer_base<N,I,A,L,T>::neural_layer_base;
  layer_activation ( T* data ) : base( data ) { }
};

template<size_t N, size_t I, learning L, class T>
class layer_activation<N,I,LINEAR,L,T> : public layer_base<N,I,LINEAR,L,T> {
  typedef layer_base<N,I,LINEAR,L,T> base;
public:
  typedef typename base::outType outType;

  layer_activation ( T* data ) : base( data ) {
    //cout << "Using LINEAR specialization.\n";
    this->eta = 0.1;
  }

  inline outType& activation ( const outType& net ) {
    return this->__output = net;
  }

  // compute the component-wise delta calculation (in-place)
  inline outType computedelta ( outType& v ) {
    return v;
  }
};

template<size_t N, size_t I, learning L, class T>
class layer_activation<N,I,SIGMOID,L,T> : public layer_base<N,I,SIGMOID,L,T> {
  typedef layer_base<N,I,SIGMOID,L,T> base;

  inline static double sigmoid( const double& value, const double lambda = 1.0 ) {
    return 1.0 / ( 1.0 + exp( -lambda * value ) );
  }

public:
  typedef typename base::outType outType;

  layer_activation ( T* data ) : base( data ) {
    //cout << "Using SIGMOID specialization.\n";
    this->eta = 0.8;
  }

  inline outType& activation ( const outType& net ) {
    for ( size_t i = 0; i < N; ++i ) {
      this->__output[i] = sigmoid( net[i] );
    }

    return this->__output;
  }

  inline outType computedelta ( outType& v ) {
    for ( size_t k = 0; k < N; ++k )
        v[k] = v[k] * (1.0 - this->__output[k]) * this->__output[k];

    return v;
  }
};

// the final interface, using both specializations
template<size_t N, size_t I, activation A, learning L, class T>
class neural_layer : public layer_activation<N,I,A,L,T> {
  typedef layer_activation<N,I,A,L,T> base;
  typedef typename base::outType outType;
  typedef typename base::inType inType;
  typedef typename base::richInType richInType;
public:
  typedef typename base::weightsType weightsType;

  neural_layer ( T* data ) : base( data ) { }

    // compute the output of the network given an input
  const outType& compute ( const inType& input ) {
    // the calculation is basically a matrix-vector multiplication where
    // the input has been added a ficticious value, -1, so there are two possibilities:
    //   - copy the input vector in a bigger vector and perform the multiplication
    //   - split the multiplication as follows:
    //     out = -W0 + W*in
    // i choose to copy the input, it's O(n) instead of O(n^2)
    this->setInput( input );

    //return __output;
    return this->activation( this->__weights * this->__input );
  }

  // the back-propagation (GD) algorithm
  inType backprop ( outType& error, weightsType& dw, bool first = false ) {
    outType delta = this->computedelta( error );
    inType out;

    // compute the deltas for the previous layer
    if ( !first ) {
      richInType propagate = this->__weights.transpose() * delta;
      for ( size_t i = 0; i < I; ++i ) out[i] = propagate[i];
    }

    // update the weights, the delta rule
    dw += this->eta * delta * this->__input.transpose();

    return out;
  }
};

} // namespace ml

#endif // NEURAL_LAYER_HPP
