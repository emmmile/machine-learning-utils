#ifndef ANN_HPP
#define ANN_HPP

#include "neural_layer.hpp"
#include <boost/progress.hpp>
using namespace boost;

template<size_t I, size_t H, size_t O, class T = double>
class ann {
  Random __generator;
  size_t __evaluations;
  vect<neural_layer<H,I,T>::size() + neural_layer<O,H,T>::size()> __weights;
  neural_layer<H,I,T> __first;
  neural_layer<O,H,T> __second;

public:
  typedef vect<neural_layer<H,I,T>::size() + neural_layer<O,H,T>::size()> vector_type;
  typedef vect<I, T> inputType;
  typedef vect<O, T> outputType;
  typedef vect<H, T> hiddenType;
  typedef T scalar;

  ann ( uint32_t seed = time(0) ) :
    __generator( seed ),
    __evaluations( 0 ),
    __first( __weights.data() ),
    __second( __weights.data() + neural_layer<H,I,T>::size() ) {

    init( __generator );
  }

  ann& init ( Random& another ) {
    // initialization to small, zero-centered weights
    for ( size_t i = 0; i < size(); ++i )
      __weights[i] = 0.5 * another.realnegative();

    return *this;
  }

  const outputType& compute ( const inputType& input ) {
    ++__evaluations;
    return __second.compute( __first.compute( input ) );
  }

  void backprop ( const outputType& target ) {
    outputType hodelta = target - __second.output();
    hiddenType ihdelta = __second.backprop( hodelta );
    __first.backprop( ihdelta, true );
  }

  template <typename InputIter, typename OutputIter>
  void train ( InputIter inputs, OutputIter targets,
               const size_t patterns, const size_t epochs ) {

    for ( size_t e = 0; e < epochs; ++e ) {
      for ( size_t i = 0; i < patterns; ++i ) {
        compute( inputs[i] );
        backprop( targets[i] );
      }

      //cout << "epoch " << e << endl;
    }
  }

  template <typename InputIter, typename OutputIter>
  T error ( InputIter inputs, OutputIter targets, const size_t patterns ) {
    T sum = 0;
    for ( size_t i = 0; i < patterns; ++i ) {
      auto diff = targets[i] - compute( inputs[i] );
      sum += diff.squaredNorm();
    }

    return sum;
  }

  static constexpr size_t size ( ) {
    return neural_layer<H,I,T>::size() + neural_layer<O,H,T>::size();
  }

  const size_t evaluations ( ) const {
    return __evaluations;
  }




  // for pso
  vector_type operator- ( const ann& another ) const {
    return this->__weights - another.__weights;
  }

  ann& operator+= ( const vector_type& dw ) {
    this->__weights += dw;
    return *this;
  }
};

#endif // ANN_HPP
