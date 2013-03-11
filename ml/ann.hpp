#ifndef ANN_HPP
#define ANN_HPP

#include "neural_layer.hpp"
#include <boost/progress.hpp>
using namespace boost;


template<size_t I, size_t H, size_t O, activation A = SIGMOID, class T = double>
class ann {
  typedef neural_layer<H,I,SIGMOID,T> firstLayer;
  typedef neural_layer<O,H,A,T> secondLayer;

  Random __generator;
  size_t __evaluations;
  vect<firstLayer::size() + secondLayer::size()> __weights;
  firstLayer __first;
  secondLayer __second;

public:
  typedef vect<firstLayer::size() + secondLayer::size()> vector_type;
  typedef vect<I, T> inputType;
  typedef vect<O, T> outputType;
  typedef vect<H, T> hiddenType;
  typedef T scalar;

  ann ( uint32_t seed = time(0) ) :
    __generator( seed ),
    __evaluations( 0 ),
    __first( __weights.data() ),
    __second( __weights.data() + firstLayer::size() ) {

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
      //cout << error( inputs, targets, patterns ) << endl;

      for ( size_t i = 0; i < patterns; ++i ) {
        compute( inputs[i] );
        backprop( targets[i] );
      }

      //cout << error( inputs, targets, patterns ) << endl;
      //getchar();
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
    return firstLayer::size() + secondLayer::size();
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
