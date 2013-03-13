#ifndef ANN_HPP
#define ANN_HPP

#include "neural_layer.hpp"
#include <boost/progress.hpp>
using namespace boost;


template<size_t I, size_t H, size_t O, activation A, learning L, class T>
class ann_base {
  typedef neural_layer<H,I,SIGMOID,L,T> firstLayer;
  typedef neural_layer<O,H,A,L,T> secondLayer;
  typedef typename firstLayer::weightsType firstLayerWeights;
  typedef typename secondLayer::weightsType secondLayerWeights;

  Random __generator;
  size_t __evaluations;
  vect<firstLayer::size() + secondLayer::size(), T> __weights;
  firstLayer __first;
  secondLayer __second;

public:
  typedef vect<firstLayer::size() + secondLayer::size()> vector_type;
  typedef vect<I, T> inputType;
  typedef vect<O, T> outputType;
  typedef vect<H, T> hiddenType;
  typedef T scalar;

  ann_base ( uint32_t seed = time(0) ) :
    __generator( seed ),
    __evaluations( 0 ),
    __first( __weights.data() ),
    __second( __weights.data() + firstLayer::size() ) {

    init( __generator );
  }

  ann_base& init ( Random& another ) {
    // initialization to small, zero-centered weights
    for ( size_t i = 0; i < size(); ++i )
      __weights[i] = 0.5 * another.realnegative();

    return *this;
  }

  const outputType& compute ( const inputType& input ) {
    ++__evaluations;
    return __second.compute( __first.compute( input ) );
  }

  void backprop ( const outputType& target, firstLayerWeights& dwh, secondLayerWeights& dwo ) {
    outputType hodelta = target - __second.output();
    hiddenType ihdelta = __second.backprop( hodelta, dwo );
    __first.backprop( ihdelta, dwh, true );
  }

  template <typename InputIter, typename OutputIter>
  void train ( InputIter inputs, OutputIter targets,
               const size_t patterns, const size_t epochs ) {
    vector_type dw;
    firstLayerWeights dwh ( dw.data() );
    secondLayerWeights dwo ( dw.data() + firstLayer::size() );

    for ( size_t e = 0; e < epochs; ++e ) {
      dw = vector_type( 0 );
      for ( size_t i = 0; i < patterns; ++i ) {
        compute( inputs[i] );
        backprop( targets[i], dwh, dwo );
      }
      __weights += dw;
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
};



template<size_t I, size_t H, size_t O, activation A = SIGMOID, learning L = ONLINE, class T = double>
//using ann = ann_base<I,H,O,A,T>;
class ann : public ann_base<I,H,O,A,L,T> { };




/*template<size_t I, size_t H, size_t O, activation A, class T>
class ann<I,H,O,A,BATCH,T> : public ann_base<I,H,O,A,BATCH,T> {
  typedef ann_base<I,H,O,A,BATCH,T> base;
  vect<base::size()> __dw;
public:
  ann ( ) : base( ) {
    cout << "Using template specialization.\n";
  }
};*/

#endif // ANN_HPP
