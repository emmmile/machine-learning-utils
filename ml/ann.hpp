#ifndef ANN_HPP
#define ANN_HPP

#include "neural_layer.hpp"
#include "dataset.hpp"
#include <boost/progress.hpp>
using namespace boost;


namespace ml {

// base class, it is goog for non-pso usage (shared == NOSHARED)
template<size_t I, size_t H, size_t O, activation A = SIGMOID, learning L = ONLINE,
         class T = double, shared S = NOSHARED, bool = false>
class ann {
protected:
  typedef neural_layer<H,I,SIGMOID,L,HIDDEN,T,S> firstLayer;
  typedef neural_layer<O,H,A,L,OUTPUT,T,S> secondLayer;
  typedef typename firstLayer::weightsType firstLayerWeights;
  typedef typename secondLayer::weightsType secondLayerWeights;

  Random __generator;
  size_t __evaluations;
  firstLayer __first;
  secondLayer __second;

public:
  typedef vect<I, T> inputType;
  typedef vect<O, T> outputType;
  typedef vect<H, T> hiddenType;
  typedef T scalarType;

  ann ( uint32_t seed = Random::seed() ) :
    __generator( seed ),
    __evaluations( 0 ),
    __first( __generator ),
    __second( __generator ) {
  }

  const outputType& compute ( const inputType& input ) {
    ++__evaluations;
    return __second.compute( __first.compute( input ) );
  }

  void train_gd( const dataset<I,O,T>& set, const size_t epochs,
                 firstLayerWeights& dwh, secondLayerWeights& dwo ) {
    for ( size_t e = 0; e < epochs; ++e ) {
      for ( size_t i = 0; i < set.patterns(); ++i ) {
        compute( set.input(i) );

        outputType hodelta = set.target(i) - __second.output();
        hiddenType ihdelta = __second.backprop( hodelta, dwo );
                              __first.backprop( ihdelta, dwh );
      }

      if ( L == BATCH ) {
        __first.update( dwh );
        __second.update( dwo );
        dwh = firstLayerWeights( 0 );
        dwo = secondLayerWeights( 0 );
      }
    }
  }

  void train ( const dataset<I,O,T>& set, const size_t epochs ) {
    firstLayerWeights dwh;
    secondLayerWeights dwo;

    train_gd( set, epochs, dwh, dwo );
  }

  T error ( const dataset<I,O,T>& set ) {
    T sum = 0;
    for ( size_t i = 0; i < set.patterns(); ++i ) {
      auto diff = set.target(i) - compute( set.input(i) );
      sum += diff.squaredNorm();
    }

    return sum / set.patterns();
  }

  void results ( const dataset<I,O,T>& set, T threshold = 0.25 * 0.25 ) {
    size_t errors = 0;
    for ( size_t i = 0; i < set.patterns(); ++i ) {
      outputType out = compute( set.input(i) );
      bool diff = (set.target(i) - out).squaredNorm() > threshold;
      cout << out << "\t(" << set.target(i) << ")" << (diff ? " <-" : "") << "\n";
      if ( diff ) ++errors;
    }

    cout << "total error: " << error( set ) << endl;
    cout << "wrong patterns: " << errors << " (" << double( 100 * errors ) / set.patterns() << "%)\n";
  }

  static constexpr size_t size ( ) {
    return firstLayer::size() + secondLayer::size();
  }

  const size_t evaluations ( ) const {
    return __evaluations;
  }
};

template<size_t I, size_t H, size_t O, activation A, learning L, class T>
class ann<I, H, O, A, L, T, SHARED, true> : public ann<I, H, O, A, L, T, SHARED, false> {
public:
  ann ( uint32_t seed = Random::seed() ) :
  {

  }
// definire vettore di pesi
// ridefinire costruttuore, passando i puntatori
// ridefinire train(), passando i puntatori
};


} // namespace ml

#endif // ANN_HPP
