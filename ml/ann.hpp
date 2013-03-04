#ifndef ANN_HPP
#define ANN_HPP

#include "neural_layer.hpp"
#include <boost/progress.hpp>
using namespace boost;

template<size_t I, size_t H, size_t O, class T = double>
class ann {
  Random gen;
  vect<neural_layer<H,I,T>::size() + neural_layer<O,H,T>::size()> weights;
  neural_layer<H,I,T> first;
  neural_layer<O,H,T> second;

  typedef vect<I, T> inputType;
  typedef vect<O, T> outputType;
  typedef vect<H, T> hiddenType;

public:
  ann ( uint32_t seed = 123456789 ) :
    gen( seed ),
    first( weights.data(), gen ),
    second( weights.data() + neural_layer<H,I,T>::size(), gen ) {
  }

  const outputType& compute ( const inputType& input ) {
    return second.compute( first.compute( input ) );
  }

  void backprop ( const outputType& target ) {
    hiddenType delta = second.backprop( target - second.output() );
    first.backprop( delta, true );
  }

  template <typename InputIter, typename OutputIter>
  void train ( InputIter ibeg, InputIter iend, OutputIter tbeg, OutputIter tend,
               const size_t epochs ) {
    assert( (iend - ibeg ) == (tend - tbeg) );

    for ( size_t e = 0; e < epochs; ++e ) {
      OutputIter j = tbeg;

      for ( InputIter i = ibeg; i != iend; ++i, ++j ) {
        compute( *i );
        backprop( *j );
      }

      //cout << "epoch " << e << endl;
    }
  }
};

#endif // ANN_HPP
