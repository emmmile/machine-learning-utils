#ifndef ANN_HPP
#define ANN_HPP

#include "neural_layer.hpp"
#include "dataset.hpp"
#include <boost/progress.hpp>
using namespace boost;


namespace ml {

template<size_t I, size_t H, size_t O, activation A, learning L, class T>
class ann_base {
protected:
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
	typedef T scalarType;

	ann_base ( uint32_t seed = Random::seed() ) :
    __generator( seed ),
    __evaluations( 0 ),
    __first( __weights.data() ),
    __second( __weights.data() + firstLayer::size() ) {
		//cout << seed << endl;
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

  void train ( const dataset<I,O,T>& set, const size_t epochs ) {
    vector_type dw;
    firstLayerWeights dwh ( dw.data() );
    secondLayerWeights dwo ( dw.data() + firstLayer::size() );

    for ( size_t e = 0; e < epochs; ++e ) {
      for ( size_t i = 0; i < set.patterns(); ++i ) {
        dw = vector_type( 0 );
        compute( set.input(i) );
        backprop( set.target(i), dwh, dwo );
        __weights += dw;
      }
    }
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

	const vector_type& weights ( ) const {
		return __weights;
	}

	// for pso
	vector_type operator- ( const ann_base& another ) const {
		return this->__weights - another.__weights;
	}

	ann_base& operator+= ( const vector_type& dw ) {
		this->__weights += dw;
		return *this;
	}
};



template<size_t I, size_t H, size_t O, activation A = SIGMOID, learning L = ONLINE, class T = double>
//using ann = ann_base<I,H,O,A,T>;
class ann : public ann_base<I,H,O,A,L,T> { };




template<size_t I, size_t H, size_t O, activation A, class T>
class ann<I,H,O,A,BATCH,T> : public ann_base<I,H,O,A,BATCH,T> {
	typedef ann_base<I,H,O,A,BATCH,T> base;
public:
	ann ( uint32_t seed = Random::seed() ) : base( seed ) { }

	void train ( const dataset<I,O,T>& set, const size_t epochs ) {
		cout << "Using BATCH specialization.\n";
		typename base::vector_type dw;
		typename base::firstLayerWeights dwh ( dw.data() );
		typename base::secondLayerWeights dwo ( dw.data() + base::firstLayer::size() );

		for ( size_t e = 0; e < epochs; ++e ) {
			dw = typename base::vector_type( 0 );
			for ( size_t i = 0; i < set.patterns(); ++i ) {
				this->compute( set.input(i) );
				this->backprop( set.target(i), dwh, dwo );
			}
			this->__weights += dw;
		}
	}
};

} // namespace ml

#endif // ANN_HPP
