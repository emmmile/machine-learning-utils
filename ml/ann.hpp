#ifndef ANN_HPP
#define ANN_HPP

#include "neural_layer.hpp"
#include "dataset.hpp"
#include <boost/progress.hpp>
using namespace boost;


namespace ml {

// The first 6 template parameters define the behaviour of the neural network.
// The last two makes possible the impossible: use this structure for PSO, i.e. using a vector
// of weights. This is really a hack because there are really many problems (concettually it
// breaks the OOP paradigm, it relies on specific vector and matrix types, with memory shared).
// But currently does his job.

// The best thing is that, if S == NOSHARED, everything is as it seems: ann class with two layers
// of weights inside.
// When S == SHARED, the specialization at the end of the file is used (notice the bool == false
// trick as last template argument, to ``avoid'' the inheritance of the base class).
// In this case a vector of weights is defined directly inside the ann class, and the two layers
// share this representation using shared matrices.

// Major tricks are present in the classes constructors and in neural_layer members definitions.

// Another point (not regarding PSO) is that I did not used explicit inheritance. The points that
// differs would be really few lines, a lot less than the code needed to actually implement
// the hierarchy. Instead I used a lot of template arguments, that are integers, and are compiled
// very very efficiently already with a -O1 flag (an "if ( 1 == 0 )" is easily removed at that
// optimization level).
template<size_t I, size_t H, size_t O, activation A = SIGMOID, learning L = ONLINE, class T = double,
         shared S = NOSHARED, bool = false>
class ann {
protected:
  typedef neural_layer<H,I,SIGMOID,L,HIDDEN,T,S> firstLayer;
  typedef neural_layer<O,H,A,L,OUTPUT,T,S> secondLayer;

  Random __generator;
  size_t __evaluations;
  firstLayer __first;
  secondLayer __second;

public:
  typedef vect<I, T> inputType;
  typedef vect<O, T> outputType;
  typedef vect<H, T> hiddenType;
  typedef T scalarType;


  // This constructor works both for the shared version and other one.
  // the type of the last two arguments is a pointer or a value depending on
  // the value of the S template parameter, so in the shared version I call
  // the layer constructors with a pointer (that will point to the memory
  // area where the weights are stored), or with a value (for an unnecessary
  // initialization).
  ann ( uint32_t seed = Random::seed(),
        typename conditional<S == SHARED, T*, T>::type fdata = 0,
        typename conditional<S == SHARED, T*, T>::type sdata = 0 ) :
    __generator( seed ),
    __evaluations( 0 ),
    __first( __generator, fdata ),
    __second( __generator, sdata ) {
  }

  const outputType& compute ( const inputType& input ) {
    ++__evaluations;
    return __second.compute( __first.compute( input ) );
  }

  void train( const dataset<I,O,T>& train, const size_t epochs,
              const dataset<I,O,T>& test = dataset<I,O,T>(), bool verbose = false ) {
    for ( size_t e = 0; e < epochs; ++e ) {
      for ( size_t i = 0; i < train.patterns(); ++i ) {
        compute( train.input(i) );

        outputType hodelta = train.target(i) - __second.output();
        hiddenType ihdelta = __second.backprop( hodelta );
                              __first.backprop( ihdelta );
      }

      if ( L == BATCH ) {
        __first.update( );
        __second.update( );
      }
      
      if ( verbose )
        cout << error( train ) << " " << error( test ) << endl;
    }
  }

  // returns the MSE, mean square error
  T error ( const dataset<I,O,T>& set ) {
    T sum = 0;
    for ( size_t i = 0; i < set.patterns(); ++i ) {
      auto diff = set.target(i) - compute( set.input(i) );
      sum += diff.squaredNorm();
    }

    return sum / set.patterns();
  }

  // prints a comparison between outputs and targets
  void results ( const dataset<I,O,T>& set ) {
    size_t errors = 0;
    for ( size_t i = 0; i < set.patterns(); ++i ) {
      outputType out = set.transform( compute( set.input(i) ) );
      outputType tar = set.transform( set.target(i) );

      bool error = (tar - out).squaredNorm() > set.threshold();
      cout << out.transpose() << "\t(" << tar.transpose() << ")" << (error ? " ### wrong" : "") << "\n";
      errors += error;
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


// This is the specialization for in-PSO usage
template<size_t I, size_t H, size_t O, activation A, learning L, class T>
class ann<I, H, O, A, L, T, SHARED, false> : public ann<I, H, O, A, L, T, SHARED, true> {
  typedef ann<I, H, O, A, L, T, SHARED, true> base;

  vect<base::size(), T> __weights;
public:
  typedef vect<base::size(), T> vector_type;

  // This is really an hack. Notice that if vector_type is non-static, this can cause
  // a nice segfault because __weights will be initialized AFTER the base class (so the memory
  // area is still uninitialized when calling ::base()).
  ann ( uint32_t seed = Random::seed() ) :
    base( seed, __weights.data(), __weights.data() + base::firstLayer::size() ) {
    // need to perform another initialization because the constructor of vector_type
    // has ben callen after the previous line, zeroing all weights.
    init( this->__generator );
  }

  // operators for PSO
  void init ( Random& gen ) {
    this->__first.init( gen );
    this->__second.init( gen );
  }

  vector_type operator- ( const ann& another ) const {
    return __weights - another.__weights;
  }

  ann& operator+= ( const vector_type& dw ) {
    __weights += dw;
    return *this;
  }
};

// Alias for that
template<size_t I, size_t H, size_t O, activation A = SIGMOID, class T = double>
using pann = class ann<I, H, O, A, ONLINE, T, SHARED, false>;

} // namespace ml

#endif // ANN_HPP
