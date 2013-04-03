#ifndef DATASET_HPP
#define DATASET_HPP

#include <string>
#include <iostream>
#include <vector>
#include "vect.hpp"
#include <fstream>
using namespace std;
using namespace math;

template<size_t I, size_t O, class T = double>
class dataset {
  typedef vect<I, T> inputType;
  typedef vect<O, T> outputType;

  vector<inputType> __inputs;
  vector<outputType> __targets;
  T __threshold;
  matrix<I,I,T> __iscale;
  matrix<O,O,T> __tscale; // affine transformations for normalization etc.
  inputType __itrasl;
  outputType __ttransl;

  template <typename Container>
  const void stats ( Container& v, size_t j, T& min, T& max ) const {
    assert( j <= I );

    min = max = v[0][j];
    for ( size_t i = 0; i < v.size(); ++i ) {
      if ( min > v[i][j] ) min = v[i][j];
      if ( max < v[i][j] ) max = v[i][j];
    }

    //cout << "min and max on dimension " << j << " = " << min << " " << max << endl;
  }

  // common code between all constructors
  void init ( ) {
    //__inputs.reserve( 1000 );
    //__targets.reserve( 1000 );
    for ( size_t i = 0; i < O; ++i ) __tscale(i,i) = 1;
    for ( size_t i = 0; i < I; ++i ) __iscale(i,i) = 1;
  }

public:
  // load a dataset from a file (every row must have exactly I + O fields)
  dataset ( string filename, T t = 0.25 ) : __threshold( t ) {
    init();
    ifstream file( filename, ifstream::in );
    inputType itmp;
    outputType otmp;

    while ( file.good() ) {
			size_t read = 0;	
      for ( size_t i = 0; i < O && file.good(); ++i, ++read ) file >> otmp[i];
      for ( size_t i = 0; i < I && file.good(); ++i, ++read ) file >> itmp[i];

      if ( read != O + I ) break;
      __inputs.push_back( itmp );
      __targets.push_back( otmp );
    }
  }

  // load a dataset from an existing sequence
  template <typename InputIter, typename OutputIter>
  dataset ( InputIter in, OutputIter out, size_t patterns, T t = 0.25 ) : __threshold( t ) {
    init();
    for ( size_t i = 0; i < patterns; ++i, ++in, ++out ) {
      __inputs.push_back( *in );
      __targets.push_back( *out );
    }
  }

  size_t patterns ( ) const {
    return __inputs.size();
  }

  const inputType& input ( size_t i ) const {
    return __inputs[i];
  }

  const T& threshold ( ) const {
    return __threshold;
  }

  const outputType& target ( size_t i ) const {
    return __targets[i];
  }

  dataset& normalize ( ) {
    T min, max;
    for ( size_t j = 0; j < I; ++j ) {
      stats( __inputs, j, min, max );
      __itrasl[j] = ( max + min ) * 0.5;
      __iscale(j,j) = 2 / ( max - min ); // [-1.0, 1.0]
    }

    for ( size_t j = 0; j < O; ++j ) {
      stats( __targets, j, min, max );
      __ttransl[j] = ( max + min ) * 0.5;
      __tscale(j,j) = 2 / ( max - min ); // [-1.0, 1.0]
    }

    for ( size_t i = 0; i < patterns(); ++i ) {
      __inputs[i] = __iscale * ( __inputs[i] - __itrasl );
      __targets[i] = __tscale * ( __targets[i] - __ttransl );
    }

    return *this;
  }

  outputType transform ( const outputType& v ) const {
    return __tscale.inverse() * v + __ttransl;
  }

  friend ostream& operator<< ( ostream & os, const dataset& d ) {
    for ( size_t i = 0; i < d.patterns(); ++i )
      os << d.__inputs[i].transpose() << " | " << d.__targets[i].transpose() << endl;
    return os;
  }
};

#endif // DATASET_HPP
