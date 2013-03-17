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
public:
  // load a dataset from a file (every row must have exactly I + O fields)
  dataset ( string filename, T t = 0.25 ) : __threshold( t ) {
    ifstream file( filename, ifstream::in );
    inputType itmp;
    outputType otmp;

    while ( file.good() ) {
      for ( size_t i = 0; i < O; ++i ) file >> otmp[i];
      for ( size_t i = 0; i < I; ++i ) file >> itmp[i];

      if ( !file.good() ) break;
      __inputs.push_back( itmp );
      __targets.push_back( otmp );
    }
  }

  // load a dataset from an existing sequence
  template <typename InputIter, typename OutputIter>
  dataset ( InputIter in, OutputIter out, size_t patterns, T t = 0.25 ) : __threshold( t ) {
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

  friend ostream& operator<< ( ostream & os, const dataset& d ) {
    //os << d.inputs.size();
    for ( size_t i = 0; i < d.__inputs.size(); ++i )
      os << d.__inputs[i].transpose() << " " << d.__targets[i].transpose() << endl;
    return os;
  }
};

#endif // DATASET_HPP
