#ifndef VECT_HPP
#define VECT_HPP

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <initializer_list>
#include <valarray>
#include <array>
#include <forward_list>
#include "random.hpp"
using namespace std;

#ifdef _WIN32
typedef unsigned int uint;
#endif

// considerare std::valarray per N grande
// ridefinire ordine template
// considerare std::array



template<uint N, uint M, class T = double>
class matrix {
protected:
  array<T, N * M> values;

public:
  // initializes the vector in the origin
  matrix ( const T value = 0 ) {
    fill( values.begin(), values.end(), value );
  }

  // copy constructor
  matrix ( const matrix& v ) {
    //copy( v.values.begin(), v.values.end(), values.begin() );
    values = v.values;
  }


  // initializes from a list (only works with c++0x)
  matrix ( std::initializer_list<T> v ) {
    //assert( v.size() == sizeof( values ) / sizeof( T ) );
    copy( v.begin(), v.end(), values.begin() );
  }

  // this should be better than the previous
  //template<typename ...E>
  //vect(E... &&e) : values{{ std::forward<E>(e)... }} {}

  // returns a reference to the i-th component. I avoid checks for more speed
  T& operator[] ( uint i ) {
    return this->values[i];
  }

  T operator[] ( uint i ) const {
    return this->values[i];
  }

  // assign v to *this
  matrix& operator= ( const matrix& v ) {
    //copy( v.values.begin(), v.values.end(), values );
    values = v.values;
    return *this;
  }

  bool operator== ( const matrix& v ) {
    for ( uint i = 0; i < N; ++i )
      if ( values[i] != v.values[i] )
        return false;
    return true;
  }

  // subtract v from *this
  matrix& operator-= ( const matrix& v ) {
    for ( uint i = 0; i < N; ++i )
      values[i] -= v.values[i];
    return *this;
  }

  // returns the difference vector (*this - v)
  matrix operator- ( const matrix& v ) const {
    return matrix(*this) -= v;
  }

  // unary - (changes sign to the components)
  matrix operator- ( ) const {
    return matrix() -= (*this);
  }

  matrix& operator+= ( const matrix& v ) {
    for ( uint i = 0; i < N; ++i )
      values[i] += v.values[i];
    return *this;
  }

  matrix operator+ ( const matrix& v ) const {
    return matrix(*this) += v;
  }

  // euclidean norm
  T norm ( ) const {
    /*T out = 0.0;
    for ( uint i = 0; i < N; ++i )
      out += values[i] * values[i];*/

    return sqrt( (*this) * (*this) );
  }

  matrix& abs ( ) {
    for ( uint i = 0; i < N; ++i )
      values[i] = fabs( values[i] );
    return *this;
  }

  // dot product of two vectors
  T operator* ( const matrix& v ) const {
    //return inner_product( values, values + N, v.values, 0 );
    T out = 0;
    for ( uint i = 0; i < N; ++i )
      out += values[i] * v[i];

    return out;
  }

  // multiplication by a scalar
  matrix& operator*= ( const T& s ) {
    for ( uint i = 0; i < N; ++i )
      values[i] *= s;
    return *this;
  }

  friend matrix operator* ( T s, const matrix& v ) {
    return matrix(v) *= s;
  }

  friend ostream& operator<< ( ostream & os, const matrix& v ) {
    os << "[";
    //os.precision( 2 );

    for ( uint i = 0; i < N - 1; ++i )
      os << v.values[i] << ", ";

    return os << v.values[N-1] << "]";
  }
};



/*template<, uint N = 2>
class vect : public matrix<N, 1, T> {
public:

};*/

#endif // VECT_HPP
