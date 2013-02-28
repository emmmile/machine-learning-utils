#ifndef VECT_HPP
#define VECT_HPP

#include <iostream>
#include <initializer_list>
#include <array>
#include "random.hpp"
using namespace std;


template<size_t N, size_t M, class T>
class matrix_base {
protected:
  array<T, N * M> values;

public:
  // initializes the vector in the origin
  matrix_base ( const T value = 0 ) {
    fill( values.begin(), values.end(), value );
  }

  // copy constructor
  matrix_base ( const matrix_base& v ) : values( v.values ) {
  }


  // initializes from a list (only works with c++0x)
  matrix_base ( std::initializer_list<T> v ) {
    //assert( v.size() == sizeof( values ) / sizeof( T ) );
    copy( v.begin(), v.end(), values.begin() );
  }

  // returns a reference to the i-th component. I avoid checks for more speed
  T& operator[] ( size_t i ) {
    return values[i];
  }

  T operator[] ( size_t i ) const {
    return values[i];
  }

  T& operator() ( size_t i, size_t j ) {
    return values[M * i + j];
  }

  T operator() ( size_t i, size_t j ) const {
    return values[M * i + j];
  }

  // assign v to *this
  matrix_base& operator= ( const matrix_base& v ) {
    values = v.values;
    return *this;
  }

  bool operator== ( const matrix_base& v ) {
    return values != v.values;
  }

  // subtract v from *this
  matrix_base& operator-= ( const matrix_base& v ) {
    for ( size_t i = 0; i < N * M; ++i )
      values[i] -= v.values[i];
    return *this;
  }

  // returns the difference vector (*this - v)
  matrix_base operator- ( const matrix_base& v ) const {
    return matrix_base(*this) -= v;
  }

  // unary - (changes sign to the components)
  matrix_base operator- ( ) const {
    return matrix_base() -= (*this);
  }

  matrix_base& operator+= ( const matrix_base& v ) {
    for ( size_t i = 0; i < N * M; ++i )
      values[i] += v.values[i];
    return *this;
  }

  matrix_base operator+ ( const matrix_base& v ) const {
    return matrix_base(*this) += v;
  }

  // TODO hermitian conjugate?
  matrix_base<M,N,T> transpose ( ) const {
    matrix_base<M,N,T> out;
    for ( size_t i = 0; i < N; ++i )
      for ( size_t j = 0; j < M; ++j )
        out(j,i) = (*this)(i,j);

    return out;
  }

  // row-column matrix multiplication
  template<size_t S>
  matrix_base<N,S,T> operator* ( const matrix_base<M,S,T>& v ) const {
    matrix_base<N,S,T> out;
    for ( size_t i = 0; i < N; ++i )
      for ( size_t j = 0; j < S; ++j )
        for ( size_t k = 0; k < M; ++k )
          out(i,j) += (*this)(i,k) * v(k,j);

    return out;
  }

  // right multiplication by a scalar
  matrix_base& operator*= ( const T& s ) {
    for ( size_t i = 0; i < N * M; ++i )
      values[i] *= s;
    return *this;
  }

  // left multiplication by a scalar
  friend matrix_base operator* ( T s, const matrix_base& v ) {
    return matrix_base(v) *= s;
  }

  // scalar conversion, in case of 1x1 matrices, for example if I want the inner product
  //operator T ( ) const {
  T scalar ( ) const {
	  static_assert( N == 1 && M == 1, "scalar conversion on a matrix bigger than 1x1" );
    return values[0];
  }

  friend ostream& operator<< ( ostream & os, const matrix_base& v ) {
    //os.precision( 2 );
    os << "[";

    for ( size_t i = 0; i < N; ++i ) {
      if ( i > 0 ) os << endl << " ";

      size_t j = 0;
      for ( j = 0; j < M-1; ++j )
        os << v(i,j) << ", ";

      os << v(i,j);
    }

    return os << "]";
  }
};


// I love C++11
template<size_t N, size_t M, class T = double>
using matrix = matrix_base<N, M, T>;

template <size_t N, class T = double>
using vect = matrix_base<N, 1, T>;


#endif // VECT_HPP
