#ifndef VECT_HPP
#define VECT_HPP

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <initializer_list>
using namespace std;

#ifdef _WIN32
typedef unsigned int uint;
#endif





template<class T = double, uint N = 2>
class vect {
	T values [N];

public:
	typedef T CType;		// component type
	static const uint dims = N;	// vector size (number of dimensions)

	// initializes the vector in the origin
	vect ( const T value = 0 ) {
		fill( values, values + N, value );
	}

	// initializes vector with random components uniformly drawn from a specific range
	vect ( const vect& minvalues, const vect& maxvalues, Random& gen ) {
		for ( uint i = 0; i < N; ++i )
			values[i] = ( maxvalues[i] - minvalues[i] ) * gen.real() + minvalues[i];
	}

	// copy constructor
	vect ( const vect& v ) {
		copy( v.values, v.values + N, values );
	}

	// initializes from a list (only works with c++0x)
	vect ( std::initializer_list<T> v ) {
		assert( v.size() == sizeof( values ) / sizeof( T ) );
		copy( v.begin(), v.end(), values );
	}

	// returns a reference to the i-th component. I avoid checks for more speed
	T& operator[] ( uint i ) {
		return values[i];
	}

	T operator[] ( uint i ) const {
		return values[i];
	}

	// assign v to *this
	vect& operator= ( const vect& v ) {
		copy( v.values, v.values + N, values );
		return *this;
	}

	// subtract v from *this
	vect& operator-= ( const vect& v ) {
		for ( uint i = 0; i < N; ++i )
			values[i] -= v.values[i];
		return *this;
	}

	// returns the difference vector (*this - v)
	vect operator- ( const vect& v ) const {
		return vect(*this) -= v;
	}

	// unary - (changes sign to the components)
	vect operator- ( ) const {
		return vect() -= (*this);
	}

	vect& operator+= ( const vect& v ) {
		for ( uint i = 0; i < N; ++i )
			values[i] += v.values[i];
		return *this;
	}

	vect operator+ ( const vect& v ) const {
		return vect(*this) += v;
	}

	// euclidean norm
	T norm ( ) const {
		/*T out = 0.0;
		for ( uint i = 0; i < N; ++i )
			out += values[i] * values[i];*/

		return sqrt( (*this) * (*this) );
	}

	vect& abs ( ) {
		for ( uint i = 0; i < N; ++i )
			values[i] = fabs( values[i] );
		return *this;
	}

	// dot product of two vectors
	T operator* ( const vect& v ) const {
		//return inner_product( values, values + N, v.values, 0 );
		T out = 0;
		for ( uint i = 0; i < N; ++i )
			out += values[i] * v[i];

		return out;
	}

	// multiplication by a scalar
	vect& operator*= ( const T& s ) {
		for ( uint i = 0; i < N; ++i )
			values[i] *= s;
		return *this;
	}

	// these functions are specifically meant for PSO. Instead of scaling the vector
	// by the scalar phi, it makes a random scaling
	vect random_stretch ( const T& phi, Random& gen ) {
		//vect out( *this );
		for ( uint i = 0; i < N; ++i )
			values[i] *= phi * gen.real();

		return *this;
	}

	// makes a little mutation to one component of the vector, like in genetic algorithms
	vect& random_mutate( const T& step, Random& gen, const T& p = 0.01 ) {
		// XXX an exponentially decreasing function (of the distance) could be used, but is slower!
		//if ( gen.real() < 0.01 * exp( -social.norm() ) ) {
		if ( gen.real() < p )
			values[gen.integer() % N] += gen.realnegative() * step;

		return *this;
	}

	friend vect operator* ( T s, const vect& v ) {
		return vect(v) *= s;
	}

	friend ostream& operator<< ( ostream & os, const vect& v ) {
		os << "[";
		//os.precision( 2 );

		for ( uint i = 0; i < N - 1; ++i )
			os << v.values[i] << ", ";

		return os << v.values[N-1] << "]";
	}
};

#endif // VECT_HPP
