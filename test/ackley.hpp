#ifndef ACKLEY_HPP
#define ACKLEY_HPP

#include "vect.hpp"
#include <math.h>

#ifdef __WIN32
template<class S>
#else
template<class V, class S>
#endif
S parabola ( const V& v ) {
	return v.norm();
}

#ifdef __WIN32
template<class S>
#else
template<class V, class S, uint N>
#endif
S ackley ( const V& v ) {
	S a = 20;
	S b = 0.2;
	S c = 2.0 * M_PI;

	S x = 0.0;
	S y = 0.0;

	for ( uint i = 0; i < N; ++i ) {
		x += v[i] * v[i];
		y += cos( v[i] * c );
	}

	return -a * exp( -b * sqrt( 1.0 / N * x ) ) - exp( 1.0 / N * y ) + a + M_E;
}

#endif // ACKLEY_HPP
