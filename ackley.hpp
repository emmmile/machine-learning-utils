#ifndef ACKLEY_HPP
#define ACKLEY_HPP

#include "vect.hpp"
#include <math.h>

#ifdef __WIN32
template<class S>
#else
template<class S = vect<double, 2>, class CType = typename S::CType>
#endif
CType parabola ( const S& v ) {
	return v.norm();
}

#ifdef __WIN32
template<class S>
#else
template<class S = vect<double, 2>, class CType = typename S::CType, uint N = S::dims>
#endif
CType ackley ( const S& v ) {
	CType a = 20;
	CType b = 0.2;
	CType c = 2.0 * M_PI;

	CType x = 0.0;
	CType y = 0.0;

	for ( uint i = 0; i < N; ++i ) {
		x += v[i] * v[i];
		y += cos( v[i] * c );
	}

	return -a * exp( -b * sqrt( 1.0 / N * x ) ) - exp( 1.0 / N * y ) + a + M_E;
}

#endif // ACKLEY_HPP
