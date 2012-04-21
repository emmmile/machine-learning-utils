#ifndef ACKLEY_HPP
#define ACKLEY_HPP

#include "vect.hpp"
#include <math.h>

#ifdef __WIN32
template<class T, uint N>
#else
template<class T = double, uint N = 2>
#endif
T parabola ( const vect<T, N>& v ) {
    return v.norm();
}

#ifdef __WIN32
template<class T, uint N>
#else
template<class T = double, uint N = 2>
#endif
T ackley ( const vect<T, N>& v ) {
    T a = 20;
    T b = 0.2;
    T c = 2.0 * M_PI;

    T x = 0.0;
    T y = 0.0;

    for ( uint i = 0; i < N; ++i ) {
        x += v[i] * v[i];
        y += cos( v[i] * c );
    }

    return -a * exp( -b * sqrt( 1.0 / N * x ) ) - exp( 1.0 / N * y ) + a + M_E;
}

#endif // ACKLEY_HPP
