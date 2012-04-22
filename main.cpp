#include <algorithm>
#include <iostream>
#include <functional>
#include <vector>
#include <unistd.h>
#include <initializer_list>
#include <gmpxx.h>
#include "random.hpp"
#include "pso.hpp"
#include "ackley.hpp"
using namespace std;


#define dim 20
#define trials 20


// simple test for functions of type T (*f)( vect<T, dim> )
template<class T = double>
void testFunction ( T (*f)( const vect<T,dim>& ) ) {
	vect<T, dim> zero;
	cout << "f(0) = " << f( zero ) << endl;

	vect<T, dim> minv ( 32.0 );
	vect<T, dim> maxv = -minv;

	int counter = 0;
	T avg = 0.0;

	for ( uint i = 0; i < trials; ++i ) {
		pso<vect<T, dim>, T > solver( 30, f, minv, maxv, i );
		solver.run( 1000, 1.1, 3.3 );// 2.245, 1.925 );
		cout << solver << endl;

		if ( solver.get_best_value() < 0.1 )
			++counter;

		avg += solver.get_best_value();
		//getchar();
	}

	cout << "OVERALL PRECISION = " << counter / double( trials ) * 100.0 << "%" << endl;
	cout << "AVERAGE RESULT = " << avg / trials << endl;
}

// 30 particles, 5000 iterations, 1.1, 3.3, mutation radius = 10 / costriction, probability = 0.01
//OVERALL PRECISION = 100%
//AVERAGE RESULT = 4.03836e-05

#define test( T, dim, fun ) \
	testFunction<T> ( fun<vect<T,dim> > )

int main() {
	test( int, dim, parabola );


	return 0;
}

