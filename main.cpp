#include <algorithm>
#include <iostream>
#include <functional>
#include <vector>
#include <unistd.h>
#include <initializer_list>
#include "random.hpp"
#include "pso.hpp"
#include "ackley.hpp"
//#include "cameraraw.hpp"
using namespace std;


#define dim 20
#define trials 20



// test for solving ackley problem with many dimensions
void testAckley ( ) {
	vect<double, dim> zero;
	cout << "f(0) = " << ackley<double, dim>( zero ) << endl;

	vect<double, dim> minv ( 32.0 );
	vect<double, dim> maxv = -minv;

	int counter = 0;
	double avg = 0.0;

	for ( uint i = 0; i < trials; ++i ) {
		pso<vect<double, dim> > solver( 30, ackley<double, dim>, minv, maxv, i );
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



int main() {
	testAckley();


	return 0;
}

