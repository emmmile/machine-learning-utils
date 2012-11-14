#include <algorithm>
#include <iostream>
#include <functional>
#include <vector>
#include <unistd.h>
#include <initializer_list>
//#include <gmpxx.h>
#include "random.hpp"
#include "swarm.hpp"
#include "population.hpp"
#include "ackley.hpp"
using namespace std;
using namespace ml;

#define dim 20
#define trials 20
#define iterations 1000


template<class T, uint N>
struct MyInit {
  inline void operator() ( T& p, Random& gen ) {
    for ( uint i = 0; i < N; ++i )
      p[i] = 32 * gen.real();
  }
};

template<class T, uint N>
struct MyMutation {
  //double step;
  //MyMutation( double step ) : step( step ) {}
  inline void operator() ( T& p, Random& gen ) {
    double step = 1.0;
    p[gen.integer() % N] += gen.realnegative() * step;
  }
};

template<class T, uint N>
struct MyCrossover {
  inline void operator() ( T& a, T& b, Random& gen ) {
    uint pos = gen.integer() % a.dims;
    for ( uint i = pos; i < N; ++i ) swap( a[i], b[i] );
  }
};




typedef double S;           // type of the scalar in the search space
typedef vect<S, dim> V;     // type of the search space
typedef MyInit<V, dim> I;   // initializer type
typedef MyMutation<V, dim> M; // mutation type
typedef MyCrossover<V, dim> C; // crossover type

template<typename F>
void testSwarm ( F function ) {
  V zero;
  S avg = 0.0;
  int counter = 0;
  cout << "f(0) = " << function( zero ) << endl;

  for ( uint i = 0; i < trials; ++i ) {
    swarm<V, dim> solver( 30, i, I() );
    solver.run( iterations, function, 1.1, 3.3 );// 2.245, 1.925 );
    cout << solver << endl;

    if ( solver.get_best_value() < 0.2 )
            ++counter;

    avg += solver.get_best_value();
  }

  cout << "OVERALL PRECISION = " << counter / double( trials ) * 100.0 << "%" << endl;
  cout << "AVERAGE RESULT = " << avg / trials << endl;
}

// 30 particles, 5000 iterations, 1.1, 3.3, mutation radius = 10 / costriction, probability = 0.01
//OVERALL PRECISION = 100%
//AVERAGE RESULT = 4.03836e-05


template<typename F>
void testPopulation ( F function ) {
  V zero;
  S avg = 0.0;
  int counter = 0;
  cout << "f(0) = " << function( zero ) << endl;

  for ( uint i = 0; i < trials; ++i ) {
    population<V> solver( 100, i, I() );
    solver.run( iterations, function, M(), C() );
    cout << solver << endl;

    if ( solver.get_best_value() < 0.2 )
            ++counter;

    avg += solver.get_best_value();
  }

  cout << "OVERALL PRECISION = " << counter / double( trials ) * 100.0 << "%" << endl;
  cout << "AVERAGE RESULT = " << avg / trials << endl;
}



int main() {
  testSwarm( ackley<V, S, dim> );
  testPopulation( ackley<V, S, dim> );

  return 0;
}

