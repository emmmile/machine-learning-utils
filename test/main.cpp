#include "random.hpp"
#include "swarm.hpp"
#include "population.hpp"
#include "ackley.hpp"
using namespace std;
using namespace ml;

template <class T = double, uint N = 2>
using vect = matrix<N, 1, T>;

#define dim       20        // dimensions of the search space
typedef double S;           // type of the scalar in the search space
typedef vect<S, dim> V;     // type of the search space (vectors)

// define the initialization concept as a function
inline void MyInit( V& v, Random& gen ) {
  for ( uint i = 0; i < dim; ++i )
    v[i] = 32 * gen.real();
}

// define the crossover operator as a functor
struct MyCrossover {
  inline void operator() ( V& a, V& b, Random& gen ) {
    uint pos = gen.integer() % dim;
    for ( uint i = pos; i < dim; ++i ) swap( a[i], b[i] );
  }
};

// define the mutation as template function object with state,
// the most general case.
template<class T, uint N>
struct MyMutation {
  double step;
  MyMutation( double step ) : step( step ) {}
  inline void operator() ( T& p, Random& gen ) {
    p[gen.integer() % N] += gen.realnegative() * step;
  }
};

int main() {
  vect<double, 2> t = { 2, 1 };
  cout << t << endl;

  swarm<V, dim> pso( 30, MyInit );
  pso.run( 2000, ackley<V, S, dim> );
  cout << "PSO result:\n  " << pso << "\n  "
       << "function evaluations:  " << pso.explored() << endl;

  population<V> ec( 100, MyInit );
  ec.run( 500, ackley<V, S, dim>, MyMutation<V,dim>( 1.0 ), MyCrossover() );
  cout << "EC result:\n  " << ec << "\n  "
       << "function evaluations:  " << ec.explored() << endl;

  return 0;
}
