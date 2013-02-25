#include "random.hpp"
#include "swarm.hpp"
#include "population.hpp"
#include "ackley.hpp"
using namespace std;
using namespace ml;



#define dim       20        // dimensions of the search space
typedef double S;           // type of the scalar in the search space
typedef vect<dim, S> V;     // type of the search space (vectors)

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
  population<V> ec( 100, MyInit );
  ec.run( 5000, ackley<V, S, dim>, MyMutation<V,dim>( 1.0 ), MyCrossover() );
  cout << "EC result:\n  " << ec << endl;

  // dim = 20
  // ~11.85 s
  // 3,698,894 allocs

  // dim = 100
  // ~27 s
  // ~same allocs


  // without pointers

  // dim = 20
  // ~6 s
  // 5 allocs

  // dim = 100
  // ~31 s
  // ~same allocs

  return 0;
}

