#include "random.hpp"
#include "swarm.hpp"
#include "population.hpp"
#include "ackley.hpp"
#include "ann.hpp"
#include "vect.hpp"
#include "neural_pso.hpp"
#include <boost/progress.hpp>
using namespace std;
using namespace ml;
using namespace boost;


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
  vect<2> in  [] = { {0,0}, {0,1}, {1,0}, {1,1} };
  vect<1> out [] = { {0},   {1},   {1},   {0}   };
  dataset<2,1> set( in, out, 4 );
  typedef ann<2,2,1> xorann;

  cout.precision( 5 );
  cout << fixed;

  xorann neural;
  neural.train( set, 4000 );
  cout << "GD, XOR neural nework training:\n";
  neural.results( set );
  cout << "  network evaluations: " << neural.evaluations() << endl;

  typedef pann<2,2,1> pxorann;
  /*swarm<pxorann, pxorann::size(), pxorann::vector_type> npso( 20, neural_pso_init<pxorann> );
  npso.run( 500, neural_pso<2,1,xorann>( set ) );
  cout << "\nPSO, XOR neural network training:\n";
  npso.best().results( set );
  cout << "  network evaluations: " << npso.best().evaluations() * 20 << endl;*/

  swarm<V, 20> pso( 30, MyInit );
  pso.run( 2000, ackley<V, S, dim> );
  cout << "\nPSO, ackley minimization:\n  " << pso << "\n  "
       << "function evaluations:  " << pso.explored() << endl;

  population<V> ec( 100, MyInit );
  ec.run( 500, ackley<V, S, dim>, MyMutation<V,dim>( 1.0 ), MyCrossover() );
  cout << "\nEC, ackley minimization:\n  " << ec << "\n  "
       << "function evaluations:  " << ec.explored() << endl;

  return 0;
}
