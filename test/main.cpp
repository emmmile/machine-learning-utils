#include "random.hpp"
#include "swarm.hpp"
#include "population.hpp"
#include "ackley.hpp"
#include "ann.hpp"
#include "vect.hpp"
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
  vect<2> in  [4] = { {0,0}, {0,1}, {1,0}, {1,1} };
  vect<1> out [4] = { {0},   {1},   {1},   {0}   };

  ann<2,2,1> neural;
  neural.train( in, in + 4, out, out + 4, 200 );
  cout << "XOR Neural nework result:\n";
  for ( int i = 0; i < 4; ++i )
    cout << "  " << neural.compute( in[i] ) << "\t(" << out[i] << ")\n";

  swarm<V, dim> pso( 30, MyInit );
  pso.run( 2000, ackley<V, S, dim> );
  cout << "\nPSO result:\n  " << pso << "\n  "
       << "function evaluations:  " << pso.explored() << endl;

  population<V> ec( 100, MyInit );
  ec.run( 500, ackley<V, S, dim>, MyMutation<V,dim>( 1.0 ), MyCrossover() );
  cout << "\nEC result:\n  " << ec << "\n  "
       << "function evaluations:  " << ec.explored() << endl;

  return 0;
}
