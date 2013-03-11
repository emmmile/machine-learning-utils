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


template<class N>
class sse {
public:
  typedef typename N::inputType inputType;
  typedef typename N::outputType outputType;

  inputType* inputs;
  outputType* targets;
  size_t patterns;

  sse( inputType* i, outputType* t, const size_t p ) :
    inputs( i ), targets( t ), patterns( p ) {
  }

  inline S operator() ( N& v ) {
    return v.error( inputs, targets, patterns );
  }

  void show ( N v ) {
    for ( size_t i = 0; i < patterns; ++i )
      cout << "  " << v.compute( inputs[i] ) << "\t(" << targets[i] << ")\n";
      cout << "  total error: " << v.error( inputs, targets, patterns ) << endl;
  }
};

// define the initialization concept as a function
template<class V>
inline void annInit( V& v, Random& gen ) {
  v.init( gen );
}


int main() {
  vect<2> in  [] = { {0,0}, {0,1}, {1,0}, {1,1} };
  vect<1> out [] = { {0},   {1},   {1},   {0}   };
  typedef ann<2,2,1> xorann;

  cout.precision( 5 );
  cout << fixed;

  xorann neural;
  neural.train( in, out, 4, 2000 );
  sse<xorann> error( in, out, 4 );
  cout << "GD, XOR neural nework training:\n";
  error.show( neural );
  cout << "  network evaluations: " << neural.evaluations() << endl;

  swarm<xorann, xorann::size(), xorann::vector_type> neural_pso( 20, annInit<xorann> );
  neural_pso.run( 500, error );
  cout << "\nPSO, XOR neural network training:\n";
  error.show( neural_pso.best() );
  cout << "  network evaluations: " << neural_pso.best().evaluations() * 20 << endl;

  swarm<V, dim> pso( 30, MyInit );
  pso.run( 2000, ackley<V, S, dim> );
  cout << "\nPSO, ackley minimization:\n  " << pso << "\n  "
       << "function evaluations:  " << pso.explored() << endl;

  population<V> ec( 100, MyInit );
  ec.run( 500, ackley<V, S, dim>, MyMutation<V,dim>( 1.0 ), MyCrossover() );
  cout << "\nEC, ackley minimization:\n  " << ec << "\n  "
       << "function evaluations:  " << ec.explored() << endl;

  return 0;
}
