
#include <iostream>
#include <boost/progress.hpp>
#include "ann.hpp"
#include "vect.hpp"
#include "swarm.hpp"
#include "dataset.hpp"
#include "neural_pso.hpp"
using namespace std;
using namespace boost;
using namespace ml;

// http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/


int main ( ) {
  dataset<17,1> train( "monks-3.train" );
  dataset<17,1> test( "monks-3.test" );
  typedef ann<17,4,1> monks;

  cout.precision( 5 );
  cout << fixed;

  progress_timer t;
  monks neural;
  neural.train( train, 200 );
  neural.results( train );
  getchar();
  neural.results( test );


  /*swarm<monks, monks::size(), monks::vector_type> npso( 20, neural_pso_init<monks> );
  npso.run( 1000, neural_pso<6,1,monks>( train ) );
  npso.best().results( train );
  getchar();
  npso.best().results( test );*/

  return 0;
}
