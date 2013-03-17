
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

  monks neural;
  neural.train( train, 300 );
  neural.results( train );
  getchar();
  neural.results( test );

  return 0;
}
