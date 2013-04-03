
#include <iostream>
#include <boost/progress.hpp>
#include "ann.hpp"
#include "vect.hpp"
#include "swarm.hpp"
#include "dataset.hpp"
#include "boost/progress.hpp"
#include "neural_pso.hpp"
using namespace std;
using namespace boost;
using namespace ml;


int main ( ) {
  cout << "ciao" << endl;
  dataset<6,2> train( "LOC-TR", 1 );
  typedef ann<6,100,2,LINEAR> aa1;

  progress_timer timer;
  train.normalize();

  aa1 neural;
  neural.train( train, 10000 );
  neural.results( train );

  return 0;
}
