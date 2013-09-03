
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
  dataset<6,2> train( "LOC-TR", 1 );
  train.normalize();

  dataset<6,2> validation;
  train.split( validation, 5 );
  typedef ann<6,20,2,LINEAR> aa1;

  //progress_timer timer;

  aa1 neural;
  neural.train( train, 100000, validation, true );
  //neural.results( train );
  //neural.results( validation );

  return 0;
}
