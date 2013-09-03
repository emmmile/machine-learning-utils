
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

  dataset<6,2> validation;
  train.split( validation, 50 );
  typedef ann<6,5,2,LINEAR> aa1;

  neural.train( train, 100000, validation, true );
  //neural.results( validation );

  return 0;
}
