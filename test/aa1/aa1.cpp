
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
  typedef ann<6,32,2,LINEAR> aa1;

  //progress_timer timer;
  train.normalize();
  dataset<6,2> test( train, 30 );
  //cout << test << endl;


  aa1 neural;
  neural.train( train, test, 120 );
  neural.results( test );

  return 0;
}
