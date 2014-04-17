
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
  double total = 0;
  int runs = 100;

  for ( int i = 0; i < runs; ++i ) {
    dataset<6,2> train( "LOC-TR", 1 );
    train.normalize();
    dataset<6,2> validation;
    train.split( validation, 50 );
    typedef ann<6,10,2,LINEAR> aa1;




    aa1 neural;
    total += neural.train( train, 5000, validation, false );
    //neural.results( validation );
  }

  cout << total / runs << endl;
  return 0;
}
