
#include <iostream>
#include <boost/progress.hpp>
#include "ann.hpp"
#include "vect.hpp"
#include "dataset.hpp"
using namespace std;
using namespace boost;
using namespace ml;


int main ( ) {
  dataset<6,1> train( "monks-1.train" );
  dataset<6,1> test( "monks-1.test" );
  typedef ann<6,4,1> monks;

  cout.precision( 5 );
  cout << fixed;

  monks neural;
  neural.train( train, 20000 );
  neural.results( train );
  getchar();
  neural.results( test );

  return 0;
}
