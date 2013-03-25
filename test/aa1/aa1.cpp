
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


int main ( ) {
  dataset<6,2> train( "LOC-TR", 1 );
  //dataset<6,2> test( "LOC-TS" );
  typedef ann<6,20,2,LINEAR> aa1;

  train.normalize();
  //test.normalize();
  //test = train.split( 272 );
  cout << train << endl;



  aa1 neural;
  neural.train( train, 20000 );
  neural.results( train );
  /*strain.normalize();

  neural_pso<6,2,aa1> aux( train ); // this is needed for the initialization and the fitness
  swarm<aa1, aa1::size(), aa1::vector_type> neural( 20, aux );
  neural.run( 1000, aux );

  neural.best().results( train );*/

  return 0;
}
