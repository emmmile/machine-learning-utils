#ifndef POPULATION_HPP
#define POPULATION_HPP
#include <type_traits>
#include "concepts.hpp"
#include "random.hpp"
using namespace std;
namespace ml {


// default Generation function caller, it does nothing!
template<class I>
struct Generation {
  inline void operator() (I&, uint ) {
  }
};



template<class I,
         class ValueType = double>
class population {
  class triple {
  public:
    I* individual;
    bool changed;
    ValueType fitness;

    triple( ) : changed( true ), fitness( 0.0 ) { }
    triple( I* i, bool b, ValueType d = 0.0 ) : individual(i), changed(b), fitness(d) { }
    inline bool operator< (const triple& another ) const {
      return fitness < another.fitness;
    }
  };

  vector<triple> individuals;
  Random gen;
  double pmutation;
  double pcrossover;
  bool stationary;
  unsigned int age;
  unsigned int explored;

  const static uint maximumPopulation = 400;
  const static uint initialPopulation = 100;

  void erase(const uint i) {
    // remove the i-th individual from the population
    assert(i < individuals.size());
    delete individuals[i].individual;
    // here is better to first move the empty slot at the end
    swap( individuals[i], individuals.back() );
    // and then remove from the end, since we have O(1) there (and no constraints on the order)
    individuals.pop_back();
  }

public:

  template<typename Init>
  explicit population( uint size,
                       uint seed = 12345678,
                       Init init = Init(),
                       bool st = true,
                       double pc = 0.9,
                       double pm = 0.05 )
    : individuals( size ), gen( seed ), pmutation( pm ),
      pcrossover( pc ), stationary( st ), age( 0 ), explored( size )
  {
    for (uint i = 0; i < individuals.size(); ++i) {
      individuals[i].individual = new I;
      init( *individuals[i].individual, gen );
    }
  }

  ~population( ) {
    for (uint i = 0; i < individuals.size(); ++i)
      delete individuals[i].individual;
  }

  uint size() {
    return individuals.size();
  }

  uint get_age() {
    return age;
  }

  I& get_best ( ) const {
    return *individuals[0].individual;
  }

  ValueType get_best_value ( ) const {
    return individuals[0].fitness;
  }


  // The function takes as parameters the 4 functions/functors described above. For more details:
  // http://stackoverflow.com/questions/1174169/function-passed-as-template-argument
  // Called without arguments it will call the default functions inside ::I, otherwise the function/functor
  // passed as argument.
  template<typename F, typename M, typename C, typename G = Generation<I> >
  void run ( uint generations, F fitness, M mutate, C crossover, G generation = G() ) {

    for ( uint i = 0; i < generations; ++i ) {
      // this executes the genetic operators
      genetic_operators( mutate, crossover );

      // this execute the (probabilistic) selection step
      selection( fitness, generation, age);
      ++age;
    }
  }

  template<typename M, typename C>
  void genetic_operators( M mutate, C crossover ) {
    uint lastSize = individuals.size();

    // important: the size increases during execution, so we have to stop
    // when we finish the OLD (current) population
    for ( uint j = 0; j < lastSize; ++j ) {
      if ( gen.real() < pmutation ) {
        // creates a copy (in our case with a new, empty, tape)
        I* newone = new I( *individuals[j].individual );
        // executes mutation on the NEW copy and push it to the end
        mutate( *newone, gen );
        individuals.push_back( triple( newone, true ) );
        explored++;
      }

      if ( gen.real() < pcrossover ) {
        uint partner_id;
        do {
          partner_id = gen.integer() % size();
        } while (partner_id == j);

	I* newone = new I( *individuals[j].individual );
	I* newtwo = new I( *individuals[partner_id].individual );
	crossover( *newone, *newtwo, gen );
	individuals.push_back( triple( newone, true ) );
	individuals.push_back( triple( newtwo, true ) );

        explored += 2;
      }
    }
  }

  template<typename F, typename G>
  void selection ( F fitness, G generation, uint generationNumber ) {

    // run Turing machines if needed
    for (uint j = 0; j < individuals.size(); ++j) {
      generation( *individuals[j].individual, generationNumber );

      // if needed, call the fitness and store the result
      if ( individuals[j].changed || !stationary ) {
        individuals[j].fitness = fitness( *individuals[j].individual );
        individuals[j].changed = false;
      }
    }

    sort( individuals.begin(), individuals.end() );

    // Here we have to be careful.. There was some very subdle errors.
    // The idea is: iterate over every individual and if it has to die, delete it.
    // There are two problems:
    // 1) if one individual is deleted we can't increase the index i, because another individual
    // has been moved there (from erase). So we have to call early_death also for that individual.
    // 2) also with this correction we are actually wrong for two reasons (depending if we deleted
    // or not the previous individual):
    //    a) the individual now at position i can now be the last individual we had in the ranking!!
    //    b) otherwise we are using a bigger value for the sigmoid function, because it's like if we
    //       moved the whole population back of (number-of-erased-individuals) positions,
    //       while we still get the value of sigmoid(i). I verified this printing the population
    //       size, that was bigger than expected.
    //
    //for (uint i = 0; i < individuals.size(); ) {
    //  if (early_death(i)) erase(i);
    //  else  ++i;
    //}
    //
    // I think that the quicker way to implement this is to simply begin the scan from the back.
    // In this case the order of the individuals that still has to be decided, is preserved.

    // some individuals die in an accident :

    if ( individuals.size() < maximumPopulation ) return;

    int safe = 100;
    int nonSafe = individuals.size() - safe;
    double threshold = double( individuals.size() - maximumPopulation ) / nonSafe;

    for (int i = individuals.size()-1; i >= safe; --i) {
      if ( gen.real() < threshold ) erase(i);
    }
  }

  friend ostream& operator<< ( ostream & os, const population& o ) {
    //for ( uint i = 0; i < o.particles.size(); ++i )
    //    os << o.particles[i] << endl;

    os << "BEST: " << o.get_best_value();
    //os << " at " << o.get_best();
    return os;
  }
};

} // namespace ml
#endif // POPULATION_HPP
