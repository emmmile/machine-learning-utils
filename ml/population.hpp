#ifndef POPULATION_HPP
#define POPULATION_HPP
#include <type_traits>
#include <vector>
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
         class ValueType = double,
         class V = unsigned short int>
class population {
  // this is used to avoid recalculating the fitness function in case
  // the individual survived immutated from the last generation
  class triple {
  public:
    I individual;
    bool changed;
    ValueType fitness;

    triple( ) : changed(true), fitness( 0.0 ) {
    }

    triple( const I& i, bool changed, const ValueType& d = 0.0 )
      : individual(i), changed(changed), fitness(d)
    {
    }

    inline bool operator< ( const triple& another ) const {
      return fitness < another.fitness;
    }
  };

  vector<triple> __individuals;
  Random __generator;
  double __mutation_probability;
  double __crossover_probability;
  uint __age;
  uint __explored;

  const static uint __maximum_opulation = 400;
  const static uint __initial_population = 100;



public:

  template<typename Init>
  explicit population( uint size,
                       Init init = Init(),
                       double pc = 0.9,
                       double pm = 0.05 )
    : __generator( 12345678 ), __mutation_probability( pm ),
      __crossover_probability( pc ), __age( 0 ), __explored( size )
  {
    __individuals.reserve( size );

    for (uint i = 0; i < size; ++i) {
      I newone;
      init( newone, __generator );
      __individuals.push_back( triple(newone, true) );
    }
  }

  ~population( ) {
  }

  uint size() const {
    return __individuals.size();
  }

  uint age() const {
    return __age;
  }

  uint explored() const {
    return __explored;
  }

  I& best ( ) const {
    return __individuals[0].individual;
  }

  ValueType best_value ( ) const {
    return __individuals[0].fitness;
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
      selection( fitness, generation, __age);
      ++__age;
    }
  }

  template<typename M, typename C>
  void genetic_operators( M mutate, C crossover ) {
    uint lastSize = __individuals.size();

    // important: the size increases during execution, so we have to stop
    // when we finish the OLD (current) population
    for ( uint j = 0; j < lastSize; ++j ) {
      if ( __generator.real() < __mutation_probability ) {
        // creates a copy
        I newone( __individuals[j].individual );
        // executes mutation on the NEW copy and push it to the end
        mutate( newone, __generator );
        __individuals.push_back( triple( newone, true ) );
        __explored++;
      }

      if ( __generator.real() < __crossover_probability ) {
        uint partner;
        do {
          partner = __generator.integer() % size();
        } while (partner == j);

	I newone( __individuals[j].individual );
	I newtwo( __individuals[partner].individual );
	crossover( newone, newtwo, __generator );
	__individuals.push_back( triple( newone, true ) );
	__individuals.push_back( triple( newtwo, true ) );

        __explored += 2;
      }
    }
  }

  void erase( const uint i ) {
    swap( __individuals[i], __individuals.back() );
    // and then remove from the end, since we have O(1) there (and no constraints on the order)
    __individuals.pop_back();
  }

  template<typename F, typename G>
  void selection ( F fitness, G generation, uint generationNumber ) {

    // executes the fitness if needed
    for (uint j = 0; j < __individuals.size(); ++j) {
      generation( __individuals[j].individual, generationNumber );

      // if needed, call the fitness and store the result
      if ( __individuals[j].changed ) {
        __individuals[j].fitness = fitness( __individuals[j].individual );
        __individuals[j].changed = false;
      }
    }

    sort( __individuals.begin(), __individuals.end() );

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
    // I think that the quicker way to implement this is to simply begin the scan from the back.
    // In this case the order of the individuals that still has to be decided, is preserved.

    if ( size() < __maximum_opulation ) return;

    int safe = 100;
    int nonSafe = size() - safe;
    double threshold = double( size() - __maximum_opulation ) / nonSafe;

    for (int i = size() - 1; i >= safe; --i) {
        if ( __generator.real() < threshold ) erase(i);
    }
  }

  friend ostream& operator<< ( ostream & os, const population& o ) {
    //for ( uint i = 0; i < o.particles.size(); ++i )
    //    os << o.particles[i] << endl;

    os << "best value found: " << o.best_value();
    //os << " at " << o.get_best();
    return os;
  }
};

} // namespace ml
#endif // POPULATION_HPP
