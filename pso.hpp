#ifndef PSO_HPP
#define PSO_HPP

#include <iostream>
#include "vect.hpp"
#include "particle.hpp"
#include <iomanip>
using namespace std;


template<class SwarmType>
struct Init {
  inline void operator() ( SwarmType& p, Random& gen ) {
    p.init(gen);
  }
};

// per le differenze
// DiffType SwarmType::operator- ( const SwarmType& ) const

// per la velocity
// DiffType DiffType::operator+ ( const DiffType& ) const
// VelocityType VelocityType::operator+ ( const DiffType& ) const
// friend VelocityType operator* ( const VelocityType&, const double& )

// per l'aggiornamento della posizione
// SwarmType SwarmType::operator+= ( const VelocityType& )





// MAYBE DiffType == VelocityType?

// VelocityType SwarmType::operator- ( const SwarmType& ) const
// VelocityType VelocityType::operator+ ( const VelocityType& ) const
// friend VelocityType operator* ( const VelocityType&, const double& )
// SwarmType SwarmType::operator+= ( const VelocityType& )

template<class SwarmType,
         class VelocityType = SwarmType,
         class DiffType = SwarmType,
         class ValueType = double>
class pso {
  typedef particle<SwarmType, VelocityType, DiffType, ValueType> particleType;

  Random gen;
  vector<particleType> particles;


public:

  template<typename I = Init<SwarmType> >
  pso ( uint size, int seed = 12345678, I init = I() ) : gen( seed ), particles( size ) {
    for ( uint i = 0; i < particles.size(); ++i ) {
      SwarmType p;
      VelocityType v;
      init( p, gen );

      vector<particleType*> ring ( 2 );
      ring[0] = &particles[(i - 1 + particles.size() ) % particles.size()];
      ring[1] = &particles[(i + 1 + particles.size() ) % particles.size()];

      particles[i].set( p, v, ring );
    }
  }

  template<typename F>
  void run ( uint iterations, F function, double phi1 = 1.8, double phi2 = 2.3 ) {
    double phi = phi1 + phi2;
    double costriction = 2.0 / ( phi - 2.0 + sqrt( phi * phi - 4.0 * phi ) );
    //costriction *= 1.2;	// this is good for the 100 dimensional case, maybe because increases the convergence time

    //cout << costriction << endl << phi1 * costriction << endl << phi2 * costriction << endl;
    // implementation of linear inertia
    /*T winitial = 0.9;
    T wfinal = 0.4;
    T w = winitial;
    T wstep = ( winitial - wfinal ) / iterations;*/

    for ( uint i = 0; i < particles.size(); ++i )
      particles[i].initialize( function );

    for ( uint j = 0; j < iterations; ++j ) {
      for ( uint i = 0; i < particles.size(); ++i )
        particles[i].move( function, costriction, phi1, phi2, gen );
    }
  }

  SwarmType get_best ( ) const {
    return min_element( particles.begin(), particles.end() )->get_best();
  }

  ValueType get_best_value ( ) const {
    return min_element( particles.begin(), particles.end() )->get_best_value();
  }

  friend ostream& operator<< ( ostream & os, const pso& o ) {
    //for ( uint i = 0; i < o.particles.size(); ++i )
    //    os << o.particles[i] << endl;

    os << "BEST: " << o.get_best_value();
    //os << " at " << o.get_best();
    return os;
  }
};




#endif // PSO_HPP
