#ifndef SWARM_HPP
#define SWARM_HPP

#include <iostream>
#include "vect.hpp"
#include "particle.hpp"
#include "concepts.hpp"
#include <iomanip>
using namespace std;
namespace ml {

// needed operators
// VelocityType SwarmType::operator- ( const SwarmType& ) const
// VelocityType VelocityType::operator+ ( const VelocityType& ) const
// friend VelocityType operator* ( const VelocityType&, const double& )
// SwarmType SwarmType::operator+= ( const VelocityType& )




template<class SwarmType,
         uint N,
         class VelocityType = SwarmType,
         class ValueType = double>
class swarm {
  typedef particle<SwarmType, N, VelocityType, ValueType> particleType;

  Random gen;
  vector<particleType> particles;
public:

  template<typename I>
  explicit swarm ( uint size,
                   uint seed = 12345678,
                   I init = I() )
    : gen( seed ), particles( size )
  {
    for ( uint i = 0; i < particles.size(); ++i ) {
      SwarmType p;
      VelocityType v;
      init( p, gen );

      particleType* ring [2];
      ring[0] = &particles[(i - 1 + particles.size() ) % particles.size()];
      ring[1] = &particles[(i + 1 + particles.size() ) % particles.size()];

      particles[i].set( p, v, ring, ring + 2 );
    }
  }

  template<typename F>
  void run ( uint iterations, F function, double phi1 = 1.8, double phi2 = 2.3 ) {
    double phi = phi1 + phi2;
    double costriction = 2.0 / ( phi - 2.0 + sqrt( phi * phi - 4.0 * phi ) );
    //costriction *= 1.2;	// this is good for ackley-100, maybe because increases the convergence time

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

  friend ostream& operator<< ( ostream & os, const swarm& o ) {
    //for ( uint i = 0; i < o.particles.size(); ++i )
    //    os << o.particles[i] << endl;

    os << "BEST: " << o.get_best_value();
    //os << " at " << o.get_best();
    return os;
  }
};



} // namespace ml
#endif // SWARM_HPP
