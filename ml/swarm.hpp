#ifndef SWARM_HPP
#define SWARM_HPP

#include <iostream>
#include <algorithm>
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
         class ValueType = double,
         uint M = 2>
class swarm {
  typedef particle<SwarmType, N, VelocityType, ValueType, M> particleType;

  Random __generator;
  vector<particleType> __particles;
  uint __explored;
public:

  template<typename I>
  explicit swarm ( uint size,
                   I init = I() )
    : __generator( 12345678 ), __particles( size ), __explored( 0 )
  {
    for ( uint i = 0; i < __particles.size(); ++i ) {
      SwarmType p;
      VelocityType v;
      init( p, __generator );

      array<particleType*, M> ring;
      ring[0] = &__particles[(i - 1 + __particles.size() ) % __particles.size()];
      ring[1] = &__particles[(i + 1 + __particles.size() ) % __particles.size()];

      __particles[i].set( p, v, ring.begin(), ring.end() );
    }
  }

  template<typename F>
  void run ( uint iterations, F function, double phi1 = 1.8, double phi2 = 2.3 ) {
    double phi = phi1 + phi2;
    double costriction = 2.0 / ( phi - 2.0 + sqrt( phi * phi - 4.0 * phi ) );
    //costriction *= 1.2;	// this is good for ackley-100, maybe because increases the convergence time

    for ( uint i = 0; i < __particles.size(); ++i, ++__explored )
      __particles[i].initialize( function );

    for ( uint j = 0; j < iterations; ++j ) {
      for ( uint i = 0; i < __particles.size(); ++i, ++__explored )
        __particles[i].move( function, costriction, phi1, phi2, __generator );
    }
  }

  uint explored() const {
    return __explored;
  }

  SwarmType best ( ) const {
    return min_element( __particles.begin(), __particles.end() )->best();
  }

  ValueType best_value ( ) const {
    return min_element( __particles.begin(), __particles.end() )->best_value();
  }

  friend ostream& operator<< ( ostream & os, const swarm& o ) {
    //for ( uint i = 0; i < o.particles.size(); ++i )
    //    os << o.particles[i] << endl;

    os << "best value found: " << o.best_value();
    //os << " at " << o.get_best();
    return os;
  }
};



} // namespace ml
#endif // SWARM_HPP
