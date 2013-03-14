#ifndef SWARM_HPP
#define SWARM_HPP

#include <iostream>
#include <algorithm>
#include "particle.hpp"
#include "concepts.hpp"
#include <iomanip>
using namespace std;
namespace ml {

// needed operators
// VelocityType VelocityType::operator+ ( const VelocityType& ) const
// friend VelocityType operator* ( const VelocityType&, const double& )
// VelocityType SwarmType::operator- ( const SwarmType& ) const
// SwarmType SwarmType::operator+= ( const VelocityType& )

/* If VelocityType is a vector there no problem with the first two.
 * VelocityType is a vector in general, PSO is meant for that. But the last two operator
 * has no meaning outside PSO, in general.
 * So I think is better to design PSO specifically for a vector type, and demand
 * the conversion from a generic type T to vector outside of PSO. This could lead
 * to some complications in the design of the T type if efficiency is needed, I see this
 * as the only problem.
 *
 * Example: if T is neural network, can be meaningful to test a network topology (T type),
 *   with a certain set of weights, different from the ones currently in the network.
 *   Maybe this functionality can be given as a static method, or a constructor from a
 *   vector of weights (doubles).
 *   An operator T - T, or T += VelocityType has no meaning, except for PSO.
 */



template<class SwarmType, size_t N, class VelocityType = SwarmType, class ValueType = double>
class swarm {
  static const size_t __neighbours = 2;
  typedef particle<SwarmType, N, VelocityType, ValueType, __neighbours> __particleType;

  Random __generator;
  vector<__particleType> __particles;
  size_t __explored;

public:
  typedef __particleType particleType;

  template<typename I>
  explicit swarm ( size_t size,
									 I init = I(), int32_t seed = Random::seed() )
		: __generator( seed ), __particles( size ), __explored( 0 )
  {
    for ( size_t i = 0; i < __particles.size(); ++i ) {
      SwarmType p;
      VelocityType v;
      init( p, __generator );

      array<particleType*, __neighbours> ring;
      ring[0] = &__particles[(i - 1 + __particles.size() ) % __particles.size()];
      ring[1] = &__particles[(i + 1 + __particles.size() ) % __particles.size()];

      __particles[i].set( p, v, ring.begin(), ring.end() );
    }
  }

  template<typename F>
  void run ( size_t iterations, F function, double phi1 = 1.8, double phi2 = 2.3 ) {
    double phi = phi1 + phi2;
    double costriction = 2.0 / ( phi - 2.0 + sqrt( phi * phi - 4.0 * phi ) );
    //costriction *= 1.2;	// this is good for ackley-100, maybe because increases the convergence time

    for ( size_t i = 0; i < __particles.size(); ++i, ++__explored )
      __particles[i].initialize( function );

    for ( size_t j = 0; j < iterations; ++j ) {
      for ( size_t i = 0; i < __particles.size(); ++i, ++__explored )
        __particles[i].move( function, costriction, phi1, phi2, __generator );
    }
  }

  size_t explored() const {
    return __explored;
  }

  const SwarmType& best ( ) const {
    return min_element( __particles.begin(), __particles.end() )->best();
  }

	SwarmType best ( ) {
		return min_element( __particles.begin(), __particles.end() )->best();
	}

  const ValueType& best_value ( ) const {
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
