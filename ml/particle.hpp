#ifndef PARTICLE_H
#define PARTICLE_H

#include "random.hpp"
#include "concepts.hpp"
#include <iostream>
#include <array>
using namespace std;
namespace ml {


// The basic particle step in the swarm. Here we loose of generality
// in the sense S must have operators compatible with a vector space
// (e.g. taking the i-th element with [] and multiplying it by a double).
template<class S, unsigned int N>
class RandomStretch : public Mutation<S> {
public:
  inline void operator() ( S& p, const double& phi, Random& gen ) {
    // the signs of p are conserved
    for ( unsigned int i = 0; i < N; ++i )
      p[i] *= phi * gen.real();
  }
};



template<class S, unsigned int N>
class SwarmMutation : public Mutation<S> {
public:
  inline void operator() ( S& p, const double& step, Random& gen ) {
    p[gen.integer() % N] += gen.realnegative() * step;
  }
};



template<class SwarmType,
         uint N,
         class VelocityType,
         class ValueType,
         uint M>
class particle {
  SwarmType __position;
  VelocityType __velocity;
  SwarmType __personal_best;		// personal best position
  ValueType __personal_best_value;		// personal best value

  array<particle*, M> __neighbours;

public:
  particle ( ) {
  }

  template<typename Iterator>
  void set ( const SwarmType& p, const VelocityType& v, Iterator beg, Iterator end ) {
    this->__position = p;
    this->__velocity = v;
    this->__personal_best = p;
    for( Iterator i = beg; i != end; ++i ) __neighbours[i-beg] = *i;
  }

  template<typename F>
  void initialize ( F function ) {
    this->__personal_best_value = function( this->__position );
  }

  // NOTE. This is extremely interesting. In a minimization problem I erroneusly set max_element, instead of
  // min_element, so I was searching for the maximum value in the neighbours instead of the maximum!!!
  // The algorithm was working anyway, slowly but working.
  // XXX review but I think this parameters should be double. I cannot be aware of the type of the vector
  // or its components, must be something generic. And since we are dealing with probabilities I think the must
  // be double.
  // TODO generalize this
  template<typename F>
  void move ( F function, double costriction, double phi1, double phi2, Random& gen ) {
    // this maybe can be something else, i.e. it's not straightforward that a difference
    // between two S is again something meaningful in that space (think at turing machines).
    VelocityType cognitive = __personal_best - this->__position;
    VelocityType social = ( *min_element( __neighbours.begin(), __neighbours.end(), __cmp ) )->__personal_best - this->__position;

    RandomStretch<VelocityType, N> stretch;
    stretch( cognitive, phi1, gen );
    stretch( social,    phi2, gen );

    this->__velocity = ( costriction * ( this->__velocity + cognitive + social ) );
    // EXPERIMENTAL, sometimes I mutate the velocity to avoid getting stuck somewhere
    //SwarmMutation<VelocityType, N> mutation;
    //if ( gen.real() < 0.05 ) mutation( this->__velocity, 10.0 / costriction, gen );

    // adds the velocity to the current position
    this->__position += this->__velocity;

    // calculates the new value of the function
    ValueType current = function( this->__position );
    if ( current < __personal_best_value ) {
      __personal_best = this->__position;
      __personal_best_value = current;
    }
  }

  SwarmType position ( ) const {
    return __position;
  }

  ValueType best_value( ) const {
    return __personal_best_value;
  }

  SwarmType best ( ) const {
    return __personal_best;
  }

  bool operator< ( const particle& a ) const {
    return __personal_best_value < a.__personal_best_value;
  }

  static bool __cmp ( const particle* a, const particle* b ) {
    return a->__personal_best_value < b->__personal_best_value;
  }

  friend ostream& operator<< ( ostream & os, const particle& p ) {
    //os << "position: " << p.position << endl;
    //os << "velocity:" << p.velocity << endl;
    //os << "personal best: " << p.pbest << endl;
    os << "personal best value: " << p.__personal_best_value;
    return os;
  }
};

} // namespace ml
#endif // PARTICLE_H
