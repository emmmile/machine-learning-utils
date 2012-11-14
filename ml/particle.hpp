#ifndef PARTICLE_H
#define PARTICLE_H

#include "vect.hpp"
#include "random.hpp"
#include "concepts.hpp"
#include <iostream>
using namespace std;



// The basic particle step in the swarm. Here we loose of generality
// in the sense S must have operators compatible with a vector space
// (e.g. taking the i-th element with [] and multiplying it by a double).
template<class S, unsigned int N>
class RandomStretch : public Mutation<S,N> {
public:
  inline void operator() ( S& p, const double phi, Random& gen ) {
    for ( unsigned int i = 0; i < N; ++i )
      p[i] *= phi * gen.real();
  }
};



template<class S, unsigned int N>
class SwarmMutation : public Mutation<S,N> {
public:
  inline void operator() ( S& p, const double step, Random& gen ) {
    p[gen.integer() % N] += gen.realnegative() * step;
  }
};



template<class SwarmType,
         uint N,
         class VelocityType = SwarmType,
         class ValueType = double>
class particle {
  SwarmType position;
  VelocityType velocity;
  SwarmType pbest;		// personal best position
  ValueType value;		// personal best value

  vector<particle*> neighbours;

public:
  particle ( ) {
  }

  void set ( const SwarmType& p, const VelocityType& v, particle** beg, particle** end ) {
    this->position = p;
    this->velocity = v;
    this->pbest = p;
    this->neighbours = vector<particle*>( beg, end );
  }

  void set ( const SwarmType& p, const VelocityType& v, const vector<particle*>& n ) {
    this->position = p;
    this->velocity = v;
    this->neighbours = n;
    this->pbest = p;
    //this->value = ?;
  }

  template<typename F>
  void initialize ( F function ) {
    this->value = function( this->position );
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
    VelocityType cognitive = pbest - this->position;
    VelocityType social = (*min_element( neighbours.begin(), neighbours.end(), cmp_ptr ))->pbest - this->position;

    RandomStretch<VelocityType, N> stretch;
    stretch( cognitive, phi1, gen );
    stretch( social,    phi2, gen );

    this->velocity = ( costriction * ( this->velocity + cognitive + social ) );
    // EXPERIMENTAL, sometimes I mutate the velocity to avoid getting stuck somewhere
    //SwarmMutation<VelocityType, N> mutation;
    //mutation( this->velocity, 10.0 / costriction, gen );

    // adds the velocity to the current position
    this->position += this->velocity;
    
    // calculates the new value of the function
    ValueType current = function( this->position );
    if ( current < value ) {
      pbest = this->position;
      value = current;
    }
  }

  SwarmType get_position ( ) const {
    return position;
  }

  ValueType get_best_value( ) const {
    return value;
  }

  SwarmType get_best ( ) const {
    return pbest;
  }


  bool operator< ( const particle& a ) const {
    return value < a.value;
  }

  static bool cmp_ptr ( const particle* a, const particle* b ) {
    return *a < *b;
  }

  friend ostream& operator<< ( ostream & os, const particle& p ) {
    //os << "position: " << p.position << endl;
    //os << "velocity:" << p.velocity << endl;
    //os << "personal best: " << p.pbest << endl;
    os << "personal best value: " << p.value;
    return os;
  }
};

#endif // PARTICLE_H
