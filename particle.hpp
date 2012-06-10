#ifndef PARTICLE_H
#define PARTICLE_H

#include "vect.hpp"
#include "random.hpp"
#include <iostream>
using namespace std;



template<class SwarmType,
         class VelocityType = SwarmType,
         class DiffType = SwarmType,
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
    DiffType cognitive = pbest - this->position;
    DiffType social = (*min_element( neighbours.begin(), neighbours.end(), cmp_ptr ))->pbest - this->position;

    // is costriction meaningful in every type?
    this->velocity = ( costriction * ( this->velocity + cognitive.random_stretch( phi1, gen ) + social.random_stretch( phi2, gen ) ) );
    // EXPERIMENTAL, sometimes I mutate the velocity to avoid getting stuck somewhere
            //).random_mutate( 10.0 / costriction, gen );
    this->position += this->velocity;

    ValueType current = function(this->position);
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
   //return a->get_best_value() < b->get_best_value();
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
