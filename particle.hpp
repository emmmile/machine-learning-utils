#ifndef PARTICLE_H
#define PARTICLE_H

#include "vect.hpp"
#include <iostream>
using namespace std;

template<class S>
class particle;


template<class S>
bool cmp ( const particle<S>& a, const particle<S>& b ) {
	return a.get_best_value() < b.get_best_value();
}

template<class S>
bool cmp_ptr ( const particle<S>* a, const particle<S>* b ) {
	return cmp<S>( *a, *b );
	//return a->get_best_value() < b->get_best_value();
}


template<class S = vect<double, 2> >
class particle {
	typedef typename S::ctype ctype;

	S position;
	S velocity;
	S pbest;		// personal best position
	ctype value;		// personal best value

	vector<particle*> neighbours;
	ctype (*f)( const S& );
public:
	particle ( ) {

	}

	void set ( const S& p, const S& v, const vector<particle*>& n, ctype (*f)( const S& ) ) {
		this->position = p;
		this->velocity = v;
		this->neighbours = n;
		this->pbest = p;
		this->value = f(p);
		this->f = f;
	}

	// NOTE. This is extremely interesting. In a minimization problem I erroneusly set max_element, instead of
	// min_element, so I was searching for the maximum value in the neighbours instead of the maximum!!!
	// The algorithm was working anyway, slowly but working.
	void move ( ctype costriction, ctype phi1, ctype phi2, Random& gen ) {
		S cognitive = pbest - this->position;
		S social = (*min_element( neighbours.begin(), neighbours.end(), cmp_ptr<S> ))->pbest - this->position;

		this->velocity = ( costriction * ( this->velocity + cognitive.random_stretch( phi1, gen ) + social.random_stretch( phi2, gen ) ) );
		// EXPERIMENTAL, sometimes I mutate the velocity to avoid getting stuck somewhere
			//).random_mutate( 10.0 / costriction, gen );
		this->position += this->velocity;

		ctype current = f(this->position);
		if ( current < value ) {
			pbest = this->position;
			value = current;
		}
	}

	S get_position ( ) const {
		return position;
	}

	ctype get_best_value( ) const {
		return value;
	}

	S get_best ( ) const {
		return pbest;
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
