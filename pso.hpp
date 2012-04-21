#ifndef PSO_HPP
#define PSO_HPP

#include <iostream>
#include "vect.hpp"
#include "particle.hpp"
//#include <QImage>
//#include <sstream>
//#include <QPainter>
#include <iomanip>
using namespace std;


template<class S = vect<double, 2> >
class pso {
	typedef typename S::ctype ctype;	// XXX already in particle

	Random gen;
	vector<particle<S> > particles;
	S minvalues;
	S maxvalues;
public:
	//TODO implement constructor that takes also a functor
	pso ( uint size, ctype (*f)( const S& ), const S& minvalues,
		const S& maxvalues, int seed = 12345678 ) : particles( size ) {

		gen.seed( seed );
		this->minvalues = minvalues;
		this->maxvalues = maxvalues;

		for ( uint i = 0; i < particles.size(); ++i ) {
			// random position between minvalues and maxvalues
			S p ( minvalues, maxvalues, gen );
			S v;

			// lbest implementation
			vector<particle<S>*> ring ( 2 );
			ring[0] = &particles[(i - 1 + particles.size() ) % particles.size()];
			ring[1] = &particles[(i + 1 + particles.size() ) % particles.size()];

			particles[i].set( p, v, ring, f );
		}
	}

	void run ( uint iterations, ctype phi1 = 1.8, ctype phi2 = 2.3 ) {
		ctype phi = phi1 + phi2;
		ctype costriction = 2.0 / ( phi - 2.0 + sqrt( phi * phi - 4.0 * phi ) );
		//costriction *= 1.2;	// this is good for the 100 dimensional case, maybe because increases the convergence time

		//cout << costriction << endl << phi1 * costriction << endl << phi2 * costriction << endl;
		// implementation of linear inertia
		/*T winitial = 0.9;
		T wfinal = 0.4;
		T w = winitial;
		T wstep = ( winitial - wfinal ) / iterations;*/

		for ( uint j = 0; j < iterations; ++j ) {
			for ( uint i = 0; i < particles.size(); ++i )
				particles[i].move( costriction, phi1, phi2, gen );
		}
	}

	S get_best ( ) const {
		return min_element( particles.begin(), particles.end(), cmp<S> )->get_best();
	}

	ctype get_best_value ( ) const {
		return min_element( particles.begin(), particles.end(), cmp<S> )->get_best_value();
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
