#ifndef NEURAL_PSO_HPP
#define NEURAL_PSO_HPP

#include "dataset.hpp"


template<size_t I, size_t O, class N, class S = double>
class neural_pso {
public:
	dataset<I, O> set;

	neural_pso( dataset<I, O>& set ) : set( set ) {
	}

	inline S operator() ( N& v ) {
		return v.error( set );
	}
};

template<class V>
inline void neural_pso_init( V& v, Random& gen ) {
	v.init( gen );
}


#endif // NEURAL_PSO_HPP
