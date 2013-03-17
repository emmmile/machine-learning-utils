#ifndef NEURAL_PSO_HPP
#define NEURAL_PSO_HPP

#include "dataset.hpp"


template<size_t I, size_t O, class N, class S = double>
class neural_pso {
public:
  dataset<I, O> set;

  neural_pso( dataset<I, O>& set ) : set( set ) {
  }

  // this is for the initialization, using the generator of PSO
  inline void operator() ( N& v, Random& gen ) {
    v.init( gen );
  }

  // this is the objective function, the MSE error of the network
  inline S operator() ( N& v ) {
    return v.error( set );
  }
};

#endif // NEURAL_PSO_HPP
