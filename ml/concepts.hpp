#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP
#include <sys/types.h>
#include "random.hpp"
using namespace math;

namespace ml {

template<class S>
struct Initialization {
  inline void operator() ( S& p, Random& gen ) {
  }
};


template<class S>
class Mutation {
public:
  // basic mutation operator
  inline void operator() ( S& p, Random& gen ) {
  }
};


template<class S>
class Crossover {
public:
  // basic crossover operator
  inline void operator() ( S& a, S& b, Random& gen ) {
  }
};

} // namespace ml
#endif // CONCEPTS_HPP
