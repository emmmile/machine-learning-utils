#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP

template<class S, unsigned int N>
struct Initialization {
  inline void operator() ( S& p, Random& gen ) {
    // default init, do nothing
  }
};


template<class S, unsigned int N>
class Mutation {
public:
  // basic mutation operator
  inline void operator() ( S& p, Random& gen ) {
    // mutation code
  }
};


#endif // CONCEPTS_HPP
