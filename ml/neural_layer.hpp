#ifndef NEURAL_LAYER_HPP
#define NEURAL_LAYER_HPP

#include "vect.hpp"
#include <iostream>
#include <cmath>
using namespace std;
using namespace math;


enum activation { LINEAR, SIGMOID };
enum learning { ONLINE, BATCH };

template<size_t N, size_t I, activation A, learning L, class T>
class layer_base {
protected:
  typedef shared_matrix<N, I + 1, T> weightsType;
  typedef vect<N, T> outType;
  typedef vect<I, T> inType;
  typedef vect<I + 1, T> richInType;

  weightsType __weights;
  outType     __output;   // last output
  richInType  __input;    // last input

  // add the fictitious input and save it in the member variable
  const richInType& setInput( const inType& input ) {
    __input[I] = -1.0;
    copy( input.data(), input.data() + I, __input.data() );

    return __input;
  }

public:
	typedef vect<N * (I + 1),T> vector_type;

	layer_base ( T* data ) : __weights( data ) {
	}

	// compute the component-wise delta calculation (in-place)
	virtual inline outType computedelta ( outType& v ) = 0;

	// the activation is component-wise too
	virtual inline outType& activation ( const outType& net ) = 0;

	// the back-propagation (GD) algorithm
	inType backprop ( outType& error, bool first = false );

  const weightsType& weights() const {
    return __weights;
  }

  const outType& output() const {
    return __output;
  }

  inline static constexpr size_t size ( ) {
    return N * (I + 1);
  }

	friend ostream& operator<< ( ostream & os, const layer_base& l ) {
    //os.precision( 2 );
    return os << "weights in the layer:\n" << l.weights();
  }
};


// specializations for the activation function
template<size_t N, size_t I, activation A, learning L, class T>
class layer_activation : public layer_base<N,I,A,L,T> {
	typedef layer_base<N,I,A,L,T> base;
public:
	// available in gcc 4.8
	//using neural_layer_base<N,I,A,L,T>::neural_layer_base;
	layer_activation ( T* data ) : base( data ) { }
};

template<size_t N, size_t I, learning L, class T>
class layer_activation<N,I,LINEAR,L,T> : public layer_base<N,I,LINEAR,L,T> {
	typedef layer_base<N,I,LINEAR,L,T> base;
public:
	layer_activation ( T* data ) : base( data ) { }

	inline typename base::outType& activation ( const typename base::outType& net ) {
		return this->__output = net;
	}

	// compute the component-wise delta calculation (in-place)
	inline typename base::outType computedelta ( typename base::outType& v ) {
		return v;
	}
};

template<size_t N, size_t I, learning L, class T>
class layer_activation<N,I,SIGMOID,L,T> : public layer_base<N,I,SIGMOID,L,T> {
	typedef layer_base<N,I,SIGMOID,L,T> base;

	inline static double sigmoid( const double& value, const double lambda = 1.0 ) {
		return 1.0 / ( 1.0 + exp( -lambda * value ) );
	}

public:
	layer_activation ( T* data ) : base( data ) { }

	inline typename base::outType& activation ( const typename base::outType& net ) {
		for ( size_t i = 0; i < N; ++i ) {
			this->__output[i] = sigmoid( net[i] );
		}

		return this->__output;
	}

	inline typename base::outType computedelta ( typename base::outType& v ) {
		for ( size_t k = 0; k < N; ++k )
				v[k] = v[k] * (1.0 - this->__output[k]) * this->__output[k];

		return v;
	}
};


// specializations for the kind of learning
template<size_t N, size_t I, activation A, learning L, class T>
class layer_learning : public layer_activation<N,I,A,L,T> {
	typedef layer_activation<N,I,A,L,T> base;
public:
	layer_learning ( T* data ) : base( data ) { }
};

template<size_t N, size_t I, activation A, class T>
class layer_learning<N,I,A,ONLINE,T> : public layer_activation<N,I,A,ONLINE,T> {
	typedef layer_activation<N,I,A,ONLINE,T> base;
public:
	layer_learning ( T* data ) : base( data ) { }
};

template<size_t N, size_t I, activation A, class T>
class layer_learning<N,I,A,BATCH,T> : public layer_activation<N,I,A,BATCH,T> {
	typedef layer_activation<N,I,A,BATCH,T> base;
public:
	layer_learning ( T* data ) : base( data ) { }
};



// the final interface, using both specializations
template<size_t N, size_t I, activation A, learning L, class T>
class neural_layer : public layer_learning<N,I,A,L,T> {
	typedef layer_learning<N,I,A,L,T> base;
public:
	neural_layer ( T* data ) : base( data ) { }

		// compute the output of the network given an input
	const typename base::outType& compute ( const typename base::inType& input ) {
		// the calculation is basically a matrix-vector multiplication where
		// the input has been added a ficticious value, -1, so there are two possibilities:
		//   - copy the input vector in a bigger vector and perform the multiplication
		//   - split the multiplication as follows:
		//     out = -W0 + W*in
		// i choose to copy the input, it's O(n) instead of O(n^2)
		this->setInput( input );

		//return __output;
		return this->activation( this->__weights * this->__input );
	}
};



#endif // NEURAL_LAYER_HPP
