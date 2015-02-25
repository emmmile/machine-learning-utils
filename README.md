# README #

This little project contains a very simple (but quite fast) implementation of neural networks, swarms (PSO) and populations (in the sense of genetic algorithms).
It can be opened with Qt Creator IDE or compiled directly from shell using [Sconstruct](http://www.scons.org/).

### How to set up? ###

It is assumed that the following dependencies are installed:

* Python 3 (needed only for some test scripts, it could be easily disabled)
* [Eigen 3](http://eigen.tuxfamily.org/)

To compile the tests just run:

```
#!sh
git clone https://emmmile@bitbucket.org/emmmile/ml.git
cd ml/
git submodule init
git submodule update
scons
```


To run the example application instead:

```
#!sh
cd test
./example
```


### Questions? ###

Write me at emilio.deltessa@gmail.com.