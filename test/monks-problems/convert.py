#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# ./convert path/files.*

# these are taken from the monks problem pdf, but can be
# also inferred from the data files
maxvalues = [1,3,3,2,3,4,2]

def convertFile ( lines, outfile ):
  f = open(outfile,"w")
  for line in lines:
    numbers = line.rstrip().split()
    i = 0
    for number in numbers[:-1]:  #the last element is an id
      out = [0] * maxvalues[i]
      out[int(number)-1] = min(1,int(number))

      for l in out:
        f.write( "" + str(l) + " " )
      i = i + 1
    
    f.write( "\n" )
  f.close()

def convert ( files ):
  for filename in files:
    with open(filename) as f:
      convertFile( f, os.path.split(filename)[1] )
	    

if __name__ == '__main__':
	convert( sys.argv[1:] )
