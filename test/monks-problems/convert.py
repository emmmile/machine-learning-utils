#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fileinput
import sys

# these are taken from the monks problem pdf, but can be
# also inferred from the data files
maxvalues = [1,3,3,2,3,4,2]

def convert ( ):
  for line in fileinput.input():
    numbers = line.rstrip().split()
    i = 0
    for number in numbers[:-1]:  #the last element is an id
      out = [0] * maxvalues[i]
      out[int(number)-1] = min(1,int(number))

      for l in out:
        sys.stdout.write( "" + str(l) + " " )
      i = i + 1
    
    sys.stdout.write( "\n" )


if __name__ == '__main__':
	convert( )
