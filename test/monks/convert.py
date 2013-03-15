#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys


maxvalues = [1,3,3,2,3,4,2]

def convert ( filename ):
	for line in open( filename ):
		numbers = line.rstrip().split()
		i = 0
		for number in numbers:
			out = [0] * maxvalues[i]
			if ( maxvalues[i] == 1 ):
				out[0] = int(number)
			else:
				out[int(number)-1] = 1
			
			for l in out:
				print( l, end=" " )
			i = i + 1
		print()

if __name__ == '__main__':
	
	convert( sys.argv[1] )
