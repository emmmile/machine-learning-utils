#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fileinput
import sys

def convert ( ):
  for line in fileinput.input():
    if line.startswith( "#" ) or not line.strip():
      continue

    rowid, *inputs, targetx, targety = line.rstrip().split(',')

    #sys.stdout.write( targetx + " " + targety + " " )
    sys.stdout.write( "{:10.7} {:10.7} ".format( float(targetx), float(targety) ) )
    for num in inputs:
      sys.stdout.write( "{:10.7}".format( float(num) ) )

    sys.stdout.write( "\n" )

if __name__ == '__main__':
  convert( )
