#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

train = []
test = []

for i in open( "output", "r" ):
        first, second = i.split()
        train.append( first )
        test.append( second )
	

plt.plot( train )
plt.plot( test )
plt.ylabel('error')
plt.show()
