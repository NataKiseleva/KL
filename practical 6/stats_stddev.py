# -*- coding: utf-8 -*-
"""stats-stddev.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18a_OSVkD66VHxWIph402diwOE2BAYLp1
"""

#!/usr/bin/python3

import sys
import statistics

s= []

for line in sys.stdin:
  s.append(int(line))

print(statistics.pstdev(s))