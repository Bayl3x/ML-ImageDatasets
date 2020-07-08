
import numpy as np
import math
import sys

from .random_vector import RandomVectorGenerator


class HashFamily:
    
    def __init__(self, r, length, val_range, num_buckets):
        self.r = r
        self.random_vector = RandomVectorGenerator(length, val_range)
        self.n = num_buckets

    def random_a(self):
        return self.random_vector.get()

    def random_b(self):
        return np.random.uniform(high=self.r)
    
    def random_hash(self):
        return HashFunction(self.random_a(), self.random_b(), self.r, self.n)


class HashFunction:
    def __init__(self, a, b, r, n):
        self.a = a
        self.b = b
        self.r = r
        self.n = n

    def hash(self, v):
        if self.a.size != v.size:
            print("Not of same size")
            print(self.a)
            print(v)
            print(len(self.a))
            print(len(v))
            sys.exit(-1)
            
        return int(math.floor((np.dot(self.a, v) + self.b) / self.r)) % self.n
    
    def print_info(self):
        print("Hash function - a: {}, b: {}, r: {}, n: {}".format(len(self.a), self.b, self.r, self.n))

