import math
import numpy as np
import sys

class RandomVectorGenerator:
    def __init__(self, length, v_range):
        self.vec_length = length
        self.vec_range = v_range
        self.vector = np.empty(length)

    def get(self):
        for i in range(self.vec_length):
            # val_range = self.vec_range.get_range(i % self.vec_range.num_of_ranges)
            val_range = self.vec_range.get_range(i)
            # print(val_range)
            abs_range = val_range[1] - val_range[0]
            # print(f'{abs_range} = {val_range[1]} - {val_range[0]}')
            # print(abs_range)
            # rand = abs(np.random.normal())
            # print(f'rand: {rand}')
            # with_range = rand* abs_range
            # print(f'with_range: {with_range}')
            # val = with_range - abs(val_range[0])
            # print(f'val: {val}')


            val = abs(np.random.normal()) * abs_range - abs(val_range[0])

            self.vector[i] = val
        # print(self.vector)
        # sys.exit(-1)
        # abs_range = self.vec_range[1] - self.vec_range[0]
        # for i in range(self.vec_length):
        #     self.vector[i] = abs(np.random.normal()) * abs_range - abs(self.vec_range[0])

        return self.vector

    def get_(self):
        for i in range(self.vec_length):
            self.vector[i] = np.random.normal()

        return self.vector

class ValueRanges:
    def __init__(self, num_of_ranges, maxes, mins):
        self.maxes = maxes
        self.mins = mins
        self.num_of_ranges = num_of_ranges

    def get_range(self, i):
        if i > self.num_of_ranges or i < 0:
            print("Error -- i out of bounds for ValueRanges class")
        return (math.floor(self.mins[i]), math.ceil(self.maxes[i]))