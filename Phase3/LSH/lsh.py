from .hash_family import HashFamily

import numpy as np


class LocalitySensitiveHashing:
    def __init__(self, l, k, r, length, val_range, num_buckets_per_hash):
        self.l = l
        self.k = k
        self.family = HashFamily(r, length, val_range, num_buckets_per_hash)
        self.total_buckets = self.k * num_buckets_per_hash

        self.layers = []
        
        self.buckets_used = []

        for i in range(self.l):
            temp = []
            for j in range(self.k):
                temp.append(self.family.random_hash())

            self.layers.append(temp)

    def hash_layer(self, v, layer):
        hash_value = ''
        for hash_function in self.layers[layer]:
            temp = hash_function.hash(v)
            hash_value += '{}'.format(temp)
        return hash_value

    def hash(self, v):
        buckets = []
        for i in range(self.l):
            buckets.append(self.hash_layer(v, i))
        self.buckets_used += buckets
        return buckets
        # return (hash1, hash2, hash3, ..., hashL)

    # def get_bucket_ids(self):
    #     for i in range(self.num_buckets_per_hash):
    #         for j in range(self.k):
    #             id = 


        
    

    
    
