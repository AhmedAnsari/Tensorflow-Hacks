#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 01:16:02 2018

@author: Ahmed Ansari
@email: ansarighulamahmed@gmail.com
"""
import tensorflow as tf
import numpy as np

def immutable_scatter_nd(inp1, inp2, inp3):
    def map_index_to_flattened(number, dimensions):
        dimensions = tf.unstack(tf.to_int32(dimensions), axis = 0)
        dimensions.append(tf.constant(1, tf.int32))

        out = []
        for i in range(0,len(dimensions)-1):
            out.append(tf.reduce_prod(tf.stack(dimensions[i+1:] , axis = 0) ,axis = 0))
        out = tf.stack(out)
        out = tf.multiply(number,out)
        out = tf.reduce_sum(out, len(number.get_shape())-1)
        return out

    def immutable_scatter_nd_constant_update(inp1, inp2, inp3):
        shape = inp1.get_shape()
        inp1 = tf.to_float(inp1)
        inp1 = tf.reshape(inp1, [-1])
        inp2 = map_index_to_flattened(inp2, shape)
        z1 = tf.to_float(tf.one_hot(inp2, inp1.get_shape()[0]))
        z2 = tf.to_float(tf.reshape(inp3,[-1,1]))
        z3 = tf.multiply(z1,z2)
        update_input = tf.reduce_sum(tf.add(z3, tf.zeros_like(inp1)),axis = 0)

        m1 = tf.reduce_sum(z1, axis = 0)
        m1 = 1-m1
        new_inp1 = tf.multiply(inp1,m1)
        out = tf.add(new_inp1, update_input)
        return tf.reshape(out, shape)

    index_shape_len = inp2.get_shape().as_list()[1]
    shape = inp1.get_shape().as_list()

    if index_shape_len == len(shape):
        return immutable_scatter_nd_constant_update(inp1, inp2, inp3)

    index_shape  = shape[0:index_shape_len]
    slice_shape = shape[index_shape_len:]

    inp1 = tf.to_float(inp1)
    inp1 = tf.reshape(inp1, slice_shape + [-1])
    inp2 = map_index_to_flattened(inp2, index_shape)
    z1 = tf.to_float(tf.one_hot(inp2, inp1.get_shape()[-1]))
    z1 = tf.expand_dims(tf.transpose(z1,[1,0]),axis = 2)
    z2 = tf.to_float(tf.reshape(inp3,[-1]+[sum(slice_shape)]))
    z3 = tf.multiply(z2,z1)
    update_input = tf.reduce_sum(z3,axis = 1)

    m1 = tf.reduce_sum(z1, axis = 1)
    m1 = 1-m1
    inp1 = tf.reshape(inp1, [-1]+[sum(slice_shape)])
    new_inp1 = tf.multiply(inp1,m1)
    out = tf.add(new_inp1, update_input)
    return tf.reshape(out, shape)

#examples with usage
# example 1
inp1 = tf.constant(np.ones([3,4,5,2,2]),tf.int32)
inp2 = tf.constant([[0,0,0],[1,1,1],[0,1,1]])
inp3 = tf.constant([[111,111],[222,222],[4,5]])
inp3 = tf.stack(2*[inp3],axis=2)

x = immutable_scatter_nd(inp1,inp2,inp3)
print x[0,0,0]
print x[1,1,1]
print x[0,1,1]

# example 2
inp1 = tf.constant(np.ones([3,4,5]),tf.int32)
inp2 = tf.constant([[0,0,0],[1,1,1],[0,1,1]])
inp3 = tf.constant([0,2,3])

x = immutable_scatter_nd(inp1,inp2,inp3)
print x[0,0,0]
print x[1,1,1]
print x[0,1,1]