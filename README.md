# Tensorflow-Hacks

# usage:
- import immutable_scatter_nd_update from scatter_nd_for_immutable_tensor.py
- Just invoke immutable_scatter_nd_update(args) instead of usual tensorflow's scatter_nd_update(args) in your code.
- All arguments are same as scatter_nd_update(args) provided by tensorflow

# examples with usage
  # example 1
  - tf.InteractiveSession()
  - inp1 = tf.constant(np.ones([3,4,5,2,2]),tf.int32)
  - inp2 = tf.constant([[0,0,0],[1,1,1],[0,1,1]])
  - inp3 = tf.constant([[111,111],[222,222],[4,5]])
  - inp3 = tf.stack(2*[inp3],axis=2)
  - x = immutable_scatter_nd_update(inp1,inp2,inp3)
  - print x[0,0,0].eval()
  - print x[1,1,1].eval()
  - print x[0,1,1].eval()
  # example 2
  - inp1 = tf.constant(np.ones([3,4,5]),tf.int32)
  - inp2 = tf.constant([[0,0,0],[1,1,1],[0,1,1]])
  - inp3 = tf.constant([0,2,3])
  - x = immutable_scatter_nd_update(inp1,inp2,inp3)
  - print x[0,0,0].eval()
  - print x[1,1,1].eval()
  - print x[0,1,1].eval()
