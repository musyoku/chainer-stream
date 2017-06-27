import chainer
from chainer import functions
from chainer.links import *

def ReLU():
	return functions.relu

def LeakyReLU():
	return functions.leaky_relu

def ELU():
	return functions.elu

class Maxout():
	def __init__(self, pool_size=0.5):
		self.pool_size = pool_size

	def __call__(self, x):
		return functions.maxout(x, self.pool_size)

class Dropout():
	def __init__(self, ratio=0.5):
		self.ratio = ratio

	def __call__(self, x):
		return functions.dropout(x, self.ratio)

class Residual(object):
	def __init__(self, *layers):
		self.layers = layers

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

class Stream(chainer.Chain):
	def __init__(self, *layers):
		super(Stream, self).__init__()
		assert not hasattr(self, "layers")
		self.layers = []
		if len(layers) > 0:
			self.layer(*layers)

	def layer(self, *layers):
		self.layers += layers
		with self.init_scope():
			for i, layer in enumerate(layers):
				index = i + len(self.layers)

				if isinstance(layer, chainer.Link):
					setattr(self, "layer_%d" % index, layer)

				if isinstance(layer, Residual):
					for _index, _layer in enumerate(layer.layers):
						if isinstance(_layer, chainer.Link):
							setattr(self, "layer_{}_{}".format(index, _index), _layer)

	def __call__(self, x):
		for layer in self.layers:
			y = layer(x)
			if isinstance(layer, Residual):
				y += x
			x = y
		return x