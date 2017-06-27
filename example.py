import chainer, cupy, sys
import numpy as np
from chainer import optimizers, cuda, Variable
from chainer import functions as F
from chainer import links as L
from stream import Stream
import stream as nn

def get_mnist():
	mnist_train, mnist_test = chainer.datasets.get_mnist()
	train_data, train_label = [], []
	test_data, test_label = [], []
	for data in mnist_train:
		train_data.append(data[0])
		train_label.append(data[1])
	for data in mnist_test:
		test_data.append(data[0])
		test_label.append(data[1])
	train_data = np.asanyarray(train_data, dtype=np.float32)
	test_data = np.asanyarray(test_data, dtype=np.float32)
	train_data = (train_data - np.mean(train_data)) / np.std(train_data)
	test_data = (test_data - np.mean(test_data)) / np.std(test_data)
	return (train_data, np.asanyarray(train_label, dtype=np.int32)), (test_data, np.asanyarray(test_label, dtype=np.int32))

def compute_classification_accuracy(model, x, t):
	xp = model.xp
	batches = xp.split(x, len(x) // 100)
	scores = None
	for batch in batches:
		p = F.softmax(model(batch)).data
		scores = p if scores is None else xp.concatenate((scores, p), axis=0)
	return float(F.accuracy(scores, Variable(t)).data)

class OldChain(chainer.Chain):
	def __init__(self):
		super(OldChain, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(None, 1024)
			self.l2 = L.Linear(None, 512)
			self.l3 = L.Linear(None, 256)
			self.l4 = L.Linear(None, 128)
			self.l5 = L.Linear(None, 10)
			self.bn1 = L.BatchNormalization(1024)
			self.bn2 = L.BatchNormalization(512)
			self.bn3 = L.BatchNormalization(256)
			self.bn4 = L.BatchNormalization(128)

	def __call__(self, x):
		out = self.l1(x)
		out = F.leaky_relu(out)
		out = self.bn1(out)
		out = self.l2(out)
		out = F.relu(out)
		out = self.bn2(out)
		out = self.l3(out)
		out = F.elu(out)
		out = self.bn3(out)
		out = self.l4(out)
		out = F.relu(out)
		out = self.bn4(out)
		out = self.l5(out)
		return out

def main():
	# config
	batchsize = 128
	gpu_device = 0

	# specify model
	if True:
		model = Stream()
		model.layer(
			nn.Linear(None, 1024),
			nn.LeakyReLU(),
			nn.Linear(None, 512),
			nn.LeakyReLU(),
			lambda x: x[:, 256:]
		)
		model.layer(
			nn.Residual(
				nn.BatchNormalization(256),
				nn.Linear(None, 128),
				nn.ReLU(),
				nn.BatchNormalization(128),
				nn.Linear(None, 256),
			),
			nn.ELU()
		)
		model.layer(
			nn.Linear(None, 128),
			nn.Maxout(4),
			nn.Linear(None, 10)
		)

	if True:
		model = Stream()
		model.layer(
			nn.Linear(None, 1024),
			nn.ReLU(),
			nn.BatchNormalization(1024),
			nn.Linear(None, 512),
			nn.ReLU(),
			nn.BatchNormalization(512),
			nn.Linear(None, 256),
			nn.ReLU(),
			nn.BatchNormalization(256),
			nn.Linear(None, 128),
			nn.ReLU(),
			nn.BatchNormalization(128),
			nn.Linear(None, 10),
		)

	if False:
		model = Stream(
			nn.Linear(None, 1024),
			nn.ReLU(),
			nn.BatchNormalization(1024),
			nn.Linear(None, 512),
			nn.ReLU(),
			nn.BatchNormalization(512),
			nn.Linear(None, 256),
			nn.ReLU(),
			nn.BatchNormalization(256),
			nn.Linear(None, 128),
			nn.ReLU(),
			nn.BatchNormalization(128),
			nn.Linear(None, 10),
		)

	if gpu_device >= 0:
		model.to_gpu(gpu_device)

	# load MNIST
	mnist_train, mnist_test = get_mnist()

	# init optimizer
	optimizer = optimizers.Adam(alpha=0.001, beta1=0.9)
	optimizer.setup(model)

	train_data, train_label = mnist_train
	test_data, test_label = mnist_test
	if gpu_device >= 0:
		train_data = cuda.to_gpu(train_data)
		train_label = cuda.to_gpu(train_label)
		test_data = cuda.to_gpu(test_data)
		test_label = cuda.to_gpu(test_label)
	train_loop = len(train_data) // batchsize
	train_indices = np.arange(len(train_data))

	# training cycle
	for epoch in xrange(1, 100):
		np.random.shuffle(train_indices)	# shuffle data
		sum_loss = 0

		with chainer.using_config("train", True):
			# loop over all batches
			for itr in xrange(1, train_loop + 1):
				# sample minibatch
				batch_range = np.arange(itr * batchsize, min((itr + 1) * batchsize, len(train_data)))
				x = train_data[train_indices[batch_range]]
				t = train_label[train_indices[batch_range]]

				# to gpu
				if model.xp is cuda.cupy:
					x = cuda.to_gpu(x)
					t = cuda.to_gpu(t)

				logits = model(x)
				loss = F.softmax_cross_entropy(logits, Variable(t))

				# update weights
				optimizer.update(lossfun=lambda: loss)

				if itr % 50 == 0 or itr == train_loop:
					sys.stdout.write("\riteration {}/{}".format(itr, train_loop))
					sys.stdout.flush()
				sum_loss += float(loss.data)

		with chainer.using_config("train", False):
			accuracy_train = compute_classification_accuracy(model, train_data, train_label)
			accuracy_test = compute_classification_accuracy(model, test_data, test_label)

		sys.stdout.write("\r\033[2KEpoch {} - loss: {:.8f} - acc: {:.5f} (train), {:.5f} (test)\n".format(epoch, sum_loss / train_loop, accuracy_train, accuracy_test))
		sys.stdout.flush()

if __name__ == "__main__":
	main()
