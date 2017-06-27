# chainer.Stream

## Usage

### #1

```
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
```