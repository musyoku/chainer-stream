# chainer.Stream

enables you to create a sequential model easily :muscle: 

# Usage

## Basic

### #1

```
from stream import Stream
import stream as nn

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

y = model(x)
```

### #2

```
from stream import Stream
import stream as nn

model = Stream()
model.layer(
	nn.Linear(None, 1024),
	nn.ReLU(),
	nn.BatchNormalization(1024),
)
model.layer(
	nn.Linear(None, 512),
	nn.ReLU(),
	nn.BatchNormalization(512),
)
model.layer(
	nn.Linear(None, 256),
	nn.ReLU(),
	nn.BatchNormalization(256),
)
model.layer(
	nn.Linear(None, 128),
	nn.ReLU(),
	nn.BatchNormalization(128),
)
model.layer(
	nn.Linear(None, 10),
)

y = model(x)
```

### #3

```
from stream import Stream
import stream as nn

model = Stream()
model.layer(
	nn.Linear(None, 1024),
	nn.ReLU(),
	nn.BatchNormalization(1024),
	nn.Linear(None, 512),
	nn.ReLU(),
	nn.BatchNormalization(512),
)
if False:
	model.layer(
		nn.Linear(None, 256),
		nn.ReLU(),
		nn.BatchNormalization(256),
		nn.Linear(None, 128),
		nn.ReLU(),
		nn.BatchNormalization(128),
	)
model.layer(
	nn.Linear(None, 10),
)

y = model(x)
```

## ResNet

### #1

```
from stream import Stream
import stream as nn

model = Stream()
model.layer(
	nn.Linear(None, 1024),
	nn.ReLU(),
	nn.BatchNormalization(1024),
)
model.layer(
	nn.Residual(
		nn.Linear(None, 128),
		nn.ReLU(),
		nn.BatchNormalization(128),
		nn.Linear(None, 1024),
	),
)
model.layer(
	nn.Linear(None, 256),
	nn.ReLU(),
	nn.BatchNormalization(256),
)
model.layer(
	nn.Residual(
		nn.Linear(None, 32),
		nn.ReLU(),
		nn.BatchNormalization(32),
		nn.Linear(None, 256),
	),
)
model.layer(
	nn.Linear(None, 128),
	nn.ReLU(),
	nn.BatchNormalization(128),
)
model.layer(
	nn.Linear(None, 10),
)

y = model(x)
```

### #2

```
from stream import Stream
import stream as nn

model = Stream(
	nn.Linear(None, 1024),
	nn.ReLU(),
	nn.BatchNormalization(1024),
	nn.Residual(
		nn.Linear(None, 128),
		nn.ReLU(),
		nn.BatchNormalization(128),
		nn.Linear(None, 1024),
	),
	nn.Linear(None, 256),
	nn.ReLU(),
	nn.BatchNormalization(256),
	nn.Residual(
		nn.Linear(None, 32),
		nn.ReLU(),
		nn.BatchNormalization(32),
		nn.Linear(None, 256),
	),
	nn.Linear(None, 128),
	nn.ReLU(),
	nn.BatchNormalization(128),
	nn.Linear(None, 10),
)

y = model(x)
```

## Lambda

### #1

```
from stream import Stream
import stream as nn

model = Stream(
	nn.Linear(None, 1024),
	nn.ReLU(),
	nn.BatchNormalization(1024),
	lambda x: x[:, 512:],
	nn.Linear(None, 256),
	nn.ReLU(),
	nn.BatchNormalization(256),
	lambda x: x[:, 128:],
	nn.Linear(None, 64),
	nn.ReLU(),
	nn.BatchNormalization(64),
	lambda x: x[:, 32:],
	nn.Linear(None, 10),
)

y = model(x)
```