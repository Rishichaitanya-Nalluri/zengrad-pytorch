
## Installation

To install ZenGrad, run the following command:

```bash
pip install zengrad-pytorch
```

## Implementation

```python
import torch
import torch.nn as nn
from zengrad_pytorch import ZenGrad  # Custom optimizer

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Instantiate the model
model = SimpleModel()

# Define loss function
criterion = nn.CrossEntropyLoss()

# Instantiate optimizer
optimizer = ZenGrad(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4,
    momentum=0.9,
    epsilon=1e-8,
) # <---- In This Way      

# Model summary (optional)
print(model)
```

## License

ZenGrad_TensorFlow is released under the [Apache License 2.0](https://github.com/Rishichaitanya-Nalluri/zengrad-pytorch/blob/main/LICENSE).
