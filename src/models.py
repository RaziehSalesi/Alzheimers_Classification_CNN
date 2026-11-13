
import torch
import torch.nn as nn

# Model Test1: Proposed
class CNN_Mode_Proposed(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flat = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

        # Xavier initialization for weights
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight, 1)
                layer.bias.data.zero_()

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, 1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flat(x)
        x = self.classifier(x)

        return x


# Model Test2: 

class CNN_Model_Test2(nn.Module):
  def __init__(self):
    super().__init__()
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(1,32, kernel_size=3 , padding=1),     # we want to extract 32 feature from this
        nn.BatchNorm2d(32),
        nn.ReLU(),                                      # Input: (batch_size, 1, 64, 64), Output: (batch_size, 32, 64, 64)
        #nn.Dropout(0.2),
        nn.MaxPool2d(kernel_size=2, stride=2),          # Output: (batch_size, 32, 32, 32)
        nn.Conv2d(32, 32, kernel_size=3 , padding=1),   # Output: (batch_size, 32, 32, 32)
        nn.BatchNorm2d(32),
        nn.ReLU(),
        #nn.Dropout(0.2),
        nn.MaxPool2d(kernel_size=2, stride=2)    # Output: (batch_size, 32, 16, 16)
    )

    self.flat = nn.Flatten()   # Input: (batch_size, 32, 16, 16), Output: (batch_size, 32*16*16)

    self.classifier = nn.Sequential(
        nn.Linear(32*16*16, 128),  # Input: (batch_size, 8192), Output: (batch_size, 128)
        nn.ReLU(),
        nn.Linear(128, 2),           # Input: (batch_size, 128), Output: (batch_size, 2)
    )
    
    
    # Xavier initialization for weights
    for layer in self.feature_extractor:
        if isinstance(layer, nn.Conv2d):
           torch.nn.init.xavier_uniform_(layer.weight,1)
           layer.bias.data.zero_()

    for layer in self.classifier:
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight,1)
            layer.bias.data.zero_()


  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.flat(x)
    x = self.classifier(x)
    return x



# Model test3
class CNN_Model_Test3(nn.Module):
  def __init__(self):
    super().__init__()
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(1,32, kernel_size=3 , padding=1),     # we want to extract 32 feature from this
        nn.BatchNorm2d(32),
        nn.ReLU(),                                      # Input: (batch_size, 1, 64, 64), Output: (batch_size, 32, 64, 64)
        #nn.Dropout(0.5),
        nn.Conv2d(32,32, kernel_size=3 , padding=1),     # we want to extract 32 feature from this
        nn.BatchNorm2d(32),
        nn.ReLU(),                                      # Input: (batch_size, 32, 64, 64), Output: (batch_size, 32, 64, 64)
        #nn.Dropout(0.5),
        nn.MaxPool2d(kernel_size=2, stride=2),          # Output: (batch_size, 32, 32, 32)
    )

    self.flat = nn.Flatten()   # Input: (batch_size, 32, 16, 16), Output: (batch_size, 32*32*32*)

    self.classifier = nn.Sequential(
        nn.Linear(32*32*32, 128),  # Input: (batch_size, 32*32*32), Output: (batch_size, 128)
        nn.ReLU(),
        nn.Linear(128, 64),  # Input: (batch_size, 128), Output: (batch_size, 64)
        nn.ReLU(),
        nn.Linear(64, 2),           # Input: (batch_size, 64), Output: (batch_size, 2)
       
    )
   
    # Xavier initialization for weights
    for layer in self.feature_extractor:
        if isinstance(layer, nn.Conv2d):
           torch.nn.init.xavier_uniform_(layer.weight,1)
           layer.bias.data.zero_()

    for layer in self.classifier:
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight,1)
            layer.bias.data.zero_()

  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.flat(x)
    x = self.classifier(x)
    return x