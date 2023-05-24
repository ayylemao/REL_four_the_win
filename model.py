import torch as th
from torch import nn


class CoolModel(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        
        output_dim = 7
        # input_dim = 42 not necessary?
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=1) # output shape = 3x4
        self.a1 = nn.ReLU()
        
        self.linear2 = nn.Linear(in_features=4, out_features=output_dim) # output 3x7
        self.a2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), stride=1) # output shape = 3x4
        self.a3 = nn.Softmax(7)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.conv1(x)
        x = self.a1(x)
        x = self.linear2(x)
        x = self.a2(x)
        x = self.conv3(x)
        x = self.a3()
        return x


def get_params(model):
    """
    Get number of params.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)


# Create Tensors to hold input and outputs.
x = th.randint(low=-1, high=2, size=(6, 7))
x = th.rand(size=(6, 7)).unsqueeze(dim=0)
x = x.unsqueeze(dim=0).float() # for batch dim
y = th.zeros(7).unsqueeze(dim=0).unsqueeze(dim=0)

# Construct our model by instantiating the class defined above
model = CoolModel()
model.a1(model.linear2(model.a1(model.conv1(x))))

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined 
# with torch.nn.Parameter) which are members of the model.
criterion = th.nn.MSELoss(reduction='sum')
optimizer = th.optim.SGD(model.parameters(), lr=0.01)
for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


