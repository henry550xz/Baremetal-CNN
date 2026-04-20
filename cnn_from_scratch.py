"""
CNN from Scratch - Student Assignment

In this assignment, you will implement a Convolutional Neural Network from scratch using PyTorch.
You will learn how forward and backward passes work in detail, and gain a deep understanding
of how CNNs operate under the hood.

LEARNING OBJECTIVES:
- Understand forward and backward propagation in neural networks
- Implement convolution and pooling operations manually
- Build a complete CNN architecture
- Implement a training loop with gradient descent
- Gain deep understanding of PyTorch tensor operations

WHAT YOU NEED TO IMPLEMENT:
1. Forward and backward methods for each layer type
2. CNN architecture construction
3. Training loop with gradient computation and parameter updates

NOTE: All __init__ methods are provided - focus on the forward/backward pass logic!
"""
# ============================================================================
# DATA LOADING AND PREPROCESSING (PROVIDED)
# ============================================================================
DRIVE_DIR = "/enter/your/path/to/where/you/want/to/store/the/dataset/files"  # You can change this path. Please note that this should be in your scratch directory
import os
os.makedirs(DRIVE_DIR, exist_ok=True)
import torch
import torch.nn.functional as F
import math
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# Set device and random seed for reproducibility
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
print("Device:", DEVICE)

# MNIST preprocessing: normalize with dataset mean/std
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Download and load MNIST dataset
train_full = datasets.MNIST(DRIVE_DIR, train=True, download=True, transform=transform)
test_set = datasets.MNIST(DRIVE_DIR, train=False, download=True, transform=transform)

# Split training set into train/validation (80/20)
val_split_ratio = 0.2
train_size = int((1 - val_split_ratio) * len(train_full))
val_size = len(train_full) - train_size
train_set, val_set = torch.utils.data.random_split(
    train_full, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

def to_xy(dataset):
    """Convert dataset to tensors and move to device"""
    X = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(DEVICE)
    y = torch.tensor([dataset[i][1] for i in range(len(dataset))], device=DEVICE, dtype=torch.long)
    return X, y

# Convert datasets to tensors
X_train, y_train = to_xy(train_set)
X_val, y_val = to_xy(val_set)
X_test, y_test = to_xy(test_set)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(8, 3))
for ax in axes.ravel():
    i = torch.randint(0, len(train_set), (1,)).item()
    x, y = train_set[i]
    ax.imshow(x.squeeze().cpu(), cmap='gray')
    ax.set_title(f'Label: {y}')
    ax.axis('off')
plt.tight_layout()
plt.show()
# ============================================================================
# HELPER FUNCTIONS (PROVIDED)
# ============================================================================
def zeros_like(x):
    """Create zero tensor with same shape, no gradients"""
    g = torch.zeros_like(x, device=x.device)
    g.requires_grad_(False)
    return g
# ============================================================================
# BASE MODULE CLASS (PROVIDED)
# ============================================================================
class Module:
    """
    Base class for all neural network layers.
    Handles forward pass and stores intermediate values for backward pass.
    """
    def __call__(self, *args):
        """Forward pass - store args and output for backward"""
        self.args = args
        self.out = self.forward(*args)
        return self.out
    
    def forward(self, *args):
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError
    
    def backward(self):
        """Backward pass - calls bwd with stored values"""
        self.bwd(self.out, *self.args)
    
    def bwd(self, *args):
        """Backward pass implementation - to be implemented by subclasses"""
        raise NotImplementedError

# ============================================================================
# ACTIVATION FUNCTIONS - YOUR IMPLEMENTATION IS NEEDED FROM HERE ON OUT
# ============================================================================
class ReLU(Module):
    """
    ReLU activation function: f(x) = max(0, x)
    
    Forward: Apply ReLU element-wise
    Backward: Gradient is 1 where input > 0, else 0
    """
    
    def forward(self, x):
        """
        TODO: Implement the ReLU forward pass.
        
        This function should apply the ReLU activation element-wise.
        
        Args:
            x: Input tensor of any shape.
            
        Returns:
            A tensor of the same shape as x, where every negative value
            in the input is replaced with 0.
        """
        # TODO: YOUR CODE HERE
        pass
    
    def bwd(self, out, x):
        """
        TODO: Implement the ReLU backward pass.
        
        This function calculates the gradient for the input `x`. The gradient from
        the next layer (`out.g`) is passed back only for the elements of `x` that
        were positive during the forward pass. Where `x` was zero or negative,
        the local gradient is zero.
        
        Args:
            out: The output tensor from the forward pass. Its gradient is `out.g`.
            x: The original input tensor from the forward pass.
            
        Sets:
            x.g: The gradient with respect to the input `x`.
        """
        # TODO: YOUR CODE HERE
        pass

class Flatten(Module):
    """
    Flatten layer: reshapes input from [N, C, H, W] to [N, C*H*W]
    Used to connect convolutional layers to fully connected layers
    """
    def forward(self, x):
        """
        TODO: Implement the Flatten forward pass.
        
        This function reshapes a multi-dimensional tensor into a 2D tensor.
        For example, an input of shape `[N, C, H, W]` becomes `[N, C*H*W]`.
        
        Args:
            x: An input tensor, typically with shape [N, C, H, W].
            
        Returns:
            A 2D tensor of shape [N, -1], where -1 automatically calculates
            the product of the other dimensions (C*H*W).
            
        Hint:
        - You MUST save the original shape of `x` (e.g., in `self.shape`) so
          you can reverse this operation in the backward pass.
        """
        # TODO: YOUR CODE HERE
        pass
    
    def bwd(self, out, x):
        """
        TODO: Implement the Flatten backward pass.
        
        This function reshapes the incoming gradient `out.g` back to the original
        shape of the input `x`.
        
        Args:
            out: The output tensor from the forward pass. Its gradient is `out.g`.
            x: The original input tensor from the forward pass.
            
        Sets:
            x.g: The gradient with respect to the input `x`, reshaped to match
                 the original input shape you saved in the forward pass.
        """
        # TODO: YOUR CODE HERE
        pass
# ============================================================================
# FULLY CONNECTED LAYER - YOUR IMPLEMENTATION NEEDED
# ============================================================================
class Linear(Module):
    """
    Fully connected (linear) layer: y = x @ W + b
    
    This layer performs a linear transformation of the input.
    """
    def __init__(self, in_features, out_features):
        """Initialize weights and bias (PROVIDED)"""
        # Xavier/Glorot initialization for better training
        w = (torch.rand(in_features, out_features) * 2 - 1) * math.sqrt(6/(in_features + out_features))
        self.w = w.to(DEVICE)
        self.w.requires_grad_(False)
        self.b = torch.zeros(out_features, device=DEVICE)
    
    def forward(self, x):
        """
        TODO: Implement the Linear forward pass.
        
        This function performs the linear operation `y = x @ W + b`.
        
        Args:
            x: Input tensor of shape [N, in_features].
            
        Returns:
            An output tensor `y` of shape [N, out_features].
        """
        # TODO: YOUR CODE HERE
        pass
    
    def bwd(self, out, x):
        """
        TODO: Implement the Linear backward pass.
        
        This function calculates the gradients for the input, weights, and bias
        using the chain rule.
        
        Args:
            out: The output tensor from the forward pass. Its gradient is `out.g`.
            x: The original input tensor from the forward pass.
            
        Sets:
            x.g (input gradient): How much did `x` affect the loss? (out.g @ w.T)
            self.w.g (weight gradient): How much did `w` affect the loss? (x.T @ out.g)
            self.b.g (bias gradient): How much did `b` affect the loss? (sum of out.g)
        """
        # TODO: YOUR CODE HERE
        pass

# ============================================================================
# CONVOLUTIONAL LAYER - YOUR IMPLEMENTATION NEEDED
# ============================================================================
class Conv2D(Module):
    """
    2D Convolution layer.
    
    This is a complex layer. We will implement it using a trick called `im2col`
    (image-to-column) which turns the convolution operation into a single, large
    matrix multiplication, making it very efficient.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """Initialize conv layer parameters (PROVIDED)"""
        self.cin, self.cout = in_channels, out_channels
        
        # Handle scalar inputs
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        
        self.kH, self.kW = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Xavier initialization for conv weights
        fan_in = in_channels * self.kH * self.kW
        fan_out = out_channels * self.kH * self.kW
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        W = (torch.rand(out_channels, in_channels, self.kH, self.kW) * 2 - 1) * limit
        self.W = W.to(DEVICE)
        self.W.requires_grad_(False)
        self.b = torch.zeros(out_channels, device=DEVICE) if bias else None
    
    def forward(self, x):
        """
        TODO: Implement the Conv2D forward pass using the `im2col` technique.

        Args:
            x: Input tensor of shape [N, C_in, H, W].

        Returns:
            An output tensor of shape [N, C_out, H_out, W_out].

        Procedure:
        1.  Save the input shape `x.shape` for the backward pass.
        2.  Use `F.unfold` to extract all local patches from the input image and
            lay them out as columns in a matrix. This is the `im2col` step.
            Save this unfolded matrix (e.g., in `self.X_unf`) for the backward pass.
        3.  The filters `self.W` have shape [C_out, C_in, kH, kW]. Reshape them
            into a 2D matrix of shape [C_out, C_in * kH * kW].
        4.  The convolution is now just a matrix multiplication between the
            reshaped filters and the unfolded input patches.
        5.  If `self.b` is not `None`, add the bias to the result.
        6.  Calculate the height and width of the output feature map and save
            them (e.g., in `self.out_spatial`) for the backward pass.
        7.  Reshape the result of the matrix multiplication into the final 4D
            output shape: [N, C_out, H_out, W_out].
        """
        N, C, H, W = x.shape
        assert C == self.cin
        self.in_shape = (N, C, H, W)
        
        # TODO: YOUR CODE HERE
        # return out_cols.view(N, self.cout, H_out, W_out)
        pass
    
    def bwd(self, out, x):
        """
        TODO: Implement the Conv2D backward pass.

        This function calculates gradients for the input `x`, weights `self.W`,
        and bias `self.b` by reversing the `im2col` process.

        Args:
            out: The output tensor from the forward pass, with its gradient `out.g`.
            x: The original input tensor from the forward pass.

        Sets:
            self.b.g (bias gradient)
            self.W.g (weight gradient)
            x.g (input gradient)
        
        Procedure:
        - First, reshape the incoming gradient `out.g` to match the dimensions
          from the forward pass matrix multiplication.
        
        1. BIAS GRADIENT (self.b.g): The bias is added to every element in an
           output channel. Its gradient is the sum of the incoming gradients `out.g`
           for that channel, across the batch and spatial dimensions.
        
        2. WEIGHT GRADIENT (self.W.g): This is found by multiplying the incoming
           gradient with the unfolded input patches (`self.X_unf`) you saved.
           This tells you how much each weight contributed to the error. The result
           must be reshaped to match the original shape of `self.W`.
        
        3. INPUT GRADIENT (x.g): This is like a 'transposed convolution'. First,
           multiply the transposed weights with the incoming gradient to get the
           gradient in 'column' format. Then, use `F.fold` to 'stitch' these
           columns back into a gradient image with the same shape as the original
           input `x`. This becomes `x.g`.
        """
        N, C, H, W = self.in_shape
        H_out, W_out = self.out_spatial
        L = H_out * W_out
        
        # Reshape the incoming output gradient `out.g` into a "column" format
        # to match the unfolded data from the forward pass.
        G = out.g.view(N, self.cout, L)
        
        # TODO: YOUR CODE HERE
        pass

# ============================================================================
# POOLING LAYER - YOUR IMPLEMENTATION NEEDED
# ============================================================================
class MaxPool2D(Module):
    """
    2D Max Pooling layer
    
    Takes the maximum value in each pooling window
    """
    
    def __init__(self, kernel_size=2, stride=2):
        """Initialize pooling parameters (PROVIDED)"""
        self.k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
    
    def forward(self, x):
        """
        TODO: Implement the forward pass for Max Pooling.

        This function shrinks the input by taking the maximum value over a sliding window.

        Args:
            x: Input tensor of shape [N, C, H, W].

        Returns:
            The down-sampled 4D output tensor of shape [N, C, H_out, W_out].

        Procedure:
        1.  Save the input shape `x.shape` for the backward pass.
        2.  Use `F.unfold` to extract sliding windows (patches) from `x` and
            arrange them as columns.
        3.  For each column (patch), find the maximum value AND the index of that
            maximum value.
        4.  Save the indices of the max values (e.g., in `self.max_idx`). You will
            need them to know where to send the gradient in the backward pass.
        5.  Calculate the output height and width, and reshape the tensor of
            maximum values into the final output shape [N, C, H_out, W_out].
        """
        # TODO: YOUR CODE HERE
        # return self.max_vals.view(N, C, H_out, W_out)
        pass
    
    def bwd(self, out, x):
        """
        TODO: Implement the backward pass for Max Pooling.

        The main idea is that the gradient only flows back to the input neuron
        that had the maximum value. All other neurons in the window get zero gradient.

        Args:
            out: The output tensor from the forward pass. The gradient `out.g` is available.
            x: The original input tensor from the forward pass.

        Sets:
            x.g: The gradient with respect to the input `x`.

        Procedure:
        1.  Create a tensor of zeros with the same shape as the *unfolded* input
            from the forward pass.
        2.  Use the `self.max_idx` you saved to 'scatter' the incoming gradients
            `out.g` into the correct positions in the zero tensor you just created.
        3.  You now have the gradients in 'column' format. Use `F.fold` to reverse
            the `unfold` operation, converting the gradient columns back into a
            gradient image with the same shape as the original input `x`.
        4.  Set this final gradient image as `x.g`.
        """
        # TODO: YOUR CODE HERE
        pass

# ============================================================================
# LOSS FUNCTION - YOUR IMPLEMENTATION NEEDED
# ============================================================================

class CrossEntropy(Module):
    """
    Cross-entropy loss for classification
    
    Combines softmax and negative log-likelihood
    """
    
    def forward(self, logits, targets):
        """
        TODO: Implement the CrossEntropy forward pass.
        
        This calculates the cross-entropy loss, which is standard for classification.
        It combines Log-Softmax and Negative Log-Likelihood.
        
        Args:
            logits: Raw model scores of shape [N, C].
            targets: True class indices of shape [N].
            
        Returns:
            A single scalar value representing the average loss over the batch.
            
        Procedure:
        1. Save the `targets` tensor for the backward pass.
        2. To prevent numerical errors with large `logits`, subtract the maximum
           logit value from all logits in each sample (for each item in the batch).
        3. Apply the log-softmax formula: `log_softmax = logits - log(sum(exp(logits)))`.
        4. Pick out the log-softmax values corresponding to the correct `targets`
           for each sample. This is the negative log-likelihood.
        5. Return the average of these values across the batch.
        """
        # TODO: YOUR CODE HERE
        # return self.out
        pass
    
    def bwd(self, out, logits, targets):
        """
        TODO: Implement the CrossEntropy backward pass.
        
        This calculates the gradient of the loss with respect to the `logits`.
        The formula is surprisingly simple: `gradient = (softmax(logits) - one_hot_encoding_of_targets)`.
        
        Args:
            out: The scalar loss value from the forward pass.
            logits: The original input logits.
            targets: The original true targets.
            
        Sets:
            logits.g: The gradient with respect to the logits.
            
        Procedure:
        1. Calculate the softmax probabilities from the original `logits`.
        2. For each sample, subtract 1 from the probability corresponding to the
           correct class index.
        3. Since the forward pass returned the *mean* loss, divide the entire
           gradient tensor by the batch size.
        """
        # TODO: YOUR CODE HERE
        pass

# ============================================================================
# MODEL CONTAINER - YOUR IMPLEMENTATION NEEDED
# ============================================================================

class Sequential:
    """
    Container for chaining multiple layers together
    """
    
    def __init__(self, *layers):
        """Initialize with list of layers (PROVIDED)"""
        self.layers = list(layers)
    
    def __call__(self, x):
        """
        TODO: Implement the forward pass through all layers.
        
        This should pass the input `x` through each layer in sequence. The output
        of one layer becomes the input to the next. Try to write this in a generalized manner (Think about iterating through the layers using a loop)
        Args:
            x: The initial input tensor for the network.
        Returns:
            The final output from the last layer.
        """
        # TODO: YOUR CODE HERE
        pass
    
    def backward(self, last_out):
        """
        TODO: Implement the backward pass through all layers.
        
        This performs the backward pass through all layers in *reverse* order.
        It starts with the gradient from the loss function (which is stored in
        `last_out.g`) and propagates it backward from the last layer to the first.
        Args:
            last_out: The final output tensor from the forward pass.
        """
        # TODO: YOUR CODE HERE
        pass

# ============================================================================
# UTILITY FUNCTIONS (PROVIDED)
# ============================================================================

def params_of(net):
    """Extract all trainable parameters from network"""
    params = []
    for layer in net.layers:
        for name in ("W", "b", "w", "bias"):  # conv has W/b, linear has w/b
            if hasattr(layer, name):
                obj = getattr(layer, name)
                if obj is not None:
                    params.append(obj)
    return params

def sgd_step(params, lr):
    """Simple SGD parameter update"""
    with torch.no_grad():
        for p in params:
            p -= lr * p.g

def accuracy(logits, y):
    """Calculate classification accuracy"""
    return (logits.argmax(1) == y).float().mean().item()

# ============================================================================
# CNN ARCHITECTURE - YOUR IMPLEMENTATION NEEDED
# ============================================================================
def build_cnn():
    """
    TODO: Build the CNN architecture.
    
    Construct the neural network by creating an instance of each required layer
    and passing them to the `Sequential` container in the correct order.
    
    Architecture should be:
    1. Conv2D(1 -> 8 channels, kernel=3, stride=1, padding=1)  # 28x28 -> 28x28
    2. ReLU activation
    3. MaxPool2D(kernel=2, stride=2)                           # 28x28 -> 14x14
    4. Conv2D(8 -> 16 channels, kernel=3, stride=1, padding=1) # 14x14 -> 14x14
    5. ReLU activation  
    6. MaxPool2D(kernel=2, stride=2)                           # 14x14 -> 7x7
    7. Flatten                                                 # 16*7*7 -> 784
    8. Linear(784 -> 10)                                       # 784 -> 10 classes
    
    Returns:
        A `Sequential` model containing the specified architecture.
    """
    # TODO: YOUR CODE HERE
    pass

# Create the network and loss function
# TODO: Uncomment these lines after implementing build_cnn()
# net = build_cnn()
# criterion = CrossEntropy()

# ============================================================================
# TRAINING LOOP - YOUR IMPLEMENTATION NEEDED  
# ============================================================================
def train_model(net, criterion, X_train, y_train, X_val, y_val, epochs=5, batch_size=128, lr=0.01):
    """
    TODO: Implement the complete training and validation loop for the network.
    
    Args:
        net: The neural network model.
        criterion: The loss function.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        epochs: Number of times to loop over the entire training set.
        batch_size: Number of samples in each mini-batch.
        lr: Learning rate for the optimizer.
        
    Returns:
        Four lists: train_losses, val_losses, train_accs, val_accs.
        
    Training Loop Structure:
    For each epoch:
    1.  **Shuffle Data:** At the start of each epoch, shuffle the training data
        (`X_train`, `y_train`) to ensure batches are random. `torch.randperm` is
        great for creating shuffled indices.
    2.  **Mini-Batch Loop:** Iterate through the shuffled training data in chunks
        of `batch_size`.
        For each batch:
        a. **Forward Pass:** Get the model's predictions (`logits`) for the batch.
        b. **Compute Loss:** Calculate the loss between `logits` and true labels.
        c. **Zero Gradients:** Before backpropagation, you MUST reset the gradients
           of all model parameters to zero.
        d. **Backward Pass:** Call `criterion.backward()` to start backprop, then
           `net.backward()` to propagate gradients through the model.
        e. **Update Parameters:** Use `sgd_step` to update the model's parameters.
    3.  **Validation:** After each epoch, evaluate the model on the validation set.
        Use `with torch.no_grad():` to ensure no gradients are calculated.
    4.  **Log Metrics:** Print and store the training/validation loss and accuracy.
    """
    # Get all trainable parameters
    params = params_of(net)
    print(f"Total parameters: {sum(p.numel() for p in params)}")
    
    # Lists to store training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # TODO: Implement training loop here
    for epoch in range(1, epochs + 1):
        
        # TODO: Shuffle training indices
        
        # Initialize epoch statistics
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        
        # TODO: Mini-batch training loop
        # for i in range(0, X_train.size(0), batch_size):
        #     - Get batch indices and data
        #     - Forward pass
        #     - Compute loss
        #     - Zero gradients
        #     - Backward pass
        #     - Update parameters
        #     - Accumulate statistics
        
        # TODO: Compute epoch training metrics
        
        # TODO: Validation (inside a `with torch.no_grad():` block)
        #     - Forward pass on validation set
        #     - Compute validation loss and accuracy

        # TODO: Store metrics and print results for the epoch
        
        pass  # Remove this when you implement the loop
    
    return train_losses, val_losses, train_accs, val_accs

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("CNN from Scratch - Assignment")
    print("=" * 50)
    
    # TODO: Uncomment and run after implementing all components
    
    # print("Building CNN...")
    # net = build_cnn()
    # criterion = CrossEntropy()
    
    # print("Starting training...")
    # train_losses, val_losses, train_accs, val_accs = train_model(
    #     net, criterion, X_train, y_train, X_val, y_val,
    #     epochs=5, batch_size=128, lr=0.01
    # )
    
    # # Plot training curves
    # epochs_axis = range(1, len(train_losses) + 1)
    # plt.figure(figsize=(12, 4))
    
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_axis, train_losses, label='train')
    # plt.plot(epochs_axis, val_losses, label='val')
    # plt.title('Cross-Entropy Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_axis, train_accs, label='train')
    # plt.plot(epochs_axis, val_accs, label='val')
    # plt.title('Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()
    
    # # Final test evaluation
    # with torch.no_grad():
    #     test_logits = net(X_test)
    #     test_loss = criterion(test_logits, y_test).item()
    #     test_acc = accuracy(test_logits, y_test)
    # print(f"\nFinal Test Results: loss {test_loss:.4f} | acc {test_acc:.4f}")
    
    print("Please implement the TODO sections to complete the assignment!")