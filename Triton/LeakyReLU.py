import time
import torch
import triton
import triton.language as tl

# Triton Kernel for LeakyReLU forward pass
@triton.jit
def leaky_relu_forward_kernel(
    x_ptr,                # Pointer to input tensor x
    y_ptr,                # Pointer to output tensor y
    N,      # Total number of elements in x
    BLOCK_SIZE: tl.constexpr  # Number of elements processed per kernel instance
):
    # Get the current program (block) ID.
    pid = tl.program_id(0)
    # Compute offsets for this block.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for out-of-bound indices.
    mask = offsets < N
    # Load input values.
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute Leaky ReLU
    y = tl.where(x < 0, x * 0.01, x)
    # Store the result.
    tl.store(y_ptr + offsets, y, mask=mask)


# Triton Kernel for LeakyReLU Backward Pass
@triton.jit
def leaky_relu_backward_kernel(
    x_ptr,                # Pointer to saved input tensor x (from forward pass)
    grad_output_ptr,      # Pointer to gradient of the output
    grad_input_ptr,       # Pointer to store computed gradient with respect to x
    N,                    # Total number of elements in x
    BLOCK_SIZE: tl.constexpr  # Number of elements processed per kernel instance
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load input values and gradient of output.
    x = tl.load(x_ptr + offsets, mask=mask)
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask)
    # Compute gradient of Leaky ReLU:
    grad_in = tl.where(x >= 0, grad_out, grad_out * 0.01)
    tl.store(grad_input_ptr + offsets, grad_in, mask=mask)

# Custom Autograd Function 
class TritonLeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
        """
        Forward pass of the Leaky ReLU activation using the Triton kernel.
        Saves the input tensor for use in the backward pass.
        """
        N = x.numel()
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        # Launch the forward kernel.
        leaky_relu_forward_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
        # Save input tensor for the backward pass.
        ctx.save_for_backward(x)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass computes the gradient of the Leaky ReLU activation.
        """
        x, = ctx.saved_tensors
        grad_output = grad_output.contiguous().to(x.dtype)
        N = x.numel()
        grad_input = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        BLOCK_SIZE = ctx.BLOCK_SIZE
        # Launch the backward kernel.
        leaky_relu_backward_kernel[grid](x_ptr=x, grad_output_ptr=grad_output, grad_input_ptr=grad_input, N=N, BLOCK_SIZE=BLOCK_SIZE)
        # Return the gradient for x and None for BLOCK_SIZE (not a tensor).
        return grad_input, None

# Convenience function to call our custom autograd ReLU.
def triton_leaky_relu(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    return TritonLeakyReLUFunction.apply(x, BLOCK_SIZE)

# ------------------------------------------------------------------------------
# Benchmarking Function
# ------------------------------------------------------------------------------

def benchmark(func, *args, n_warmup=10, n_iters=100):
    """
    Benchmarks a function by running warm-up iterations followed by timed iterations.
    
    Args:
        func (callable): The function to benchmark.
        *args: Arguments to pass to the function.
        n_warmup (int): Number of warm-up iterations.
        n_iters (int): Number of iterations for timing.
    
    Returns:
        float: Average execution time per iteration in milliseconds.
    """
    # Warm-up iterations.
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / n_iters * 1000


# Test and Benchmark the Autograd-Compatible Leaky ReLU

if __name__ == '__main__':
    # Create a random input tensor on the GPU with gradient tracking.
    N = 1024 * 1024  # 1 million elements
    x = torch.randn(N, device='cuda', dtype=torch.float32, requires_grad=True)
    BLOCK_SIZE = 1024

    # Forward pass using our custom Triton Leaky ReLU.
    y_triton = triton_leaky_relu(x, BLOCK_SIZE)
    # Define a dummy loss (sum of outputs) and perform backward pass.
    loss_triton = y_triton.sum()
    loss_triton.backward()
    
    # For validation, compare against PyTorch's built-in Leaky ReLU.
    x_torch = x.detach().clone().requires_grad_()
    y_torch = torch.nn.functional.leaky_relu(x_torch, negative_slope=0.01)
    loss_torch = y_torch.sum()
    loss_torch.backward()

    # Check if the gradients match.
    if torch.allclose(x.grad, x_torch.grad, atol=1e-4):
        print("Success: Triton autograd Leaky ReLU backward matches PyTorch!")
    else:
        print("Error: The gradients do not match.")

    # Benchmark the forward pass.
    triton_time = benchmark(lambda: triton_leaky_relu(x, BLOCK_SIZE))
    torch_time = benchmark(lambda: torch.nn.functional.leaky_relu(x))
    print(f"Average execution time (Forward Pass):")
    print(f"  Triton Leaky ReLU = {triton_time:.3f} ms")
    print(f"  PyTorch Leaky ReLU = {torch_time:.3f} ms")
