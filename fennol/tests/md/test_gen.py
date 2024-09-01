import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Mac architecture: {sys.platform}")

# Test randn_like with generator
try:
    generator = torch.Generator()
    test_tensor = torch.randn(3, 3)
    result = torch.randn_like(test_tensor, generator=generator)
    print("torch.randn_like with generator argument works correctly")
except TypeError as e:
    print(f"Error using torch.randn_like with generator: {e}")

# Test manual seed
try:
    generator = torch.Generator()
    generator.manual_seed(42)
    random_number = torch.rand(1, generator=generator)
    print(f"Random number with seeded generator: {random_number.item()}")
except Exception as e:
    print(f"Error using generator.manual_seed: {e}")