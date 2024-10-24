import torch


def test_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        # Try to actually use CUDA
        x = torch.rand(5, 3)
        print("Created CPU tensor:")
        print(x)

        # Move to GPU
        if torch.cuda.is_available():
            x = x.cuda()
            print("\nMoved to GPU:")
            print(x)

            # Try a GPU operation
            y = x * 2
            print("\nGPU operation result:")
            print(y)
    else:
        print("CUDA is not available")
        print("\nEnvironment details:")
        import os

        print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")


if __name__ == "__main__":
    test_cuda()
