# Makefile for CUDA WMMA Driver

# Compiler
NVCC = nvcc

# Flags
# -arch=sm_80 is REQUIRED for wmma with bfloat16
# -lcuda links the CUDA driver
CFLAGS = -arch=sm_80 -std=c++17 -O3

# Files
TARGET = wmma_test
SRCS = driver.cu wmma_kernel.cu

# Build Rule
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Clean Rule
clean:
	rm -f $(TARGET)