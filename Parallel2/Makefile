NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80 -std=c++14 -rdc=true --use_fast_math

# Parallel版本的源文件
PARALLEL_SRCS = main.cu globals.cu initial.cu collision.cu streaming.cu macrovar.cu output.cu multi_gpu.cu
PARALLEL_OBJS = $(PARALLEL_SRCS:.cu=.o)
TARGET = rb3d_parallel

all: $(TARGET)

$(TARGET): $(PARALLEL_OBJS)
	$(NVCC) $(NVCCFLAGS) $(PARALLEL_OBJS) -o $(TARGET)

%.o: %.cu rb3d.h parameters.h multi_gpu.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(PARALLEL_OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run 