NVCC = nvcc
# NVCCFLAGS = -O3 -arch=sm_86 -std=c++14 -diag-suppress 20044
NVCCFLAGS = -O3 -arch=sm_80 -std=c++14 -rdc=true --use_fast_math

# SRCS = src/main.cu src/globals.cu src/initial.cu src/collision.cu src/streaming_old.cu src/macrovar.cu src/output.cu 
SRCS = src/main.cu src/globals.cu src/initial.cu src/collision.cu src/streaming.cu src/macrovar.cu src/output.cu 
OBJS = $(SRCS:.cu=.o)
TARGET = rb3d

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(TARGET)
	rm -f $(OBJS)

%.o: %.cu src/rb3d.h src/parameters.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)
