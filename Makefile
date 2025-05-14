CC = riscv64-unknown-elf-gcc

MNIST_CFLAGS = \
  -I./../../tests/saturn_tests/env \
  -I./../../tests/saturn_tests/common \
  -I./src \
  -I./layers \
  -I./models/mnist_models \
  -I./models/mnist \
  -DPREALLOCATE=1 \
  -mcmodel=medany -static -O2 -g -ffast-math \
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -march=rv64gcv_zfh_zvfh -mabi=lp64d -std=gnu99

MNIST_LDFLAGS = \
  -static -nostdlib -nostartfiles -lm -lgcc \
  -T ./../../tests/saturn_tests/common/test.ld

# list all your .c/.S files automatically
SRC_DIRS := \
  layers/fully_connected \
  layers/activation \
  layers/conv2D \
  layers/pooling \
  layers/quantization \
  models/mnist_models \
  models/mnist_models/mnist \
  src/matmul \
  src/conv2d \
  src/transpose \
  ../../tests/saturn_tests/common \
  ../../tests/saturn_tests/common/ara

SRCS := $(foreach d,$(SRC_DIRS),$(wildcard $(d)/*.c $(d)/*.S))

MNIST_OUT = mnist.riscv



MNIST_CNN_CFLAGS = \
  -I./../../tests/saturn_tests/env \
  -I./../../tests/saturn_tests/common \
  -I./src \
  -I./layers \
  -I./models/mnist_models \
  -I./models/mnist_models/mnist_cnn \
  -DPREALLOCATE=1 \
  -mcmodel=medany -static -O2 -g -ffast-math \
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -march=rv64gcv_zfh_zvfh -mabi=lp64d -std=gnu99

MNIST_CNN_LDFLAGS = \
  -static -nostdlib -nostartfiles -lm -lgcc \
  -T ./../../tests/saturn_tests/common/test.ld

# list all your .c/.S files automatically
MNIST_CNN_SRC_DIRS := \
  layers/fully_connected \
  layers/activation \
  layers/conv2D \
  layers/pooling \
  layers/quantization \
  models/mnist_models \
  models/mnist_models/mnist_cnn \
  src/matmul \
  src/conv2d \
  src/transpose \
  ../../tests/saturn_tests/common \
  ../../tests/saturn_tests/common/ara

MNIST_CNN_SRCS := $(foreach d,$(MNIST_CNN_SRC_DIRS),$(wildcard $(d)/*.c $(d)/*.S $(d)/*.h))

MNIST_CNN_OUT = mnist_cnn.riscv


MNIST_QCNN_CFLAGS = \
  -I./../../tests/saturn_tests/env \
  -I./../../tests/saturn_tests/common \
  -I./src \
  -I./layers \
  -I./models/mnist_models \
  -I./models/mnist_models/mnist_cnn_quant \
  -DPREALLOCATE=1 \
  -mcmodel=medany -static -O2 -g -ffast-math \
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -march=rv64gcv_zfh_zvfh -mabi=lp64d -std=gnu99

MNIST_QCNN_LDFLAGS = \
  -static -nostdlib -nostartfiles -lm -lgcc \
  -T ./../../tests/saturn_tests/common/test.ld

# list all your .c/.S files automatically
MNIST_QCNN_SRC_DIRS := \
  layers/fully_connected \
  layers/activation \
  layers/conv2D \
  layers/pooling \
  layers/quantization \
  models/mnist_models \
  models/mnist_models/mnist_cnn_quant \
  src/matmul \
  src/conv2d \
  src/transpose \
  ../../tests/saturn_tests/common \
  ../../tests/saturn_tests/common/ara

MNIST_QCNN_SRCS := $(foreach d,$(MNIST_QCNN_SRC_DIRS),$(wildcard $(d)/*.c $(d)/*.S $(d)/*.h))

MNIST_QCNN_OUT = mnist_cnn_quant.riscv



MNIST_Q_CFLAGS = \
  -I./../../tests/saturn_tests/env \
  -I./../../tests/saturn_tests/common \
  -I./src \
  -I./layers \
  -I./models/mnist_models \
  -I./models/mnist_quant \
  -DPREALLOCATE=1 \
  -mcmodel=medany -static -O2 -g -ffast-math \
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -march=rv64gcv_zfh_zvfh -mabi=lp64d -std=gnu99

MNIST_Q_LDFLAGS = \
  -static -nostdlib -nostartfiles -lm -lgcc \
  -T ./../../tests/saturn_tests/common/test.ld

# list all your .c/.S files automatically
MNIST_Q_SRC_DIRS := \
  layers/fully_connected \
  layers/activation \
  layers/conv2D \
  layers/pooling \
  layers/quantization \
  models/mnist_models \
  models/mnist_models/mnist_quant \
  src/matmul \
  src/conv2d \
  src/transpose \
  ../../tests/saturn_tests/common \
  ../../tests/saturn_tests/common/ara

MNIST_Q_SRCS := $(foreach d,$(MNIST_Q_SRC_DIRS),$(wildcard $(d)/*.c $(d)/*.S $(d)/*.h))

MNIST_Q_OUT = mnist_quant.riscv


MNIST_Q2_CFLAGS = \
  -I./../../tests/saturn_tests/env \
  -I./../../tests/saturn_tests/common \
  -I./src \
  -I./layers \
  -I./models/mnist_models \
  -I./models/mnist_quant2 \
  -DPREALLOCATE=1 \
  -mcmodel=medany -static -O2 -g -ffast-math \
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -march=rv64gcv_zfh_zvfh -mabi=lp64d -std=gnu99

MNIST_Q2_LDFLAGS = \
  -static -nostdlib -nostartfiles -lm -lgcc \
  -T ./../../tests/saturn_tests/common/test.ld

# list all your .c/.S files automatically
MNIST_Q2_SRC_DIRS := \
  layers/fully_connected \
  layers/activation \
  layers/conv2D \
  layers/pooling \
  layers/quantization \
  models/mnist_models \
  models/mnist_models/mnist_quant2 \
  src/matmul \
  src/conv2d \
  src/transpose \
  ../../tests/saturn_tests/common \
  ../../tests/saturn_tests/common/ara

MNIST_Q2_SRCS := $(foreach d,$(MNIST_Q2_SRC_DIRS),$(wildcard $(d)/*.c $(d)/*.S $(d)/*.h))

MNIST_Q2_OUT = mnist_quant2.riscv


MNIST_Q2_MC_CFLAGS = \
  -I./../../tests/saturn_tests/env \
  -I./src/common_multicore \
  -I./src \
  -I./layers \
  -I./models/mnist_models \
  -I./models/mnist_quant2_mc \
  -DPREALLOCATE=1 \
  -mcmodel=medany -static -O2 -g -ffast-math \
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -march=rv64gcv_zfh_zvfh -mabi=lp64d -std=gnu99

MNIST_Q2_MC_LDFLAGS = \
  -static -nostdlib -nostartfiles -lm -lgcc \
  -T ./src/common_multicore/test.ld

# list all your .c/.S files automatically
MNIST_Q2_MC_SRC_DIRS := \
  layers/fully_connected \
  layers/activation \
  layers/conv2D \
  layers/pooling \
  layers/quantization \
  models/mnist_models \
  models/mnist_models/mnist_quant2_mc \
  src/matmul \
  src/conv2d \
  src/transpose \
  src/common_multicore \
  ../../tests/saturn_tests/common/ara

MNIST_Q2_MC_SRCS := $(foreach d,$(MNIST_Q2_MC_SRC_DIRS),$(wildcard $(d)/*.c $(d)/*.S $(d)/*.h))

MNIST_Q2_MC_OUT = mnist_quant2_mc.riscv



MNIST_CNN_MC_CFLAGS = \
  -I./../../tests/saturn_tests/env \
  -I./src/common_multicore \
  -I./src \
  -I./layers \
  -I./models/mnist_models \
  -I./models/mnist_models/mnist_cnn_mc \
  -DPREALLOCATE=1 \
  -mcmodel=medany -static -O2 -g -ffast-math \
  -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -march=rv64gcv_zfh_zvfh -mabi=lp64d -std=gnu99

MNIST_CNN_MC_LDFLAGS = \
  -static -nostdlib -nostartfiles -lm -lgcc \
  -T ./src/common_multicore/test.ld

# list all your .c/.S files automatically
MNIST_CNN_MC_SRC_DIRS := \
  layers/fully_connected \
  layers/activation \
  layers/conv2D \
  layers/pooling \
  layers/quantization \
  models/mnist_models \
  models/mnist_models/mnist_cnn_mc \
  src/matmul \
  src/conv2d \
  src/transpose \
  src/common_multicore \
  ../../tests/saturn_tests/common/ara

MNIST_CNN_MC_SRCS := $(foreach d,$(MNIST_CNN_MC_SRC_DIRS),$(wildcard $(d)/*.c $(d)/*.S $(d)/*.h))

MNIST_CNN_MC_OUT = mnist_cnn_mc.riscv





.PHONY: all clean

all: $(MNIST_OUT)

# compile & link in one step; no .o files emitted
$(MNIST_OUT): $(SRCS)
	$(CC) $(MNIST_CFLAGS) -o $@ $^ $(MNIST_LDFLAGS)


mnist_cnn: $(MNIST_CNN_OUT)

# compile & link in one step; no .o files emitted
$(MNIST_CNN_OUT): $(MNIST_CNN_SRCS)
	$(CC) $(MNIST_CNN_CFLAGS) -o $@ $^ $(MNIST_CNN_LDFLAGS)


mnist_cnn_quant: $(MNIST_QCNN_OUT)

# compile & link in one step; no .o files emitted
$(MNIST_QCNN_OUT): $(MNIST_QCNN_SRCS)
	$(CC) $(MNIST_QCNN_CFLAGS) -o $@ $^ $(MNIST_QCNN_LDFLAGS)



mnist_cnn_mc: $(MNIST_CNN_MC_OUT)

# compile & link in one step; no .o files emitted
$(MNIST_CNN_MC_OUT): $(MNIST_CNN_MC_SRCS)
	$(CC) $(MNIST_CNN_MC_CFLAGS) -o $@ $^ $(MNIST_CNN_MC_LDFLAGS)


mnist_quant: $(MNIST_Q_OUT)

# compile & link in one step; no .o files emitted
$(MNIST_Q_OUT): $(MNIST_Q_SRCS)
	$(CC) $(MNIST_Q_CFLAGS) -o $@ $^ $(MNIST_Q_LDFLAGS)


mnist_quant2: $(MNIST_Q2_OUT)

# compile & link in one step; no .o files emitted
$(MNIST_Q2_OUT): $(MNIST_Q2_SRCS)
	$(CC) $(MNIST_Q2_CFLAGS) -o $@ $^ $(MNIST_Q2_LDFLAGS)

mnist_quant2_mc: $(MNIST_Q2_MC_OUT)

# compile & link in one step; no .o files emitted
$(MNIST_Q2_MC_OUT): $(MNIST_Q2_MC_SRCS)
	$(CC) $(MNIST_Q2_MC_CFLAGS) -o $@ $^ $(MNIST_Q2_MC_LDFLAGS)



clean:
	rm -f $(MNIST_OUT) $(MNIST_CNN_OUT) $(MNIST_Q_OUT) $(MNIST_Q2_OUT) $(MNIST_Q2_MC_OUT) $(MNIST_QCNN_OUT)
