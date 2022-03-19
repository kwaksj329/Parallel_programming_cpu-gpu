CC=nvcc
CFLAGS=-o
TARGET=convol
SOURCE=2d_convol.cu

all:$(TARGET)

clean:
	rm convol

$(TARGET):$(SOURCE)
	$(CC) $(SOURCE) $(CFLAGS) $(TARGET)
