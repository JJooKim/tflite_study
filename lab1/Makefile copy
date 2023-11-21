CC := g++
CFLAGS := -std=c++11 -I/usr/include 
INCLUDEPATHS := -I${HOME}/tensorflow
LDFLAGS := -pthread -ltensorflow-lite -lflatbuffers -ldl
LDPATH := -L${HOME}/tensorflow/tensorflow/lite/tools/make/gen/bbb_armv7l/lib\
	  -L${HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build

# OpenCV flags
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LDFLAGS := $(shell pkg-config --libs opencv4)

SRCS := assignment.cc
OBJS := $(SRCS:.cc=.o)
EXEC := assignment

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) $(OPENCV_CFLAGS) -o $@ $^ $(LDPATH) $(LDFLAGS) $(OPENCV_LDFLAGS)

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDEPATHS) $(OPENCV_CFLAGS) -c $< -o $@   

clean : 
	rm -f $(OBJS) $(EXEC)
