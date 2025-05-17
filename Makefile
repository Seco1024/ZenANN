CXX := g++
CXXFLAGS := -std=c++17 -O3 -fPIC -march=native
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDE := $(shell python3-config --includes)
PYTHON_LIB := $(shell python3-config --ldflags)
FAISS_ROOT    := extern/faiss/build/install

PROJECT_INCLUDE := -I./include -I./include/zenann

ALL_INCLUDES := $(PYBIND11_INCLUDES) $(PYTHON_INCLUDE) $(PROJECT_INCLUDE)
ALL_LIBS := $(PYTHON_LIB)

ALL_INCLUDES += -I$(FAISS_ROOT)/include
ALL_LIBS     += -L$(FAISS_ROOT)/lib -lfaiss

SOURCES := src/IndexBase.cpp src/IVFFlatIndex.cpp src/KDTreeIndex.cpp python/zenann_pybind.cpp
EXT_SUFFIX := $(shell python3-config --extension-suffix)
TARGET := build/zenann$(EXT_SUFFIX)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS := -undefined dynamic_lookup
endif

.PHONY: all clean prepare

all: prepare $(TARGET)

prepare:
	mkdir -p build

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(ALL_INCLUDES) -shared -o $@ $(SOURCES) $(ALL_LIBS) $(LDFLAGS)

clean:
	rm -rf build
