CXX := g++
CXXFLAGS := -std=c++17 -O3 -fPIC -march=native -fopenmp

# Python / pybind11 include flags
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDE    := $(shell python3-config --includes)
PYTHON_LIB        := $(shell python3-config --ldflags)

# Faiss submodule install path
FAISS_ROOT := extern/faiss/build/install

# Project includes
PROJECT_INCLUDE := -I./include -I./include/zenann

# Aggregate includes
ALL_INCLUDES := $(PYBIND11_INCLUDES) $(PYTHON_INCLUDE) $(PROJECT_INCLUDE) -I$(FAISS_ROOT)/include

# Python linking flags only (faiss linked below)
ALL_LIBS := $(PYTHON_LIB)

# Source files
SOURCES := \
    src/IndexBase.cpp \
    src/IVFFlatIndex.cpp \
    src/KDTreeIndex.cpp \
    src/HNSWIndex.cpp \
    python/zenann_pybind.cpp

# Extension suffix (.so or .cpython-XYm-x86_64-linux-gnu.so, etc.)
EXT_SUFFIX := $(shell python3-config --extension-suffix)
TARGET := build/zenann$(EXT_SUFFIX)

# Platform‐specific linker flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # on macOS use dynamic_lookup
    LDFLAGS := -undefined dynamic_lookup
else
    # on Linux embed rpath to pick up our extern/faiss libfaiss.so
    LDFLAGS := -Wl,-rpath,$$ORIGIN/../extern/faiss/build/install/lib
endif

LDFLAGS  += -fopenmp

.PHONY: all clean prepare

all: prepare $(TARGET)

prepare:
	mkdir -p build

# Build the Python extension, linking against our Faiss
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(ALL_INCLUDES) -shared -o $@ \
	    $(SOURCES) \
	    -L$(FAISS_ROOT)/lib -lfaiss \
	    $(ALL_LIBS) \
	    $(LDFLAGS)

clean:
	rm -rf build
