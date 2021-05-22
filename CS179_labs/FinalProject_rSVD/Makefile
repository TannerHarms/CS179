# CS 179 Lab 4 Unix Makefile
# Written by Loko Kung, 2018
# Edited by Tyler Port, 2018
# Edited by George Stathopoulos, 2019

# Input Names
CPP_FILES = rSVD.cpp

# Directory names
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# ------------------------------------------------------------------------------

# CUDA path, compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# # OS-architecture specific flags
# ifeq ($(OS_SIZE),32)
# 	NVCC_FLAGS := -m32
# else
# 	NVCC_FLAGS := -m64
# endif
# NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
#              --expt-relaxed-constexpr
# NVCC_INCLUDE =
# NVCC_LIBS = 
# NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
# 		-gencode arch=compute_35,code=sm_35 \
# 		-gencode arch=compute_50,code=sm_50 \
# 		-gencode arch=compute_52,code=sm_52 \
# 		-gencode arch=compute_60,code=sm_60 \
# 		-gencode arch=compute_61,code=sm_61 \
# 		-gencode arch=compute_61,code=compute_61


# ------------------------------------------------------------------------------
# Eigen Path
EIGEN_INC_PATH = /usr/include/eigen3

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
# CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++11 -pthread
INCLUDE = -I$(CUDA_INC_PATH) -I$(EIGEN_INC_PATH)
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas -lsndfile -lcusolver

# ------------------------------------------------------------------------------
# Object files
# ------------------------------------------------------------------------------

# C++ Object Files
CPP_OBJ = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CPP_FILES)))

# ------------------------------------------------------------------------------
# Make Rules 
# ------------------------------------------------------------------------------

# Top level rules
all: rSVD

rSVD: $(CPP_OBJ) 
	$(GPP) $(FLAGS) -o $(BINDIR)/$@ $(INCLUDE) $^ $(LIBS) 

# Compile C++ Source Files
$(CPP_OBJ): $(OBJDIR)/%.o : $(SRCDIR)/%
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

# Clean everything including temporary Emacs files
clean:
	rm -f rSVD *.o *~

.PHONY: clean
