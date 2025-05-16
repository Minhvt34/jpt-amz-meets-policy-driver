# CC=gcc
CXX = g++
all:
	make -C SRC -j 16
	make -C JPT -j 16
	$(CC) -o get_Length -O3 get_Length.c -Wno-unused-result
clean:
	make -C SRC clean
	make -C JPT clean
	rm -f get_Length
	rm -f *~

# --- Pybind11 Target --- #
# Adjust PYTHON_CFLAGS and PYTHON_LDFLAGS if necessary for your system
# Requires python3-dev and pybind11 pip package
PYBIND11_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_CFLAGS = $(shell python3-config --cflags)
PYTHON_LDFLAGS = $(shell python3-config --ldflags --embed)

# Assuming lkh_bindings.cpp is in LKH-AMZ/ directory
# Assuming liblkh.a is in LKH-AMZ/SRC/
# Assuming LKH.h and other headers are findable via -I LKH-AMZ/SRC/INCLUDE
# Output will be lkh_solver.so (adjust extension for your OS if needed, e.g. .pyd on Windows)
LKH_SOLVER_MODULE = lkh_solver.so

SRC/liblkh.a:
	$(MAKE) -C SRC liblkh.a

lkh_bindings.o: lkh_bindings.cpp SRC/INCLUDE/LKH.h
	$(CXX) -O3 -Wall -shared -std=c++11 -fPIC $(PYTHON_CFLAGS) $(PYBIND11_INCLUDES) -I SRC/INCLUDE -c lkh_bindings.cpp -o lkh_bindings.o

$(LKH_SOLVER_MODULE): lkh_bindings.o SRC/liblkh.a
	$(CXX) -O3 -Wall -shared -std=c++11 -fPIC lkh_bindings.o -LSRC -llkh $(PYTHON_LDFLAGS) -o $(LKH_SOLVER_MODULE)

python_module: $(LKH_SOLVER_MODULE)

clean_python_module:
	rm -f lkh_bindings.o $(LKH_SOLVER_MODULE)
