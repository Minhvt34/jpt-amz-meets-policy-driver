# CC = gcc
IDIR = INCLUDE
ODIR = OBJ
CFLAGS = -O3 -Wall -pedantic -fPIC -I$(IDIR) -g

# Add CXX for C++ compilation
CXX = g++
# Add CXXFLAGS, inheriting from CFLAGS but allowing C++ specific ones if needed
CXXFLAGS = $(CFLAGS) -std=c++11 $(shell python3 -m pybind11 --includes) $(shell python3-config --includes)

# Python module name (must match PYBIND11_MODULE macro in the wrapper)
MODULE_NAME = lkh_solver
# Output shared object for Python
PYTHON_SO = $(MODULE_NAME)$(shell python3-config --extension-suffix)
# Wrapper source file
WRAPPER_SRC = py_lkh_wrapper.cpp
WRAPPER_OBJ = $(ODIR)/$(WRAPPER_SRC:.cpp=.o)

_DEPS = Hashing.h Heap.h LKH.h Segment.h

DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = Activate.o AddCandidate.o AddTourCandidates.o                   \
       AdjustCandidateSet.o AllocateStructures.o Ascent.o              \
       Between.o CandidateReport.o                                     \
       C.o ChooseInitialTour.o Connect.o CreateCandidateSet.o          \
       Distance.o eprintf.o FindTour.o                                 \
       FixedCandidates.o Flip.o Forbidden.o                            \
       fscanint.o GenerateCandidates.o GetTime.o                       \
       Hashing.o Heap.o Improvement.o IsCandidate.o                    \
       IsPossibleCandidate.o KSwapKick.o LinKernighan.o                \
       MergeTourWithBestTour.o MergeWithTour.o                         \
       LKHmain.o                                                       \
       Minimum1TreeCost.o MinimumSpanningTree.o                        \
       NormalizeNodeList.o                                             \
       Penalty.o printff.o PrepareKicking.o PrintParameters.o          \
       Random.o ReadLine.o                                             \
       ReadParameters.o ReadProblem.o RecordBestTour.o                 \
       RecordBetterTour.o RemoveFirstActive.o RestoreTour.o            \
       SpecialMove.o StatusReport.o Statistics.o StoreTour.o           \
       SymmetrizeCandidateSet.o Trajectory.o WriteTour.o LKH.o

OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.c $(DEPS) | $(ODIR)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: 
	all clean

all:
	$(MAKE) LKH $(PYTHON_SO) # Add python module to 'all' target

# Add a target for the static library
liblkh.a: $(OBJ)
	ar rcs $@ $(OBJ)

LKH: $(OBJ) $(DEPS)
	$(CC) -o ../LKH $(OBJ) $(CFLAGS) -lm

# Rule to compile the C++ wrapper
$(WRAPPER_OBJ): $(WRAPPER_SRC) $(IDIR)/LKH.h | $(ODIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

# Rule to link the Python shared object
$(PYTHON_SO): $(WRAPPER_OBJ) liblkh.a
	$(CXX) -shared -o $@ $(WRAPPER_OBJ) -L. -llkh $(shell python3-config --ldflags) $(CFLAGS)

$(ODIR):
	mkdir -p $(ODIR) # Use -p to avoid error if exists

clean:
	/bin/rm -f ../LKH *~ ._* $(IDIR)/*~ $(IDIR)/._* $(PYTHON_SO) $(WRAPPER_OBJ)
	/bin/rm -fr $(ODIR)

