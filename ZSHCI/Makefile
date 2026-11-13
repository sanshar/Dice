# Compiler
CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -g -I ../eigen  -DComplex# Enable warnings, optimization, debugging

# Target Executable
TARGET = gosper

# Object Files
OBJS = gosper.o Determinants.o

# Default Rule: Compile Everything
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# Compile gosper.cpp
gosper.o: gosper.cpp
	$(CXX) -c gosper.cpp -o gosper.o $(CXXFLAGS)

# Compile Determinants.cpp
Determinants.o: Determinants.cpp
	$(CXX) -c Determinants.cpp -o Determinants.o $(CXXFLAGS)

# Compile Determinants.cpp
integral.o: integral.cpp
	$(CXX) -c integral.cpp -o integral.o $(CXXFLAGS)

# Clean rule to remove compiled files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: all
	./$(TARGET)
