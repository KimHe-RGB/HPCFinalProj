CXX = g++
CXXFLAGS = -std=c++11 -O3 -I/usr/include/eigen3
LDFLAGS =
TARGET = heat_equation_parareal
SRC = heat_equation_parareal.cpp 

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
