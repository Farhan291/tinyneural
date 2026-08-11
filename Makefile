# Makefile

CXX      := g++
CXXFLAGS := -O2 -Wall -Wextra -I src
SRC      := src/main.cpp
TARGET   := app

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) src/tensor.hpp src/activation.hpp src/linear.hpp src/losses.hpp src/optimizer.hpp
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) model.bin
