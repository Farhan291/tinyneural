# Makefile

CXX      := g++
CXXFLAGS := -O2 -Wall -Wextra
SRC      := main.cpp
TARGET   := app

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) tensor.hpp activation.hpp linear.hpp losses.hpp optimizer.hpp
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) model.bin
