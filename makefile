TARGET = simple_sort

NVCC = nvcc

NVCCFLAGS = -I /usr/include/c++/10 -I /usr/lib/cuda/include/

CUFILES = simple_sort.cu

RUNS = "1000000 20 3" "2000000 20 3" "4000000 20 3" "8000000 20 3" "1000000 10 3" "2000000 10 3" "4000000 10 3" "8000000 10 3" "1000000 5 3" "2000000 5 3" "4000000 5 3" "8000000 5 3" "1000000 2 3" "2000000 2 3" "4000000 2 3" "8000000 2 3"

all: $(TARGET)

$(TARGET): $(CUFILES)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run:
	@for params in $(RUNS); do \
		echo "Running ./$(TARGET) $$params"; \
		./$(TARGET) $$params; \
	done

clean:
	rm -f $(TARGET)
