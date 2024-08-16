TARGET = simple_sort

NVCC = nvcc

NVCCFLAGS = -I /usr/include/c++/10 -I /usr/lib/cuda/include/

CUFILES = simple_sort.cu

RUNS = "100000 20 3" "200000 20 3" "400000 20 3" "800000 20 3" "100000 10 3" "200000 10 3" "400000 10 3" "800000 10 3" "100000 5 3" "200000 5 3" "400000 5 3" "800000 5 3" "100000 2 3" "200000 2 3" "400000 2 3" "800000 2 3"

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
