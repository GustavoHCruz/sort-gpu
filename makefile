TARGET = simple_sort

NVCC = nvcc

NVCCFLAGS = --std=c++11 -I /usr/include/c++/10 -I /usr/lib/cuda/include/

CUFILES = simple_sort.cu

RUNS = "1000000 10000 30" "2000000 10000 30" "4000000 10000 30" "8000000 10000 30"

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
