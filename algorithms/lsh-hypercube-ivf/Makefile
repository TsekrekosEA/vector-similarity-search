CPP = c++
CPPFLAGS = -Wall -Wextra -I./include

# Default target: build main search binary
all: bin/search

# Help target - show available commands
.PHONY: help
help:
	@echo "Vector Similarity Search - C++ Components"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build search binary (default)"
	@echo "  make all          - Same as default"
	@echo "  make build-all    - Build all binaries (search + benchmark)"
	@echo "  make benchmark    - Build and run comprehensive benchmarks"
	@echo "  make benchmark-mnist - Run MNIST benchmarks only"
	@echo "  make benchmark-sift  - Run SIFT benchmarks only"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-tests  - Remove test outputs"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make clean && make && ./bin/search --help"

# Clean build artifacts
clean:
	rm -f bin/* obj/*
	@echo "✓ Build artifacts cleaned"

# Clean test outputs
clean-tests:
	rm -rf output/tests/* results/*
	@echo "✓ Test outputs cleaned"

# Build all binaries (search + benchmark)
build-all: bin/search bin/benchmark

# Run comprehensive benchmarks for analysis
benchmark: bin/benchmark
	@echo "Running comprehensive benchmarks..."
	@./bin/benchmark

# Run MNIST benchmarks only
benchmark-mnist: bin/benchmark
	@echo "Running MNIST benchmarks..."
	@./bin/benchmark --mnist-only

# Run SIFT benchmarks only
benchmark-sift: bin/benchmark
	@echo "Running SIFT benchmarks..."
	@./bin/benchmark --sift-only

ivfpq_test: bin/ivfpq_test
	./bin/ivfpq_test
	@echo ./bin/ivfpq_test done

silh: bin/ivfflat_trying_sihouettes
	./bin/ivfflat_trying_sihouettes
	@echo done

bin/search: obj/main.o obj/arg_parse.o obj/file_io.o obj/utils.o obj/brute_force.o obj/lsh.o obj/hypercube.o obj/ivfflat.o obj/ivfpq.o
	$(CPP) $(CPPFLAGS) -o $@ $^

bin/benchmark: obj/benchmark.o obj/arg_parse.o obj/file_io.o obj/utils.o obj/brute_force.o obj/lsh.o obj/hypercube.o obj/ivfflat.o obj/ivfpq.o
	$(CPP) $(CPPFLAGS) -o $@ $^

bin/ivfpq_test: obj/ivfpq_test.o obj/ivfflat.o obj/ivfpq.o obj/utils.o
	$(CPP) $(CPPFLAGS) -o $@ $^

bin/ivfflat_trying_sihouettes: obj/ivfflat_trying_sihouettes.o obj/ivfflat.o obj/file_io.o obj/utils.o
	$(CPP) $(CPPFLAGS) -o $@ $^

obj/main.o: src/main.cpp include/arg_parse.hpp include/brute_force.hpp include/lsh.hpp include/hypercube.hpp include/ivfflat.hpp include/ivfpq.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/arg_parse.o: src/arg_parse.cpp include/arg_parse.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/file_io.o: src/file_io.cpp include/file_io.hpp include/data_types.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/utils.o: src/utils.cpp include/utils.hpp include/data_types.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/brute_force.o: src/brute_force.cpp include/brute_force.hpp include/utils.hpp include/data_types.hpp include/arg_parse.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/lsh.o: src/lsh.cpp include/lsh.hpp include/utils.hpp include/data_types.hpp include/arg_parse.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/hypercube.o: src/hypercube.cpp include/hypercube.hpp include/utils.hpp include/data_types.hpp include/arg_parse.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/ivfflat.o: src/ivfflat.cpp include/ivfflat.hpp include/data_types.hpp include/utils.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/ivfpq.o: src/ivfpq.cpp include/ivfpq.hpp include/ivfflat.hpp include/data_types.hpp include/utils.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/ivfpq_test.o: src/ivfpq_test.cpp include/ivfpq.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/ivfflat_trying_sihouettes.o: src/ivfflat_trying_silhouettes.cpp include/ivfpq.hpp include/file_io.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

obj/benchmark.o: src/benchmark.cpp include/file_io.hpp include/brute_force.hpp include/lsh.hpp include/hypercube.hpp include/ivfflat.hpp include/ivfpq.hpp include/utils.hpp include/data_types.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<
