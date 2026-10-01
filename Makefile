.PHONY: run build test benchmark compare download install

# Forward all targets to track_2a/Makefile
run:
	$(MAKE) -C track_2a run

install:
	$(MAKE) -C track_2a install

download:
	$(MAKE) -C track_2a download

benchmark:
	$(MAKE) -C track_2a benchmark

compare:
	$(MAKE) -C track_2a compare
