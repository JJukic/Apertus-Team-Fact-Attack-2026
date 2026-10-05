.PHONY: run build test benchmark compare download install install-hybrid index web

# Forward all targets to track_2a/Makefile
run:
	$(MAKE) -C track_2a run

test:
	$(MAKE) -C track_2a test

install:
	$(MAKE) -C track_2a install

install-hybrid:
	$(MAKE) -C track_2a install-hybrid

index:
	$(MAKE) -C track_2a index

download:
	$(MAKE) -C track_2a download

benchmark:
	$(MAKE) -C track_2a benchmark

compare:
	$(MAKE) -C track_2a compare

web:
	$(MAKE) -C track_2a web
