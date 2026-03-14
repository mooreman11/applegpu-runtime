.PHONY: all build test clean setup check

all: build test

# Install tools and deps, build native extension
setup:
	uv tool install maturin
	uv sync

build-rust:
	cargo build --workspace

build-swift:
	cd swift && swift build

build-python:
	uv sync

build: build-rust build-swift build-python

test-rust:
	cargo test --workspace

test-swift:
	cd swift && swift test

test-python:
	uv run pytest -v

test: test-rust test-swift test-python

# Quick compile check (no cross-layer linking)
check:
	cargo check --workspace
	cd swift && swift build

clean:
	cargo clean
	cd swift && swift package clean
	rm -rf target
