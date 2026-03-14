.PHONY: all build test clean setup check

all: build test

setup:
	uv tool install maturin
	uv sync

build-swift:
	cd swift && swift build -c release

build-rust: build-swift
	cargo build -p applegpu-core

build-python: build-rust
	uv run maturin develop

build-service: build-rust
	cargo build -p applegpu-service

build: build-swift build-rust build-python build-service

test-rust: build-swift
	cargo test -p applegpu-core

test-swift:
	cd swift && swift test

test-python: build-python
	uv run pytest -v

test: test-rust test-swift test-python

check:
	cargo check --workspace
	cd swift && swift build

clean:
	cargo clean
	cd swift && swift package clean
	rm -rf target
