.PHONY: all build test clean setup check ci release bench bench-ops bench-training build-cpp-backend test-cpp-backend

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

# Run CI workflow locally via act (full GPU + Swift support)
ci:
	act push -P macos-14=-self-hosted --workflows .github/workflows/ci.yml

# Build release artifacts locally via act
release:
	act push -P macos-14=-self-hosted --workflows .github/workflows/release.yml \
		--eventpath /dev/stdin <<< '{"ref":"refs/tags/v$(shell grep "^version" Cargo.toml | head -1 | cut -d\" -f2)"}'

# Build release artifacts directly (without act)
release-local:
	@echo "Building wheels..."
	uv run maturin build --release
	uv run maturin build --release --target aarch64-unknown-linux-gnu --zig -i python3.10 python3.11 python3.12 python3.13
	@echo "Building binaries..."
	cargo build -p applegpu-service --release
	cd swift/GPUContainer && swift build -c release
	@echo "Collecting artifacts..."
	mkdir -p dist
	cp target/wheels/*.whl dist/
	cp target/release/gpu-service dist/
	cp swift/GPUContainer/.build/release/gpu-container dist/
	cd dist && shasum -a 256 * > checksums.txt
	@echo "Artifacts in dist/"

build-cpp-backend: build-rust
	cd backend_cpp && uv run python setup.py build_ext --inplace

test-cpp-backend: build-cpp-backend
	uv run pytest python/tests/test_cpp_backend.py -v

bench: bench-ops bench-training

bench-ops: build-python
	uv run python benchmarks/bench_ops.py --sizes 256 512 1024

bench-training: build-python
	uv run python benchmarks/bench_training.py --epochs 3

clean:
	cargo clean
	cd swift && swift package clean
	rm -rf target dist
