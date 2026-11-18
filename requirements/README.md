# Python 3.12 dependency locks

`runtime.lock` pins the default offline RAG application; `ci.lock` adds test and
build dependencies; `build.lock` pins the build backend. Files contain resolver
generated package hashes and must be installed with hash checking. Optional
`pretrained` and load-testing dependencies are outside this deployment profile.

Regenerate together from the repository root with Python 3.12 and uv 0.12.23:

```sh
mkdir -p .cache/tmp
export UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/.cache/tmp"
uv pip compile pyproject.toml --python-version 3.12 --generate-hashes --no-header --no-annotate -o requirements/runtime.lock
uv pip compile pyproject.toml --extra ci --python-version 3.12 --generate-hashes --no-header --no-annotate -c requirements/runtime.lock -o requirements/ci.lock
uv pip compile requirements/build.in --python-version 3.12 --generate-hashes --no-header --no-annotate -o requirements/build.lock
```

Run CI and rebuild the container after updates. Original resolution was performed
on macOS ARM64 using cached registry metadata; first Linux installation is still
pending. Python images and optional model downloads have separate versioning.

The optional `models.in` pins Sentence Transformers 5.1.2 and CPU PyTorch 2.9.0.
Run `scripts/lock-models` on the deployment host to resolve a complete Linux lock
with hashes for its architecture. `scripts/orbstack up-models` does this when the
lock is absent. The current agent environment could not resolve PyPI/PyTorch DNS,
so **no completed models.lock is claimed or provided**. The optional Docker target
requires that generated lock; the default offline target remains independent.
Downloaded weights have separately recorded immutable revisions and inference
checks in the private model cache, rather than being bundled in GitHub.
