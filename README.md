# chaoslang

`chaoslang` is a local pure-Python prototype of the Chaos Language Algorithm (CLA).
It starts with exact-reconstructing symbolic grammar induction:

- **chunks**: repeated sequential blocks represented by productions, e.g. `N0 -> a b c`;
- **categories**: substitutable tokens in similar contexts, represented as category
  occurrences that preserve the observed member, e.g. `M0[x]`.

Sprint-1 priorities are correctness, deterministic behavior, and backend-neutral seams.
The core has no optional runtime dependencies and is tested with stdlib `unittest`.

```python
from chaoslang import CLA

model = CLA.simple(max_iterations=8).fit_symbols("abcabcabc")
assert model.expand() == tuple("abcabcabc")
print(model.state.history)  # ('chunk N0 -> a b c',)
```

Run tests locally:

```bash
python3 -m unittest discover -s tests -v
```
