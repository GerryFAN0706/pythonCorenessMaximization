# CorenessMaximization

### Datasets
Graphs are stored as plain-text edge lists with the first line containing `n m`. You can place additional datasets into the `dataset/` directory. The algorithms renumber vertices internally, so IDs in the file can be arbitrary integers.

### Python Implementation
1. **Prerequisites**
   - Python 3.9+
   - Optional but recommended: virtual environment.

2. **Install Dependencies**
   The Python port relies only on the standard library; no extra packages are required.

3. **Run**
   ```powershell
   # From the project root on Windows PowerShell
   python -m python.main dataset/twitter_copen.txt 50 output/python_insert_b50_mode2.txt 0 2
   ```
   Parameters mirror the C++ executable. Example meanings:
   - `budget=1`: allow one edge to be inserted.
   - `check=0`: run the heuristic.
   - `mode=0`: vertex-based strategy.

4. **Verification**
   After producing an insert file, you can validate the coreness improvement:
   ```powershell
   python -m python.main dataset/twitter_copen.txt 0 output/python_insert.txt 1 0
   ```
   - `check=1` recomputes coreness, loads the inserted edges, and reports the gain.

5. **Performance Notes**
   The Python port reproduces the greedy logic but runs slower than the optimized C++ build. Expect multi-minute execution on large graphs even for small budgets.

### Output Files
- Inserted edges: Tab-separated `u	v` lines using original vertex IDs.
- Console logs: Coreness statistics and runtime per round.

