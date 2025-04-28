Steps to recompile:

- pip uninstall treeple
- cd ~/Documents/GitHub/treeple
- meson compile -C builddir
- pip install .


Notes on changes
- treeple/_lib/sklearn/tree/_classes.py, line 395, delete one param, original: sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)