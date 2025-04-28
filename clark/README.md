Steps to recompile:

- pip uninstall treeple
- cd ~/Documents/GitHub/treeple
- meson compile -C builddir
- pip install .


Notes on changes in source code
<!-- - treeple/_lib/sklearn/tree/_classes.py, line 395, delete one param, original: sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)
- treeple/datasets/hyppo.py, line 565, an additional print line -->

| Relative path | line number | comment | original |
|-----------------|-----------------|-----------------|-----------------|
| treeple/_lib/sklearn/tree/_classes.py  |  line 395  | Delete one param | sample_weight = _check_sample_weight(sample_weight, X, DOUBLE) |
| treeple/datasets/hyppo.py    | line 565    | An additional print line   | NA |