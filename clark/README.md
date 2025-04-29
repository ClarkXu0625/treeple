Steps to recompile after making changes in treeple:

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
| treeple/datasets/hyppo.py    | line 565    | Add an additional print line   | print("yesssss"), original: NA |
| treeple/tree/_oblique_splitter.pyx | line 6 | Add an additional profile line | # cython: profile=True, original: NA |
| treeple/_lib/sklearn/tree/_tree.pyx | line 3 | Add an additional profile line | # cython: profile=True, original: NA |
| treeple/_lib/sklearn/tree/_tree.pyx | line 16 | Add an additional profile line | from cython cimport profile, original: NA |
| treeple/_lib/sklearn/tree/_tree.pyx | line 706 | Add an additional profile line | @profile(True), original: NA |
| treeple/meson.build | line 105 | modify False to True| -X profile=False|