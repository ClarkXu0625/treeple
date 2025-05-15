Profile using AMD uProf: 

/opt/AMDuProf_5.0-1479/bin/AMDuProfCLI collect
--output-dir ./profile_result
python3 treeple/stats/tests/test_neofit.py


installation procedure: 

    conda install -c conda-forge joblib threadpoolctl pytest compilers llvm-openmp

    pip install -r build_requirements.txt

    pip install --force -r build_sklearn_requirements.txt

    spin build --clean

    pip install .
    