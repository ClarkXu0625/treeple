Profile using AMD uProf: 

/opt/AMDuProf_5.0-1479/bin/AMDuProfCLI collect \
  --output-dir ./profile_result \
  python3 treeple/stats/tests/test_neofit.py


/opt/AMDuProf_5.0-1479/bin/AMDuProfCLI collect \
  --output-dir ./profile_result \
  python3 treeple/stats/tests/test_morf.py



installation procedure: 

    conda install -c conda-forge joblib threadpoolctl pytest compilers llvm-openmp

    pip install -r build_requirements.txt

    pip install --force -r build_sklearn_requirements.txt

    spin build --clean

    pip install .



To view full name of profiled functions, run the following command line:

    /opt/AMDuProf_5.0-1479/bin/AMDuProfCLI report \
      -i ./ \
      --category cpu \
      --detail \
      --report-output ./function_report.csv
