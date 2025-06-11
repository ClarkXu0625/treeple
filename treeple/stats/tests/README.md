Profile using AMD uProf: 

/opt/AMDuProf_5.0-1479/bin/AMDuProfCLI collect \
  --output-dir ./profile_result \
  python3 treeple/stats/tests/test_neofit.py


/opt/AMDuProf_5.0-1479/bin/AMDuProfCLI collect \
  --output-dir ./profile_result \
  python3 treeple/stats/tests/test_morf.py

/opt/AMDuProf_5.0-1479/bin/AMDuProfCLI collect \
  --output-dir ./profile_result \
  python3 treeple/stats/time_comparison2.py

installation procedure: 

    conda install -c conda-forge joblib threadpoolctl pytest compilers llvm-openmp

    pip install -r build_requirements.txt

    pip install --force -r build_sklearn_requirements.txt

    spin build --clean

    pip install .


Other packages to install to run test_neofit

    pip install tqdm statsmodels matplotlib shap lime


To view full name of profiled functions, run the following command line:

    /opt/AMDuProf_5.0-1479/bin/AMDuProfCLI report \
      -i ./ \
      --category cpu \
      --detail \
      --report-output ./function_report.csv


Intel vtune Profiler


source /opt/intel/oneapi/vtune/latest/env/vars.sh

cd /opt/intel/oneapi/vtune/latest/sepdk/src
sudo ./insmod-sep -q

sudo apt install linux-headers-$(uname -r)

cd /opt/intel/oneapi/vtune/latest/sepdk/src
sudo ./build-driver -ni


AWS env setup

    # Download and install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    # follow prompts and restart shell
    source ~/.bashrc

    conda create -n treeple python=3.10 
    conda activate treeple

    sudo apt update
    sudo apt install -y build-essential