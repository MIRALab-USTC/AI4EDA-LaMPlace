# LaMPlace: Learning to Optimize Cross-Stage Metrics in Macro Placement
This repository contains the implementation of LaMPlace, which Learns a Mask for optimizing cross-stage metrics in macro placement.

Paper: https://openreview.net/pdf?id=YLIsIzC74j

**Note:** This is the latest version, and we're still in the process of organizing and refining the code. Updates will follow. Feel free to reach out with any questions.

# Prerequisites

## Third-party Libraries
This project depends on third-party libraries from [DREAMPlace](https://github.com/limbo018/DREAMPlace). 

To set up the third-party dependencies, run:
```bash
./setup_thirdparty.sh
```

Or manually clone the required libraries:
```bash
git clone https://github.com/limbo018/DREAMPlace.git temp_dreamplace
cp -r temp_dreamplace/thirdparty ./
rm -rf temp_dreamplace
```

For more details, see the [DREAMPlace thirdparty directory](https://github.com/limbo018/DREAMPlace/tree/master/thirdparty).

# Enviornment Setup

cd LaMPlace
pip install -r requirements.txt
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$(pwd)/.. -DPython_EXECUTABLE=$(which python)
make -j16
make install

# Usage

### 1. Preprocess
```
for design in superblue1 superblue3 superblue4 superblue5 superblue7 superblue10 superblue16 superblue18
do
    (
        python dreamplace/benchmarktrans.py test/iccad2015.ot/${design}.json
        python dreamplace/dataflow.py test/iccad2015.ot/${design}.json
        python make_graph.py --benchmark ${design}
    ) &
done

python 01_graph_data_gen.py

python 02_graph_train.py

python 03_gen_l_flow.py --benchmark superblue1

python 04_lamplace.py --benchmark superblue1

python 05_pl4gp.py superblue1 2024
```