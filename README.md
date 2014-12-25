ANCOM
=====

Analysis of Composition of Microbiomes

ANCOM is a novel statistical test that accounts for OTU dependencies when calculating group signficance.
This software makes use of GPU-accelerated matrix multiplication and permutation tests to 
allow for these statistical tests to complete within a reasonable amount of time.


Installation
============
This software requires the following dependencies
```
pandas
pyopencl
pyviennacl
numpy >= 1.7
scipy >= 0.13.0
```

If pip installed, the following command can be run
```
pip install -r requirements.txt
```
Then the following commands can be run to ANCOM
```
python setup.py build
python setup.py install
```
To make things easier, I recommend checking out [virtualenv](https://virtualenv.readthedocs.org/en/latest/)
to mitigate the dependency issues.

Getting Started
===============
After installing this software, I recommend running the unittests to make sure that everything is working.
To run ANCOM from the command, run the following command.
```
python bin/run_ancom.py 
  --otu-table=data/otu_test.txt 
  --meta-data=data/meta_test.txt 
  --variable-of-interest=GRP1 
  --output="test.out"
```
