ANCOM
=====

[Analysis of Composition of Microbiomes](http://www.microbecolhealthdis.net/index.php/mehd/article/view/27663%20)

ANCOM is a novel statistical test that accounts for OTU dependencies when calculating group signficance.
This software makes use of accelerated matrix multiplication and permutation tests to
allow for these statistical tests to complete within a reasonable amount of time.


Installation
============
This software requires the following dependencies
```
python=3
pandas
biom-format
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
That can be done by going to the cloned directory, and running the following command

```
nosetests .
```
ANCOM has been submitted to scikit-bio found [here](http://scikit-bio.org/docs/0.4.2/generated/generated/skbio.stats.composition.ancom.html#skbio.stats.composition.ancom)

To get started using the accelerated permutation tests run the following code
```python
    >>> from ancomP.stats.ancom import ancom
    >>> import pandas as pd
    >>> table = pd.DataFrame([[12, 11, 10, 10, 10, 10, 10],
    ...                       [9,  11, 12, 10, 10, 10, 10],
    ...                       [1,  11, 10, 11, 10, 5,  9],
    ...                       [2,  11, 10, 11, 10, 5,  9],
    ...                       [221, 210, 9,  10, 10, 10, 10],
    ...                       [220, 210, 9,  10, 10, 10, 10],
    ...                       [200, 220, 10, 10, 13, 10, 10],
    ...                       [230, 210, 14, 10, 10, 10, 10]],
    ...                      index=['s1','s2','s3','s4','s5','s6', 's7','s8'],
    ...                      columns=['b1','b2','b3','b4','b5','b6','b7'])
    >>> grouping = pd.Series([0, 0, 0, 0, 1, 1, 1, 1],
    ...                      index=['s1','s2','s3','s4','s5','s6', 's7','s8'])
    >>> results = ancom(table, grouping, significance_test='permutative-anova', permutations=100)
    >>> results['reject']
    b1     True
    b2     True
    b3    False
    b4    False
    b5    False
    b6    False
    b7    False
```

Some more information about ANCOM can be found at this [blog post](http://mortonjt.blogspot.com/2016/06/ancom-explained.html)
