TSD searching code using Python-Parasail to perform sequence alignment and some functions to extract matching substrings from the alignments.

Originally intended for use in an AnnoSINE v2 rewrite; this code expects you have already found short regions where you believe a TSD might be found and simply want to find it. It is not meant to search an entire genome and identify all cases of TSDs.

This is more of a developer's resource than an independent tool. It has little use outside of the context of another program which will find candidates that might have TSDs. Despite this, we do support direct use on the commandline.

Overview:

TSD searcher's primary behavior takes two sequences to search for TSDs. Internally, it refers to these as 'left' and 'right' because the original use case for this code was finding TSDs flanking SINE transposable elements. The process by which this search is achieved is:

* (1) Find the longest exact repeats shared between left and right of at least a certain minimum length (default 5) using a suffix array + longest common prefix array approach implemented via pydivsufsort (https://github.com/louisabraham/pydivsufsort)
* (2) Extend these exact repeats into their flanking regions using local sequence alignment implemented via python-parasail (https://github.com/jeffdaily/parasail-python)
* (3) Assess the quality of potential extensions on a seed exact repeat based on whether they contain more matches than mismatches from the exact repeat (see score function), and decide on either a final choice of extension or retain the seed repeat
* (4) Filter posssible TSDs based on their sequence content (primarily to remove polyA sequences) and to impose a minimum length
* (5) Optionally select a single best TSD candidate
* (6) Return recovered TSDs and localize them to their parent sequences

The entire process can also be applied to find terminal inverted repeats (TIRs) instead of TSDs, a behavior that TSD-searcher natively supports.

Notes for developers:

As TSD-searcher was originally designed as a developer resource, the main class of TSD-searcher, alignment_tsd_tir_finder, is designed to act as your API to the tool's behaviors. You initialize the class with your desired parameters specifying limits on TSD size, number of mismatches, scoring approach, and filtering approaches, and whether you'd like the tool to return one or multiple hits, if any are found, for each pair of sequences. Use of the class is given by the 'operate(left, right, is_TIR=False/True)' function, which takes a pair of sequences and your desired type of repeat.

Currently, TSD-searcher does NOT support a parallelized approach on its own. It's better that you create an instance of TSD-searcher in each of several python processes and call it from there.

When importing TSD searcher, you should exercise some caution: TSD-searcher uses python bindings to the C code for libdivsufsort, and this code is compiled with OpenMP. By default, this will cause TSD searcher to create its suffix arrays and longest common prefix arrays using all threads available on the system. Unless you are trying to search very long strings against eachother, this is very bad for TSD-searcher's performance because the parallel overhead of libdivsufsort is simply not worth the effort. In the expected use case, where fairly short strings are being interrogated, this will produce worse performance in every way.

The solution is to set the OMP_NUM_THREADS environmental variable to 1 before importing TSD-searcher. Your code should (probably) look like this:

```Python
import os
os.environ['OMP_NUM_THREADS'] = '1'
from TSD-searcher import alignment_tsd_tir_finder
```

