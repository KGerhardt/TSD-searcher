TSD searching code using Python-Parasail to perform sequence alignment and some functions to extract matching substrings from the alignments.

Originally intended for use in an AnnoSINE v2 rewrite; this code expects you have already found short regions where you believe a TSD might be found and simply want to find it. It is not meant to search an entire genome and identify all cases of TSDs.

This is more of a developer's resource than an independent tool. It has little use outside of the context of another program which will find candidates that might have TSDs. Despite this, we do support direct use on the commandline.

Overview:

TSD searcher's primary behavior takes two sequences to search for TSDs. Internally, it refers to these as 'left' and 'right' because the original use case for this code was finding TSDs flanking SINE transposable elements. The process by which this search is achieved is:

(1) Find the longest exact repeats shared between left and right of at least a certain minimum length (default 5) using a suffix array + longest common prefix array approach implemented via pydivsufsort (https://github.com/louisabraham/pydivsufsort)
(2) Extend these exact repeats into their flanking regions using local sequence alignment implemented via python-parasail (https://github.com/jeffdaily/parasail-python)
(3) Assess the quality of potential extensions on a seed exact repeat based on whether they contain more matches than mismatches from the exact repeat (see score function), and decide on either a final choice of extension or retain the seed repeat
(4) Filter posssible TSDs based on their sequence content (primarily to remove polyA sequences) and to impose a minimum length
(5) Optionally select a single best TSD candidate
(6) Return recovered TSDs and localize them to their parent sequences

The entire process can also be applied to find terminal inverted repeats (TIRs) instead of TSDs, a behavior that TSD-searcher natively supports.
