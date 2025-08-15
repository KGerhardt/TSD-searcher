TSD searching code using Python-Parasail to perform sequence alignment and some functions to extract matching substrings from the alignments.

Originally intended for use in an AnnoSINE v2 rewrite; this code expects you have already found short regions where you believe a TSD might be found and simply want to find it. It is not meant to search an entire genome and identify all cases of TSDs.

This is more of a developer's resource than an independent tool. It has little use outside of the context of another program which will find candidates that might have TSDs. Despite this, we do support direct use on the commandline.
