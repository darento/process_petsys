"""IMAS fast compact LDAT -> LM V5 with the delivered timing tables (spec 015).

``select`` (compiled selection), ``timing`` (skew + walk ``dt``), ``nn`` (NN y and DOI), ``records`` (LM V5
records and header), ``job`` (parallel workers, one NN process, merge) and ``__main__`` (docopt command line,
``python -m src.imas.listmode``).
"""
