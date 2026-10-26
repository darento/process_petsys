"""IMAS TIME-channel skew calibration (spec 011).

One command, ``python -m src.imas.skew COMMAND ...`` (see ``__main__``).
Modules: ``setup`` (config/map lookup and SM numbering), ``inventory`` (raw
counts per TIME channel), ``select`` (side selection), ``encal`` (photopeak
calibration from the skew data), ``geometry`` (SM facets, endpoints, bar
source), ``cache`` (adapters for ``src.timing.cache``) and ``pairstats``
(FR-7 measurement). The scanner-neutral core is ``src.timing``.
"""
