"""Cornell TIME-channel skew calibration (spec 008).

One command, ``python -m src.cornell.skew COMMAND ...`` (see ``__main__``).
Modules: ``dataset`` (raw inventory, holdout), ``geometry`` (shell, detector
frames, TOF), ``ingest`` (cuts and features), ``cache`` (feature and
observation caches), ``pairs`` (two-branch pair fits), ``solver`` (global
channel solve), ``lut`` (canonical LUT, native PETsys table, conversion
manifests, composition), ``plots`` (per-ASIC diagnostics), ``job`` (pooled
multi-run calibration), ``validation`` (frozen-LUT test evaluation),
``roundtrip`` (native conversion check) and ``benchmark`` (decoder baseline,
development only).
"""
