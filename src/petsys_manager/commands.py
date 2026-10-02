"""Pure argv builders. No launches, shell/conda activation or output discovery.

Callers supply reserved output paths (T4); preflight and live readiness remain
separate. The internal CLI argv contract is consumed by src.cornell.cli (T11).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from .contracts import Action, CommandSpec, DataFormat, Identity, Population
from .settings import PETSYS_PYTHON_TOOLS, RunSettings

# Activation of the manager's own Python environment, removed for the PETsys Python tools (FR-23).
ACTIVATION_VARIABLES = ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__")


def _absolute(value, label):
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\0" in str(value):
        raise ValueError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path


def _path(settings, name):
    if not isinstance(settings, RunSettings):
        raise ValueError("Command requires a typed run snapshot")
    return _absolute(settings.paths.get(name), name)


def petsys_python_environment(base, interpreter):
    """``base`` without Python/conda activation, the interpreter's directory first on PATH (FR-23)."""
    env = {key: value for key, value in base.items()
           if not key.startswith(("CONDA_", "_CE_")) and key not in ACTIVATION_VARIABLES}
    directory = str(Path(interpreter).parent)
    rest = [part for part in base.get("PATH", "").split(os.pathsep) if part and part != directory]
    env["PATH"] = os.pathsep.join([directory, *rest])
    return env


def _external(settings, identity, name, arguments=(), environment=None):
    cwd = _path(settings, "petsys_folder")
    tool = _absolute(settings.paths.get(f"tool:{name}", cwd / name), name)
    env = dict(os.environ)
    argv = (str(tool), *arguments)
    interpreter = settings.paths.get("petsys_python")
    if name in PETSYS_PYTHON_TOOLS and interpreter is not None:
        interpreter = _absolute(interpreter, "petsys_python")
        env = petsys_python_environment(env, interpreter)
        argv = (str(interpreter), *argv)
    if environment is not None:
        env.update(environment)
    return CommandSpec(argv, cwd, identity, env)


def _default_connection(settings):
    # Neither inspected Python tool accepts a custom-socket argument. A boolean
    # confirmation cannot invent its argv; record installed flags before support.
    if settings.profile.socket_path != "/tmp/d.sock":
        raise ValueError("Installed init/acquisition custom-socket argv contract is unrecorded")


def build_daqd(settings: RunSettings, identity: Identity, *, environment=None):
    profile = settings.profile
    if profile.daq_type not in ("PFP_KX7", "GBE"):
        raise ValueError("DAQ type is unsupported by the inspected DAQD")
    if profile.shared_memory_path != "/dev/shm/daqd_shm":
        raise ValueError("Inspected DAQD hardcodes /daqd_shm; custom shared memory is unsupported")
    if profile.daq_type == "PFP_KX7" and not profile.cards:
        raise ValueError("PFP_KX7 requires explicitly selected cards")
    arguments = ["--socket-name", profile.socket_path, "--daq-type", profile.daq_type]
    for card in profile.cards:
        arguments.extend(("--card", card))
    return _external(settings, identity, "daqd", tuple(arguments), environment)


def build_initialize(settings: RunSettings, identity: Identity, *, environment=None):
    _default_connection(settings)
    # The inspected init_system has no --config flag. INI is actually loaded by
    # acquisition/conversion; do not pretend initialization consumes that file.
    return _external(settings, identity, "init_system", environment=environment)


def build_acquisition(settings: RunSettings, identity: Identity, output_prefix, *, environment=None):
    _default_connection(settings)
    options = settings.options
    arguments = ["--config", str(_path(settings, "ini_file")), "-o",
                 str(_absolute(output_prefix, "output prefix")), "--time", str(options.duration_s),
                 "--mode", options.acquisition_mode]
    if options.hardware_trigger:
        arguments.append("--enable-hw-trigger")
    return _external(settings, identity, "acquire_sipm_data", tuple(arguments), environment)


def build_bias_off(settings: RunSettings, identity: Identity, *, environment=None):
    # Inspected set_bias: --power off switches bias power off on every active FEB.
    _default_connection(settings)
    return _external(settings, identity, "set_bias", ("--power", "off"), environment)


def build_conversion(settings: RunSettings, identity: Identity, output_prefix, *, raw_input=None,
                     environment=None):
    options = settings.options
    raw = _path(settings, "raw_input") if raw_input is None else _absolute(raw_input, "RAW input")
    # PETsys expects a prefix. Only its known .rawf suffix is stripped; arbitrary
    # dotted basenames and acquisition time are not parsed for metadata.
    prefix = raw.with_suffix("") if raw.suffix == ".rawf" else raw
    if options.output_format == DataFormat.FIXED and not settings.profile.capabilities.fixed_output_confirmed:
        raise ValueError("Installed converter fixed-output support is unconfirmed")
    tool = "convert_raw_to_group" if options.population == Population.GROUP else "convert_raw_to_coincidence"
    flag = "--writeBinaryFixed" if options.output_format == DataFormat.FIXED else "--writeBinaryCompact"
    arguments = ["--config", str(_path(settings, "ini_file")), "-i", str(prefix), "-o",
                 str(_absolute(output_prefix, "output prefix")), flag, "--writeMultipleHits",
                 str(options.hit_limit)]
    if options.splits > 1:
        arguments.extend(("--splitTime", str(options.duration_s / options.splits + 0.1)))
    return _external(settings, identity, tool, tuple(arguments), environment)


def build_internal(settings: RunSettings, identity: Identity, processing_action, request_path, result_path,
                   *, checkout_root=None, environment=None):
    action = Action(processing_action)
    names = {Action.CALIBRATE: "calibrate", Action.LISTMODE: "listmode", Action.QC_ANALYZE: "qc"}
    if action not in names or not isinstance(settings, RunSettings):
        raise ValueError("Internal stage must be calibration, listmode or offline QC")
    cwd = _absolute(checkout_root or Path(__file__).resolve().parents[2], "checkout root")
    env = dict(os.environ)
    if environment is not None:
        env.update(environment)
    argv = (sys.executable, "-u", "-m", "src.cornell.cli", names[action], "--request",
            str(_absolute(request_path, "request path")), "--result", str(_absolute(result_path, "result path")))
    return CommandSpec(argv, cwd, identity, env)
