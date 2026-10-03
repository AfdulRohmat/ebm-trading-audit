"""Compile only the execution accelerator (no classifier logic)."""

import ctypes
import shutil
import subprocess
from pathlib import Path


def compile_executor(directory):
    compiler = shutil.which("c++")
    if not compiler:
        raise RuntimeError("A C++17 compiler is required for full replay")
    target = Path(directory) / "execution.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(Path(__file__).with_name("execution.cpp")),
            "-o",
            str(target),
        ],
        check=True,
    )
    return ctypes.CDLL(str(target.resolve()))
