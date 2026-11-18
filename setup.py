import os
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Copy top-level nanobragg2_calibration.json alongside sim_reso_mtz.py in built wheels."""

    def run(self) -> None:
        super().run()
        cal_name = "nanobragg2_calibration.json"
        if os.path.isfile(cal_name):
            self.copy_file(cal_name, os.path.join(self.build_lib, cal_name))


setup(cmdclass={"build_py": build_py})
