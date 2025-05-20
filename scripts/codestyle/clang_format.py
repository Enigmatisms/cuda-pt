# Copyright (C) 2025 Qianyue He
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General
# Public License along with this program. If not, see
#
#             <https://www.gnu.org/licenses/>.


import sys
import subprocess
import re

VERSION = "13.0.0"


def check_python_version():
    if sys.version_info < (3, 6):
        sys.exit("Error: Python >=3.6 required for clang-format installation")


def get_clang_version():
    try:
        result = subprocess.run(
            ["clang-format", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def install_clang_format():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"clang-format=={VERSION}"], check=True
    )


def main():
    check_python_version()

    version_output = get_clang_version()
    if not re.search(rf"\b{VERSION}\b", version_output):
        install_clang_format()
        if not re.search(rf"\b{VERSION}\b", get_clang_version()):
            sys.exit(f"Failed to install clang-format {VERSION}")

    try:
        subprocess.run(["clang-format", "-i"] + sys.argv[1:], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
