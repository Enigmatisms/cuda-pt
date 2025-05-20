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

VERSION = "1.6.0"


def get_cpplint_version():
    try:
        output = subprocess.check_output(
            ["cpplint", "--version"], stderr=subprocess.STDOUT, universal_newlines=True
        )
        return output.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if isinstance(e, FileNotFoundError):
            print("cpplint not found, attempting installation...")
        else:
            print(f"Version check failed: {e.output}")
        return None


def install_cpplint():
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"cpplint=={VERSION}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        print(f"Successfully installed cpplint {VERSION}")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        sys.exit(1)


def main():
    version_output = get_cpplint_version()
    if not version_output or VERSION not in version_output:
        install_cpplint()
        version_output = get_cpplint_version()
        if not version_output or VERSION not in version_output:
            print(f"Failed to verify cpplint {VERSION} after installation")
            sys.exit(1)

    try:
        subprocess.run(["cpplint"] + sys.argv[1:], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
