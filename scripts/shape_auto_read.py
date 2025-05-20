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
import os

X = "../meshes/rich/vision"


def generate_xml_elements(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not valid")
        return

    num_obj_file = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith(".obj") and os.path.isfile(
            os.path.join(directory, filename)
        ):
            num_obj_file += 1
            file_path = os.path.join(X, filename).replace(os.sep, "/")

            print(
                f"""<shape type="obj">
    <string name="filename" value="{file_path}"/>
    <ref type="material" id="white"/>
</shape>"""
            )
    print(f"\nNumber of file: {num_obj_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python ./shape_auto_read.py <path to the folder that contains the OBJ files>"
        )
        sys.exit(1)

    target_dir = sys.argv[1]
    generate_xml_elements(target_dir)
