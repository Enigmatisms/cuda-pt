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


import struct
import csv


def csv_to_binary(input_csv_path, output_bin_path):
    with open(input_csv_path, mode="r") as csvfile:
        csvreader = csv.reader(csvfile)

        next(csvreader)

        with open(output_bin_path, mode="wb") as binfile:
            cnt = 0
            for row in csvreader:
                cnt += 1
                float_data = [float(row[1]), float(row[2]), float(row[3])]
                if cnt < 10:
                    print(float_data)

                binfile.write(struct.pack("3f", *float_data))

    print(f"Data has been written to {output_bin_path}")


csv_to_binary("../scene/data/black-body-table-float-1024.csv", "output.bin")
