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
#            <https://www.gnu.org/licenses/>.

import argparse
import datetime
import os
import re
import sys

COPYRIGHT = """Copyright (C) {year} Qianyue He

This program is free software: you can redistribute it and/or
modify it under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General
Public License along with this program. If not, see

            <https://www.gnu.org/licenses/>.
"""


def _generate_copyright(comment_mark):
    year = datetime.datetime.now().year
    copyright = COPYRIGHT.format(year=year)
    linesep = "\n"

    return [
        (f"{comment_mark} {line}{linesep}" if line else f"{comment_mark}{linesep}")
        for line in copyright.splitlines()
    ]


def _get_comment_mark(path):
    lang_type = re.compile(r"\.(py|pyi|sh)$")
    if lang_type.search(path) is not None:
        return "#"

    lang_type = re.compile(r"\.(h|c|hpp|cc|cuh|cpp|cu)$")
    if lang_type.search(path) is not None:
        return "//"

    return None


RE_ENCODE = re.compile(r"^[ \t\v]*#.*?coding[:=]", re.IGNORECASE)
RE_COPYRIGHT = re.compile(r".*Copyright \(C\) \d{4}", re.IGNORECASE)
RE_SHEBANG = re.compile(r"^[ \t\v]*#[ \t]?\!")


def _check_copyright(path):
    head = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = [next(f) for x in range(4)]
    except StopIteration:
        pass

    for idx, line in enumerate(head):
        if RE_COPYRIGHT.search(line) is not None:
            return True

    return False


def generate_copyright(path, comment_mark):
    original_contents = open(path, "r", encoding="utf-8").readlines()
    head = original_contents[0:4]

    insert_line_no = 0
    for i, line in enumerate(head):
        if RE_ENCODE.search(line) or RE_SHEBANG.search(line):
            insert_line_no = i + 1

    copyright = _generate_copyright(comment_mark)
    if insert_line_no == 0:
        new_contents = copyright
        if len(original_contents) > 0 and len(original_contents[0].strip()) != 0:
            new_contents.append(os.linesep)
        new_contents.extend(original_contents)
    else:
        new_contents = original_contents[0:insert_line_no]
        new_contents.append(os.linesep)
        new_contents.extend(copyright)
        if (
            len(original_contents) > insert_line_no
            and len(original_contents[insert_line_no].strip()) != 0
        ):
            new_contents.append(os.linesep)
        new_contents.extend(original_contents[insert_line_no:])
    new_contents = "".join(new_contents)

    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(new_contents)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Checker for copyright declaration.")
    parser.add_argument("filenames", nargs="*", help="Filenames to check")
    args = parser.parse_args(argv)

    for path in args.filenames:
        comment_mark = _get_comment_mark(path)
        if comment_mark is None:
            print("warning:Unsupported file", path, file=sys.stderr)
            continue

        if _check_copyright(path):
            continue

        generate_copyright(path, comment_mark)


if __name__ == "__main__":
    sys.exit(main())
