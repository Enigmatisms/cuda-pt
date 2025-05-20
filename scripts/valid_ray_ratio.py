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


import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = [
        0.961066,
        0.878682,
        0.811131,
        0.557415,
        0.473127,
        0.398289,
        0.331326,
        0.271257,
        0.220298,
        0.177713,
        0.143576,
        0.116826,
        0.095946,
        0.079549,
        0.066654,
        0.056299,
        0.960768,
        0.878350,
        0.811313,
        0.558644,
        0.474308,
        0.398446,
        0.332012,
        0.272559,
        0.221201,
        0.178268,
        0.144224,
        0.117132,
        0.096204,
        0.080111,
        0.067025,
        0.056655,
        0.960863,
        0.878102,
        0.811006,
        0.557529,
        0.473726,
        0.398615,
        0.331683,
        0.271914,
        0.220783,
        0.177918,
        0.143521,
        0.116800,
        0.096252,
        0.079870,
        0.066950,
        0.056543,
        0.960765,
        0.878753,
        0.811769,
        0.558162,
        0.474058,
        0.399303,
        0.332425,
        0.272365,
        0.220993,
        0.177931,
        0.143970,
        0.117282,
        0.096453,
        0.079906,
        0.066880,
        0.056626,
        0.960877,
        0.878747,
        0.811009,
        0.557480,
        0.473477,
        0.398135,
        0.331774,
        0.271868,
        0.220298,
        0.177541,
        0.143438,
        0.116667,
        0.095719,
        0.079443,
        0.066499,
        0.056281,
        0.960830,
        0.878436,
        0.810998,
        0.557137,
        0.473872,
        0.398991,
        0.332484,
        0.272552,
        0.221333,
        0.178763,
        0.144609,
        0.117682,
        0.096629,
        0.080397,
        0.067362,
        0.056813,
        0.960744,
        0.878595,
        0.811313,
        0.556904,
        0.473389,
        0.398490,
        0.331331,
        0.272083,
        0.220705,
        0.178135,
        0.144020,
        0.116930,
        0.096239,
        0.079683,
        0.066797,
        0.056471,
    ]

    x = np.arange(len(data))

    xticks = np.arange(0, len(data), 16)
    yticks = np.linspace(0, 1, 11)

    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.plot(x, data, label="GPU Utilization", color="#F36A13")
    plt.scatter(x, data, color="#F3200C", zorder=5, s=3)

    plt.legend(loc="upper right")
    plt.xticks(xticks + 8, labels=np.arange(1, len(xticks) + 1))
    plt.yticks(yticks, labels=[f"{int(y*100)}%" for y in yticks])

    for i in range(1, len(xticks)):
        plt.axvline(x=xticks[i] - 0.5, color="gray", linestyle="--", alpha=0.3)
        plt.text(
            xticks[i] - 0.5,
            0.98,
            f"iter = {i+1}",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
        )

    plt.axhline(y=1, color="gray", linestyle="--", alpha=0.3)

    plt.title("Valid ray ratio (GPU Utilization) over Iterations")
    plt.xlabel("Iterations (16 bounces per iteration)")
    plt.ylabel("Ratio of valid ray (%)")

    plt.grid(axis="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig("whiskey-ray-ratio.png", dpi=300)
    plt.show()
