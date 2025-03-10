import numpy as np

from io_utils import read_txt_volume, write_off
from mc import tmc


class Stats:
    def __init__(self):
        self.faces = 0
        self.ambiguous = 0
        self.no_decider = 0
        self.plane = 0
        self.singular = 0
        self.singular_on_face = 0

def main():
    # ("data/head_ushort_512_512_641_half_res.txt", 950)
    # files = [("Angio_ushort_384_512_80.txt",67), ("Baby_ushort_256_256_98.txt",119), ("becken_ushort_512_512_1047.txt",901), ("Bruce_ushort_256_256_156.txt",800), ("cardix_ushort_512_512_452.txt",750), ("cenovix_ushort_512_512_361.txt",1200),
    #          ("CT-Abdomen_ushort_512_512_147.txt",38), ("CT-Head_ushort_256_256_113.txt",870), ("head_ushort_512_512_641.txt",900), ("Isabel-Upwind_ushort_500_500_100.txt",1031), ("Knee_ushort_512_512_87.txt",1300),
    #          ("mecanix_ushort_512_512_743.txt",700), ("mecanix_ushort_512_512_743.txt",1200), ("MRI-Head_ushort_256_256_256.txt",100), ("MRI-Woman_ushort_256_256_109.txt",1320), ("Porsche_ushort_559_1023_347.txt",31),
    #          ("Retrograde_ushort_320_320_72.txt",2100), ("thorax_hals_ushort_512_512_497.txt",700), ("Tomato_ushort_256_256_64.txt",11), ("VisMale_ushort_128_256_256.txt",59), ("Carp_ushort_256_256_512.txt", 1281), ("Engine_ushort_256_256_256.txt", 50)]

    files = [("cardix_ushort_512_512_452.txt", 750), ("cenovix_ushort_512_512_361.txt", 1200),
             ("head_ushort_512_512_641.txt", 900), ("Isabel-Upwind_ushort_500_500_100.txt", 1031),
             ("mecanix_ushort_512_512_743.txt", 1200), ("Porsche_ushort_559_1023_347.txt", 31),
             ("Carp_ushort_256_256_512.txt", 1281)]
    files = [("Isabel-Upwind_ushort_500_500_100.txt", 1031), ("mecanix_ushort_512_512_743.txt", 1200), ("Porsche_ushort_559_1023_347.txt", 31)]
    stats_list = []

    for i, file in enumerate(files):
        filename, iso = file
        grid = read_txt_volume("data/Volumes_half/" + filename)
        if iso is None:
            iso = np.mean(grid.values)
        print(f"file: {filename}, iso: {iso}")

        # try:
        vertices, faces = tmc(grid, iso)
        write_off("meshes/Volumes_half/" + filename.replace(".txt", ".off"), vertices, faces)
        # except Exception as e:
        #     print(e)
        continue

        if iso != 0:
            for idx in grid.indices():
                grid[idx] -= iso
            iso = 0

        stats = Stats()

        for v0, v1, v2, v3 in grid.faces():
            f0 = grid[v0]
            f2 = grid[v1]
            f1 = grid[v2]
            f3 = grid[v3]
            stats.faces += 1

            if (f0 > iso and f3 > iso and f1 < iso and f2 < iso) or (f0 < iso and f3 < iso and f1 > iso and f2 > iso):
                stats.ambiguous += 1

            if abs(f0 - f1 - f2 + f3) == 0:
                stats.no_decider += 1
                asymptotic_center = -1, -1

                if f0 == iso and f1 == iso and f2 == iso and f3 == iso:
                    stats.plane += 1
                    continue
            else:
                asymptotic_center = (f0 - f1) / (f0 - f1 - f2 + f3), (f0 - f2) / (f0 - f1 - f2 + f3)

            if f0 * f3 - f1 * f2 == 0:
                stats.singular += 1
                if asymptotic_center[0] >= 0 and asymptotic_center[0] <= 1 and asymptotic_center[1] >= 0 and asymptotic_center[1] <= 1:
                    stats.singular_on_face += 1

        stats_list.append(stats)

    total_stats = Stats()
    for i, stats in enumerate(stats_list):
        print("================================================")
        print(f"{files[i]}:")
        print(f"faces {stats.faces}, ambiguous {stats.ambiguous}, no_decider {stats.no_decider}, plane {stats.plane}, singular {stats.singular}, singular on face {stats.singular_on_face}")
        total_stats.faces += stats.faces
        total_stats.ambiguous += stats.ambiguous
        total_stats.no_decider += stats.no_decider
        total_stats.plane += stats.plane
        total_stats.singular += stats.singular
        total_stats.singular_on_face += stats.singular_on_face

    print("================================================")
    print("total:")
    print(f"faces {total_stats.faces}, ambiguous {total_stats.ambiguous}, no_decider {total_stats.no_decider}, plane {total_stats.plane}, singular {total_stats.singular}, singular on face {total_stats.singular_on_face}")

if __name__ == '__main__':
    main()
