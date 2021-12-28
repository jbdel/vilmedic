import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import download_data
from constants import DATA_ZOO

if __name__ == '__main__':
    list_files = list(DATA_ZOO.keys())

    if len(sys.argv) == 2:
        res = sys.argv[1]
    else:
        for i, k in enumerate(list_files):
            print("{}. {} ({})".format(i + 1, k, DATA_ZOO[k][1]))

        res = input(
            "\nEnter the file number (1 or 2 for eg.) to download, or multiple numbers separated by a colon (1,3 for eg.):")

    if ',' in res:
        res = res.split(',')
    else:
        res = [res]

    # if we come from argv
    if len(sys.argv) == 2:
        try:
            res = [list_files.index(r.strip()) + 1 for r in res]
        except ValueError as e:
            sys.exit("{} of available downloads".format(e))
    else:
        res = [int(r.strip()) for r in res]

    assert all([0 < r <= len(list_files) for r in res]), "Numbers must be between 1 and {}".format(
        len(list_files))

    print("Selected downloads:")
    for r in res:
        print("\t{} [{}] in {}".format(list_files[r - 1],
                                       DATA_ZOO[list_files[r - 1]][1],
                                       DATA_ZOO[list_files[r - 1]][2]))

    for r in res:
        key = list_files[r - 1]
        download_data(key, DATA_ZOO[key][0], DATA_ZOO[key][2])
    print("done")
