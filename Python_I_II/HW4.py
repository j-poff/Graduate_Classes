def count_starts(seq: str) -> int:
    if len(seq) < 3:
        print("The sequence provided is less than 3 letters")
    else:
        ATGcount: int = 0
        bottomCount: int = 0
        topCount: int = 3
        length: int = len(seq) + 1
        while topCount <= length:
            if seq[bottomCount:topCount] == "ATG":
                ATGcount += 1
            bottomCount += 1
            topCount += 1
    return ATGcount

assert count_starts("ATATGCGATGGACGATGC") == 3, "Error on test 1!"
assert count_starts("ATG") == 1, "Error on test 2!"
assert count_starts("ATGCATG") == 2, "Error on test 3!"

#%%
import io
from typing import IO, List
text: IO = io.open("seqs.txt", "r")
most_count: int = -1
best_id: str = "NA"

for item in text:
    item_stripped: str = item.strip()
    item_list: List = item.split("\t")
    if count_starts(item_list[1]) > most_count:
        most_count = count_starts(item_list[1])
        best_id = item_stripped[0]
text.close()
print("The sequence id that has the most \"ATG\" elements is " + best_id + " with a count of " + str(most_count))
#%%
from typing import IO, List
def get_windows(seq: str, window_size: int, step_size: int) -> List[str]:
    charList: List = []
    bottomCount: int = 0
    topCount: int = window_size
    length: int = len(seq) + 1
    while topCount <= length:
        if len(seq[bottomCount:topCount]) == window_size:
            charList.append(seq[bottomCount:topCount])
        bottomCount += step_size
        topCount += step_size
    return charList

assert get_windows("ATATGCGATGGACG", 3, 4) == ["ATA", "GCG", "TGG"], "Error on test 1!"
assert get_windows("ATGCG", 3, 1) == ["ATG", "TGC", "GCG"], "Error on test 2!"
assert get_windows("ATGG", 2, 2) == ["AT", "GG"], "Error on test 3!"
assert get_windows("ATG", 1, 1) == ["A", "T", "G"], "Error on test 4!"