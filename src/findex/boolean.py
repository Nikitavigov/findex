from findex.models import Posting


def merge_and(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Перетин двох списків постінгів методом двох вказівників O(|p1| + |p2|)."""
    result: list[Posting] = []
    i = 0
    j = 0
    len1 = len(p1)
    len2 = len(p2)

    while i < len1 and j < len2:
        doc1 = p1[i].doc_id
        doc2 = p2[j].doc_id

        if doc1 == doc2:
            result.append(Posting(doc1, p1[i].term_frequency + p2[j].term_frequency))
            i += 1
            j += 1
        elif doc1 < doc2:
            i += 1
        else:
            j += 1

    return result


def merge_or(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Об'єднання двох списків постінгів методом двох вказівників O(|p1| + |p2|)."""
    result: list[Posting] = []
    i = 0
    j = 0
    len1 = len(p1)
    len2 = len(p2)

    while i < len1 and j < len2:
        doc1 = p1[i].doc_id
        doc2 = p2[j].doc_id

        if doc1 == doc2:
            result.append(Posting(doc1, p1[i].term_frequency + p2[j].term_frequency))
            i += 1
            j += 1
        elif doc1 < doc2:
            result.append(p1[i])
            i += 1
        else:
            result.append(p2[j])
            j += 1

    while i < len1:
        result.append(p1[i])
        i += 1
    while j < len2:
        result.append(p2[j])
        j += 1

    return result


def merge_not(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Різниця списків постінгів методом двох вказівників O(|p1| + |p2|)."""
    result: list[Posting] = []
    i = 0
    j = 0
    len1 = len(p1)
    len2 = len(p2)

    while i < len1 and j < len2:
        doc1 = p1[i].doc_id
        doc2 = p2[j].doc_id

        if doc1 == doc2:
            i += 1
            j += 1
        elif doc1 < doc2:
            result.append(p1[i])
            i += 1
        else:
            j += 1

    while i < len1:
        result.append(p1[i])
        i += 1

    return result


def set_and(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Перетин списків постінгів через множини set."""
    if not p1 or not p2:
        return []
    map1 = {p.doc_id: p for p in p1}
    map2 = {p.doc_id: p for p in p2}
    common_ids = sorted(map1.keys() & map2.keys())
    return [
        Posting(did, map1[did].term_frequency + map2[did].term_frequency)
        for did in common_ids
    ]


def set_or(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Об'єднання списків постінгів через множини set."""
    map1 = {p.doc_id: p for p in p1}
    map2 = {p.doc_id: p for p in p2}
    all_ids = sorted(map1.keys() | map2.keys())
    res: list[Posting] = []
    for did in all_ids:
        tf1 = map1[did].term_frequency if did in map1 else 0
        tf2 = map2[did].term_frequency if did in map2 else 0
        res.append(Posting(did, tf1 + tf2))
    return res


def set_not(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Різниця списків постінгів через множини set."""
    if not p1:
        return []
    if not p2:
        return p1
    map1 = {p.doc_id: p for p in p1}
    set2 = {p.doc_id for p in p2}
    diff_ids = sorted(map1.keys() - set2)
    return [map1[did] for did in diff_ids]
