from array import array
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Posting:
    """
    Елемент списку інвертованого індексу.
    Використовує slots=True для компактного збереження в пам'яті (без __dict__)
    та frozen=True для незмінності й автоматичної генерації __hash__.
    """

    doc_id: int
    term_frequency: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    """
    Метадані документа для показу результатів пошуку та обчислення довжин.
    """

    doc_id: int
    source_name: str
    title: str
    length: int


@dataclass(order=True, slots=True)
class SearchResult:
    """
    Результат пошуку з оцінкою релевантності.
    order=True дозволяє автоматично порівнювати за полем score для heapq.nlargest.
    """

    score: float
    doc_id: int = field(compare=False)
    title: str = field(compare=False, default="")
    source_name: str = field(compare=False, default="")
    snippet: str = field(compare=False, default="")

    def __str__(self) -> str:
        return f"[{self.score:.4f}] Doc #{self.doc_id} ({self.title}): {self.snippet}"


# Варіант без slots для порівняльного бенчмарку пам'яті
@dataclass
class PlainPosting:
    doc_id: int
    term_frequency: int


# Компактне представлення списку постінгів через масив цілих чисел array('I')
class ArrayPostings:
    __slots__ = ("_data",)

    def __init__(self, data: array | None = None) -> None:
        self._data = data if data is not None else array("I")

    def append(self, doc_id: int, term_frequency: int) -> None:
        self._data.append(doc_id)
        self._data.append(term_frequency)

    def __len__(self) -> int:
        return len(self._data) // 2

    def __iter__(self):
        for i in range(0, len(self._data), 2):
            yield Posting(self._data[i], self._data[i + 1])
