from dataclasses import dataclass, field


@dataclass
class Cell:
    id: int
    position: tuple
