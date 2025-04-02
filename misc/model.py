from enum import StrEnum
from typing import Optional, List

from pydantic import BaseModel


class LegalReferenceEntity(StrEnum):
    TYPE = 'types'
    REFERENCE = 'references'
    DATE = 'dates'


class LegalReference(BaseModel):
    title: str
    types: Optional[List[str]] = None
    references: Optional[List[str]] = None
    dates: Optional[List[str]] = None

    def get_span(self, entity: LegalReferenceEntity) -> Optional[List[tuple]]:
        values = getattr(self, entity.value)
        if not values:
            return None

        spans = []
        for value in values:
            if value in self.title:
                start = self.title.index(value)
                spans.append((start, start + len(value), entity.value.strip('s').upper()))

        return spans if spans else None

    def to_named_entities(self) -> tuple:
        named_entities = [
            span for entity in LegalReferenceEntity
            if (spans := self.get_span(entity)) is not None
            for span in spans  # Flatten the list of spans
        ]
        return self.title, {"entities": named_entities}
