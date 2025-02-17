from typing import Optional
from pydantic import BaseModel


class LegalReference(BaseModel):
  title: str
  type: str
  reference: Optional[str] = None
  date: Optional[str] = None


def create_named_entities(model: LegalReference) -> tuple:
    """Create named entities from the model"""
    title = model.title

    named_entities = [
        get_span(model, 'type')
    ]

    if model.reference is not None:
        named_entities.append(get_span(model, 'reference'))

    if model.date is not None:
        named_entities.append(get_span(model, 'date'))

    return title, {"entities": named_entities}



def get_span(model, component) -> tuple:
    """Get the start and end indexes of a component in the title"""
    value = getattr(model, component)

    return (
        model.title.index(value),
        model.title.index(value) + len(value),
        component.upper()
    )
