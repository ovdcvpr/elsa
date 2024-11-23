from __future__ import annotations

import magicpandas as magic
from elsa.resource import Resource


class ILabel(
    Resource,
):
    @magic.column
    def label(self) -> magic[str]:
        """
        String label for each annotation assigned by the dataset.
        For an annotation this is the singular label e.g. 'person';
        For a combo this is the concatenated label e.g. 'person walking'
        """
        labels = self.elsa.labels
        result = (
            labels
            .reset_index()
            .label
            .indexed_on(self.ilabel)
            .values
        )

        return result

    @magic.index
    def ilabel(self) -> magic[int]:
        """
        Label ID of the label; synonymous labels will have the same
        ilabel value. For example, 'person' and 'individual' have the same
        ilabel value.
        """
