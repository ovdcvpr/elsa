from __future__ import annotations
from __future__ import annotations

from pandas import Series
from typing import Literal

if False:
    from elsa.root import Elsa


class DryDoc:
    def ibox(self) -> Series[int]:
        """
        Identifier for each combined label box in the ground truth
        annotations.
        """

    def ilabel(self) -> Series[int]:
        """
        Label ID of the label; synonymous labels will have the same
        ilabel value. For example, 'person' and 'individual' have the same
        ilabel value.
        """

    def label(self) -> Series[str]:
        """
        String label for each annotation assigned by the dataset.
        For an annotation this is the singular label e.g. 'person';
        For a combo this is the concatenated label e.g. 'person walking'
        """

    def ilabels(self) -> Series[tuple[int]]:
        """
        Tuple of sorted ilabels to represent each unique combo of labels.

        labels             ilabels
        person             0
        person walking     0, 4
        """

    def cat(self) -> Series[str]:
        """
        Metaclass, or type of label, for each box

        labels              cat
        person              condition
        people              condition
        walking             state
        biking              state
        dining              activity
        cleaning            activity
        kid                 others
        police              others
        """

    def iclass(self) -> Series[int]:
        """
        Identifier for each unique class (combo)

        labels              iclass
        person              0
        person walking      1
        group talking       2
        """

    def ifile(self) -> Series[str]:
        """ Identifier for the file e.g. BSV_073 """

    def file(self) -> Series[str]:
        """
        The filename for the file e.gi
        """

    def elsa(self) -> Elsa:
        """
        A backwards reference to the root Elsa instance
        """

    def path(self) -> Series[str]:
        """
        Local filesystem path for each literal image file

        ifile       path
        BSV_001     ../../BSV_001.png
        BSV_002     ../../BSV_002.png
        """

    def nfile(self) -> Series[int]:
        """An integer [0, N] for each unique ifile"""

    def prompt(self) -> Series[str]:
        """
        String of concatenated labels, modified to sound natural, to be
        used as a prompt for the model
        """

    def iann(self) -> Series[int]:
        """Identifier for each ground truth annotation"""

    def combo(self) -> Series[str]:
        """
        String of combined labels for the combo box that the annotation
        belongs to in aggregate:

        label       combo
        person      person walking on phone
        walking     person walking on phone
        on phone    person walking on phone
        """

    def natural(self) -> Series[str]:
        """
        Natural language representation of the label:
        'person' -> 'a person'
        'sports' -> 'doing sports'
        """

    def iorder(self) -> Series[int]:
        """
        Value assigned for ordering the labels in the process of
        generating natural prompts from the combinations.
        """

    def nlabels(self) -> Series[int]:
        """How many labels are in the combo"""

    def nunique_labels(self) -> Series[int]:
        """How many unique labels are in the combo"""

    def rcombo(self) -> Series[int]:
        """
        The relative index of each combo for that
        file; the lowest index is 0 for each file e.g.
        0 1 2 3 0 1 0 1 2 0 0 rcombo
        a a a a b b c c c d d file
        """

    def icat(self) -> Series[int]:
        """The unique index of the metaclass of the synonym set that the label belongs to"""

    def syn(self) -> Series[str]:
        """An undercase label that may be associated with a set of synonyms"""

    def w(self) -> Series[float]:
        """
        minimum w value of the bounding box in pixels;
        'western' bound of the bounding box AKA minx in pixels
        """

    def e(self) -> Series[float]:
        """
        minimum e value of the bounding box in pixels;
        'eastern' bound of the bounding box AKA maxx in pixels
        """

    def n(self) -> Series[float]:
        """
        minimum n value of the bounding box in pixels;
        'northern' bound of the bounding box AKA maxy in pixels
        """

    def s(self) -> Series[float]:
        """
        minimum s value of the bounding box in pixels;
        'southern' bound of the bounding box AKA miny in pixels
        """

    def xmin(self) -> Series[float]:
        """ Minimum x value of the bounding box in pixels """

    def xmax(self) -> Series[float]:
        """ Maximum x value of the bounding box in pixels """

    def ymin(self) -> Series[float]:
        """ Minimum y value of the bounding box in pixels """

    def ymax(self) -> Series[float]:
        """ Maximum y value of the bounding box in pixels """

    def minx(self) -> Series[float]:
        """ Minimum x value of the bounding box in pixels """

    def maxx(self) -> Series[float]:
        """ Maximum x value of the bounding box in pixels """

    def miny(self) -> Series[float]:
        """ Minimum y value of the bounding box in pixels """

    def maxy(self) -> Series[float]:
        """ Maximum y value of the bounding box in pixels """

    def x(self) -> Series[float]:
        """ Center of the bounding box in the x direction pixels """

    def y(self) -> Series[float]:
        """ Center of the bounding box in the y direction in pixels """

    def height(self) -> Series[float]:
        """ Height of the bounding box in pixels """

    def width(self) -> Series[float]:
        """ Width of the bounding box in pixels """

    def normw(self) -> Series[float]:
        """ Normalized western bound of the bounding box """

    def norme(self) -> Series[float]:
        """ Normalized eastern bound of the bounding box """

    def normn(self) -> Series[float]:
        """ Normalized northern bound of the bounding box """

    def norms(self) -> Series[float]:
        """ Normalized southern bound of the bounding box """

    def normx(self) -> Series[float]:
        """ Normalized center of the bounding box in the x direction """

    def normy(self) -> Series[float]:
        """ Normalized center of the bounding box in the y direction """

    def normwidth(self) -> Series[float]:
        """ Normalized width of the bounding box """

    def normheight(self) -> Series[float]:
        """ Normalized height of the bounding box """

    def fn(self) -> Series[float]:
        """
        Normalized 'file north' of the bounding box; This is computed
        by adding the file ID to the normalized north value, resulting
        in non-intersecting bounding boxes across files
        """

    def fs(self) -> Series[float]:
        """
        Normalized 'file south' of the bounding box; This is computed
        by adding the file ID to the normalized south value, resulting
        in non-intersecting bounding boxes across files
        """

    def fw(self) -> Series[float]:
        """
        Normalized 'file west' of the bounding box; This is computed
        by adding the file ID to the normalized west value, resulting
        in non-intersecting bounding boxes across files
        """

    def fe(self) -> Series[float]:
        """
        Normalized 'file east' of the bounding box; This is computed
        by adding the file ID to the normalized east value, resulting
        in non-intersecting bounding boxes across files
        """

    def fx(self) -> Series[float]:
        """
        Normalized 'file x' of the bounding box; This is computed
        by adding the file ID to the normalized x value, resulting
        in non-intersecting bounding boxes across files
        """

    def fy(self) -> Series[float]:
        """
        Normalized 'file y' of the bounding box; This is computed
        by adding the file ID to the normalized y value, resulting
        in non-intersecting bounding boxes across files
        """

    def image_height(self) -> Series[float]:
        """ Height of the image in pixels """

    def image_width(self) -> Series[float]:
        """ Width of the image in pixels """

    def level(self) -> Series[str]:
        """
        The level of the combo, or the characters, e.g. cs, csa, csaoa

        c: condition
        cs: condition, state
        csa: condition, state, activity
        cso: condition, state, others
        csao: condition, state, activity, others
        """
    def condition(self) -> Literal['person', 'pair', 'people',]:
        """
        The mutually exclusive value for the condition category within
        a combo:
            person, pair, or people
        """
