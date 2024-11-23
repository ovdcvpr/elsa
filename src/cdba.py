
from elsa import Elsa
from elsa.scored.scored import Scored

elsa = Elsa.from_unified()
scored = elsa.scored(file='selected.nlse>.3.parquet', )
loc = scored.ifile.isin('GSV_136 GSV_37 GSV_379 BSV_593'.split())
loc &= scored.prompt.isin('a person; two people; people chatting and walking; a gathering walking'.split('; '))
scored: Scored = scored[loc].iloc[:4]
cdba = scored.cdba(anchored=True)



"""

a person
two people
people chatting and walking
a gathering walking

GSV_136
GSV_37
GSV_379
BSV_593

"""

# def contains_prompt(
#         scored: Scored,
#         prompt: str,
# ) -> Series[bool]:
#     result = (
#         scored.prompt
#         .eq(prompt)
#         .groupby(scored.ifile.values, sort=False,observed=True)
#         .any()
#     )
#     return result
#
# loc = contains_prompt(scored, 'a person')
# loc &= contains_prompt(scored, 'two people')
# loc &= contains_prompt(scored, 'people chatting and walking')
# loc &= contains_prompt(scored, 'a gathering walking')
# assert loc.any()
# loc[loc].iloc[:4]
#
#
#
#
#
#
#
# # todo: select 4 files that all contain this
