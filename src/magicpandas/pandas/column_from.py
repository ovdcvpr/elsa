# todo:
"""
@magic.column
def ntrue(self):
    result = (
        self.groups.ntrue
        .indexed_on(self.igroup.values)
        .values
    )
    return result

@magic.column.indexed_on(groups, igroup)
def ntrue(self):
    ...

@magic.align(groups).on(igroup)
def ntrue(self):
    ...

@magic.column_from(groups).using(igroup)
def ntrue(self):
    ...
"""
