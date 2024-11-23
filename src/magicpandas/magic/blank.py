class Blank:
    """
    Blank allows you to inherit from a class while preventing the
    methods and attributes from showing up in code completion.
    """
    def __get__(self, instance, owner: type) -> type:
        return owner



