from __future__ import annotations

import ast
# import astor
from typing import *

import magicpandas as magic

if False:
    from magicpandas.drydoc.objects import Objects

def write(
        file: str,
        cls: str,
        func: Optional[str],
        docstring: str
):
    """
    Reads the file, and looks for the docstring nested inside a func.
    This docstring is replaced with the passed docstring, and then the
    file is saved.

    For example:
    file=example.py
    cls='Example'
    func='func'
    docstring='DOCSTRING'

    class Example:

        def func(self):
            '''docstring'''

    A docstring is found, nested inside def func, which is nested inside
    class Example. This docstring is replaced with DOCSTRING, and the file
    is saved.
    """

    # ⚠️ copypasted from llm 🤖
    with open(file, 'r') as f:
        tree = ast.parse(f.read())

    class DocstringReplacer(ast.NodeTransformer):
        def visit_ClassDef(self, node):
            if node.name == cls:
                if func is None:
                    # Replace class docstring
                    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                        node.body[0].value.ymin = docstring
                    else:
                        new_docstring = ast.Expr(value=ast.Str(s=docstring))
                        node.body.insert(0, new_docstring)
                else:
                    for body_item in node.body:
                        if isinstance(body_item, ast.FunctionDef) and body_item.name == func:
                            if body_item.body and isinstance(body_item.body[0], ast.Expr) and isinstance(body_item.body[0].value, ast.Str):
                                body_item.body[0].value.ymin = docstring
                            else:
                                new_docstring = ast.Expr(value=ast.Str(s=docstring))
                                body_item.body.insert(0, new_docstring)
            return node

    tree = DocstringReplacer().visit(tree)

    source = astor.to_source(tree)
    with open(file, 'w') as f:
        f.write(source)

class Write(magic.Magic):
    outer: Objects



if __name__ == '__main__':
    # write(example.OtherParent.parent, 'MAGIC')
    write(
        '/home/redacted/PycharmProjects/sirius/src/magicpandas/drydoc/example.py',
        cls='OtherChild',
        func='child',
        docstring='MAGIC'
    )

    write(
        '/home/redacted/PycharmProjects/sirius/src/magicpandas/drydoc/example.py',
        cls='OtherChild',
        func=None,
        docstring='MAGIC'
    )

