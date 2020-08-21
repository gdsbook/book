#!/usr/bin/env python
# coding: utf-8


import glob
import re
import os.path
FILE_NAMES = glob.glob("notebooks/*.md")
TEST_STR = "---\n"
NEW_HEADER = """---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: 1.5.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
"""

for file_name in FILE_NAMES:
    with open(file_name) as file:
        content = file.read()
        res = [i.start() for i in re.finditer(TEST_STR, content)]
        if len(res) > 1:
            new_content = NEW_HEADER + content[res[1]+4:]
            new_content = new_content.replace("```python",
                                              "```{code-cell} ipython3")
        else:
            new_content = content
    tmp, file_name = os.path.split(file_name)
    with open(f"book/{file_name}", 'w') as output_file:
        output_file.write(new_content)
