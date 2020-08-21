#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os.path


# In[2]:


file_names = glob.glob("notebooks/*.md")


# In[3]:


file_names


# In[5]:


import re
test_str = "---\n"


# In[6]:


new_header = """---
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


# In[7]:


for file_name in file_names:
    with open(file_name) as file:
        content = file.read()
        res = [i.start() for i in re.finditer(test_str, content)] 
        if len(res) > 1:
            new_content = new_header + content[res[1]+4:]
            new_content = new_content.replace("```python","```{code-cell} ipython3")
        else:
            new_content = content
    
    tmp, file_name = os.path.split(file_name)
    with open(f"book/{file_name}", 'w') as output_file:
        output_file.write(new_content)





