---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Computational Tools for Geographic Data Science

In this tutorial, we will introduce the main tools we will be working with
throughout the rest of the book. Although very basic and seemingly abstract,
everything showed here will become the basis on top of which we will build more
sophisticated (and fun) tasks. But, before, let us get to know the tools that
will power our data science. 

## Open Source Software

This course will introduce you to a series of computational tools that make the
life of the Data Scientist possible, and much easier. All of them are [open-
source](https://en.wikipedia.org/wiki/Open_source), which means the creators of
these pieces of software have made available the source code for people to use
it, study it, modify it, and re-distribute it. This has produced a large eco-
system that today represents the best option for scientific computing, and is
used widely both in industry and academia. Thanks to this, this course can be
taught with entirely freely available tools that you can install in any of your
computers.

If you want to learn more about open-source and free software, here
are a few links:

* **[Video]**: brief
[explanation](https://www.youtube.com/watch?v=Tyd0FO0tko8) of open source.
* **[Book]** [The Cathedral and the Bazaar](https://en.wikipedia.org/wiki/The_Cathedral_and_the_Bazaar): classic
book, freely available, that documents the benefits and history of open-source
software.

## `Jupyter` Notebook

The main computational tool you will be using during this course is the [Jupyter
notebook](http://jupyter.org/). Notebooks are a convenient way to thread text,
code and the output it produces in a simple file that you can then share, edit
and modify. You can think of notebooks as the Word document of Data Scientists,
just much nicer.

### Start a notebook

Jupyter notebook is an app that must be started from a *command line*, a text-based
interface that allows you to interact directly with programs through written
commands. This is how you can fire up a terminal:

* If you are
on a **Windows** computer, you can start the "Anaconda Command Prompt" from the
Start menu. 
* On a **Mac**, use the Terminal.app, in Utilities. 
* In **Linux**, use any of the terminals available, such as GNOME Terminal, 
Konsole, or XTerm, or any other terminal you find easy to use. 

Once the terminal is opened, you should be greeted with a text interface, like other
common text entry fields. There, type the following command and press enter:

`source activate gds`

**NOTE**: ignore `source` if you are on Windows and
simply type `activate gds`.

Then, after you run that command, launch `Jupyter` by typing the following 
command and pressing enter again:

`jupyter notebook`

This should bring up a web browser window
with a home page that shows a list of files and folders in the location where the 
`jupyter notebook` was started. It should look something like like this:

![Jupyter home](figs/lab01_jupyter_home.png)

Clicking on the folders in the file browser, go to the folder where you have 
placed the `lab_01.ipynb` file for this tutorial and click on it. 
This will open the notebook on a different tab. You are now in the notebook, which
you can both run like a computer program and edit like a word document. 

Whenever you want to save your work, you can save the notebook using
`File -> Save and Checkpoint`. Everything you do in the notebook 
(text, code, and the output from code) is saved into a single `.ipynb` file
that you can open again to edit and run, share with others, or publish online. 

### Cells

The main building block of Jupyter notebooks are called *cells*. These are 
chunks of content that is all of the same "type." Cells
can be of two types:

* **Text**, like the one where this is written.
*
**Code**, like the following one below:

```python attributes={"classes": [], "id": "", "n": "1"}
# This is a code cell
```

You can create a new cell by clicking `Insert` -> `Cell Above`/`Below` in the
top menu. By default, this will be a code cell, but you can change that on the
`Cell` -> `Cell Type` menu. Choose `Markdown` for a text cell. Once a new cell
is created, you can edit it by clicking on it, which will create the cursor bar
inside for you to start typing. Cells can also be cut, pasted, moved, inserted,
and deleted in a notebook. Think of them like computational paragraphs that can
be mixed around to form larger or longer thoughts.

<div class="alert alert-info" style="font-size:110%"> <b>Pro tip</b>: cells can also be created with
shortcuts. If you press the 'escape' key and then the 'b' key (or 'a' key), a new cell will be
created below (or above) the current cell. There is a whole bunch of shortcuts you can explore by
pressing 'escape' and 'h' (press 'escape' again to leave the help).
</div>
### Code and its output

A particularly useful feature of notebooks is that you
can save, in the same place, the code you use to generate any output (tables,
figures, etc.). As an example, the cell below contains a snipet of Python that
returns a printed statement. This statement is then printed below and recorded
in the notebook as output:

```python attributes={"classes": [], "id": "", "n": "2"}
print("Hello, world!")
```

<!-- #region -->
Note how the notebook automatically colors the Python code in the code block?
This makes the code much more readable and understandable. More on Python below.

### Markdown

Text cells in a notebook a *markup* language, [Github Flavored
Markdown](https://help.github.com/articles/github-flavored-markdown/), to 
express many commonly-used structures in documents. One strength of Markdown is 
that it is written in plain, flat text, but is rendered into something that looks
more structured. The notebook does this automatically, rendering the more 
visually appealing version any markdown you write. Let's see some examples:

#### Text formatting

Bold an italic text can be obtained using asterisks. For example:

`This is **bold**.`

is rendered:

This is **bold**.

and

`This is *italic*.`

is rendered:

This is *italic*.

#### Lists

You can create bullet lists. For example:

`* Item 1`
`* Item 2`
`* ...`

will produce:
* Item 1
* Item 2
* ...

Or, you can create numbered lists using


`1. First element`
`1. Second element`
`1. ...`

which renders into:

1. First element
1. Second element
1.
...

Note that you don't have to write the actual number of the element. Markdown will
keep track of how many elements are in a given list and number them appropriately.
So, if you want, you can simply use `1.` for each element while writing, and Markdown
will render it into a simple sequence of numbered entries. 

You can also nest lists:

`*First unnumbered element, which can be split into:`
``
`  1. One numbered element`
`  2. Another numbered element`
``
`* Second element.`
`* ...`

* First unnumbered element, which can be split into:

  1. One numbered element
  2. Another numbered element

* Second element.
* ...

This creates many opportunities to combine document elements in a nice way. 

#### Hyperlinks

Markdown allows for hyperlinks to be inserted into text inline using a 
combination of brackets and parentheses. For example:

`Consult [Wikipedia](https://www.wikipedia.org/) for more information.`

becomes

Consult [Wikipedia](https://www.wikipedia.org/) for more information.

#### headings

including `#` before a line causes it to render a heading.

---
`# This is Header 1`

Turns into:

# This is Header 1

---

`## This is Header
2`

Turns into:

## This is Header 2

---

`### This is Header 3`

Turns into:
### This is Header 3

And so on.

---

More information about markdown is provided by:

[https://help.github.com/articles/markdown-basics](https://help.github.com/articles/markdown-basics)

[https://help.github.com/articles/github-flavored-markdown](https://help.github.com/articles/github-flavored-markdown)

### Rich content in a notebook

Notebooks can also include rich content from the web. For this, we need to
import the `display` module from the
<!-- #endregion -->

```python attributes={"classes": [], "id": "", "n": "3"}
import IPython.display as display
```

This makes available additional functionality that allows us to embed rich
content. For example, we can include a YouTube clip by passing the video ID:

```python attributes={"classes": [], "id": "", "n": "4"}
display.YouTubeVideo('iinQDhsdE9s')
```

Or we can pass standard HTML code:

```python attributes={"classes": [], "id": "", "n": "5"}
display.HTML("""<table>
<tr>
<th>Header 1</th>
<th>Header 2</th>
</tr>
<tr>
<td>row 1, cell 1</td>
<td>row 1, cell 2</td>
</tr>
<tr>
<td>row 2, cell 1</td>
<td>row 2, cell 2</td>
</tr>
</table>""")
```

Note that this opens the door for including a large number of elements from the
web, since an `iframe` of any other website can also be included. For example, 
interactive maps can be shown from within an `iframe`:

```python attributes={"classes": [], "id": "", "n": "6"}
osm = """
<iframe width="425" height="350" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="http://www.openstreetmap.org/export/embed.html?bbox=-2.9662737250328064%2C53.400500637844594%2C-2.964626848697662%2C53.402550738394034&amp;layer=mapnik" style="border: 1px solid black"></iframe><br/><small><a href="http://www.openstreetmap.org/#map=19/53.40153/-2.96545">View Larger Map</a></small>
"""
display.HTML(osm)
```

Sound content can also be included by providing an `iframe` over an audio website:

```python attributes={"classes": [], "id": "", "n": "7"}
sound = '''
<iframe width="100%" height="450" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/178720725&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false&amp;visual=true"></iframe>
'''
display.HTML(sound)
```

A more thorough exploration of these forms of rich content is available in
[this](http://jeffskinnerbox.me/notebooks/ipython's-rich-display-system.html)
notebook.

### Exercise to work on your own

Try to reproduce, using markdown and the different tools the notebook affords
you, the following Wikipedia entry:

[https://en.wikipedia.org/wiki/Chocolate_chip_cookie_dough_ice_cream](https://en.wikipedia.org/wiki/Chocolate_chip_cookie_dough_ice_cream).

```python attributes={"classes": [], "id": "", "n": "8"}
display.IFrame('https://en.wikipedia.org/wiki/Chocolate_chip_cookie_dough_ice_cream', 
              700, 500)
```

Pay special attention to getting the bold, italics, links, headlines and lists
correctly formated, but don't worry too much about the overall layout. Bonus if
you manage to insert the image as well!

## Python

The main component of this book relies on the [Python](https://www.python.org/)
programming language. Python is a [high-
level](https://en.wikipedia.org/wiki/High-level_programming_language)
programming language used widely in data science. To give a couple of examples of its
relevance, it powers [most of the company Dropbox's main product](https://www.quora.com/How-
does-dropbox-use-python-What-features-are-implemented-in-it-any-tangentially-
related-material?share=1), and is also heavily
[used](https://www.python.org/about/success/usa/) to control satellites at NASA.
A great deal of Science is also done in Python, from [research in
astronomy](https://www.youtube.com/watch?v=mLuIB8aW2KA) at UC Berkley, to
[courses in economics](https://lectures.quantecon.org/py/) by Nobel Prize-winning professors. 

This book uses Python because it is a good language for beginners and 
high performance science alike. For this reason, it has emerged as one of the main
and most solid options for Data Science. Python is widely used for data 
processing and analysis both in academia and in industry. There is a vibrant and 
growing scientific community (through the [Scientific Python](http://scipy.org/) 
library and the [PyData](http://pydata.org/) organization), working in
both universities and companies, to support and enhance the Python's capabilities.
New methods and usability improvements of existing packages (also known as libraries)
are continuously being released. In geocomputation, Python is also very widely
adopted: it is the language used for scripting in both the main proprietary enterprise
geographic information system,
[ArcGIS](http://www.esri.com/software/arcgis), 
and the leading open geographic information system, [QGIS](http://qgis.org). All
of this means that, whether you are thinking of continuing in Higher Education
or trying to find a job in industry, Python will be an important asset, valuable to
employers and scientists alike. 

Python code is "dynamically interpreted", which means it is run on-the-fly without 
needing to be compiled. This is in contrast to other kinds of programming
languages, which require an additional non-interactive step where a program is 
converted into a binary file, which is then run directly. With Python, one does
not need to worry about this non-interactive compilation step. Instead, we can 
simply write code, run code, fix any issues directly, and then re-run the code in a
quick cycle. This makes Python a very productive tool for science, since you can
prototype code quickly and directly. The rest of this tutorial covers
some of the basic elements of the language, from conventions like how to comment
your code, to the basic data structures available.

### Python libraries

The standard Python language includes some data structures (such as lists and
dictionaries) and allows many basic mathematical operations (e.g. sums, differences,
products). For example, right out of the box, and without any further
action needed, you can use Python as a calculator:

```python attributes={"classes": [], "id": "", "n": "9"}
3 + 5
```

```python attributes={"classes": [], "id": "", "n": "10"}
2 / 3
```

```python attributes={"classes": [], "id": "", "n": "11"}
(3 + 5) * 2 / 3
```

However, the strength of Python as a data analysis tool comes from additional
software that adds functionality to the language itself. This additional software
provides many more useful data structures and functions for data science. These
come in the form of packages, also known as libraries, need to be installed separately 
and must be explicitly included in order to be used. 
In this course, we will be using many of the core libraries of the "PyData stack",
a set of heavily-used libraries that make Python a fully-fledged
system for Data Science. We will introduce each package only as we need it for
specific tasks. For now, though, let us have a look at the foundational library,
[numpy](http://www.numpy.org/) (short for **num**erical **Py**thon). Bringing 
additional Python libraries into a session is called *importing* them, and is 
done using the `import` statement:

```python attributes={"classes": [], "id": "", "n": "12"}
import numpy as np # we rename it in the session as `np` by convention
```

Note how we `import numpy`, introducing it into our Python session, *and* rename it
in the session, suggesting that `numpy` be known as `np`, which is shorter and more convenient.

Note also how comments work in Python:
everything in a line *after* the `#` sign is ignored by Python when it evaluates
the code. This allows you to insert comments that Python will ignore but that
can help make your code more clear.

Once imports are out of the way, let us start exploring what we can do with
`numpy`. One of the most basic tasks is to create sequences of numbers:

```python attributes={"classes": [], "id": "", "n": "13"}
seq = np.arange(10)
seq
```

The first thing to note is that, in line 1, we create the sequence by calling
the function `arange` and assign it to an object called `seq`. `seq` is just a
name we choose to call the result of the computation `np.arange(10)`. There are 
many valid names we can use for variables in Python. Nearly any name is valid in Python, so
long as they:
- start with an alphabetical letter (e.g. `seq1` is valid, but `1seq` is not)
- have no additional characters aside from alphanumerics and underscores
Finally, the last line of the cell above prints the contents of `seq`. In Jupyter 
notebooks, the contents of the last line in a cell will be shown if they are not
assigned to a variable. 

Another interesting feature of Python is how it keeps track of what functionality
comes from `numpy`, and what functionality dos not. Since we
are calling a `numpy` function (`arange`), we put `np.` in front of `arange`. This
means that we are using the `arange` function from inside the `np` *namespace*.
This means that the function `arange` comes explicitly from `numpy`. To find out how
necessary this is, you can try generating the sequence without `np`:

```python attributes={"classes": [], "id": "", "n": "14"}
# NOTE: comment out to run the cell
#seq = arange(10)
```

What you get instead is an error, also called a "traceback". In particular,
Python is telling that it cannot find a function named `arange` in the core
library. This is because that particular function is only available in `numpy`,
so we have to look under `np` to find `np.arange`. 

### Variables

A fundamental feature of Python is the ability to assign a name to different "things",
or objects. These objects are also sometimes called "variables" as well. We have already seen
one variable (`seq`) in the example above but let's make things more explicit. 
For example, an object can be a single number.:

```python attributes={"classes": [], "id": "", "n": "15"}
a = 3
```

Now, we've assigned an object (the number `3`) to a variable, `a`. 

Words or sentences can also be objects. In Python, we call this kind of text a "string,"
and they are marked off from other text using either single or double quotation marks:

```python attributes={"classes": [], "id": "", "n": "16"}
b = 'Hello World'
```

You can check what "type" of object is stored in a variable using the `type` function:

```python attributes={"classes": [], "id": "", "n": "17"}
type(a)
```

Here, `int` is short for "integer" which, roughly speaking, means a whole number. 
In most cases, a number with a decimal in Python is called a "floating-point number,"
or a "float" for short:

```python attributes={"classes": [], "id": "", "n": "18"}
c = 1.5
type(c)
```

As mentioned, what we understand as text in a wide sense (spaces and other
symbols count as well) is called a "string" (`str` for short):

```python attributes={"classes": [], "id": "", "n": "19"}
type(b)
```

### Help

A very handy feature of Python is the ability to access on-the-spot help for functions. 
This means that you can check what a function is supposed to do, or how to use it
from directly inside your Python session. Of course, this
also works handsomely inside a notebook, too. There are a couple of ways to access
the help. 

Take the `numpy` function `arange` that we have used above. The
easiest way to check its help dialog from within the notebook is to add a question
mark after it:

```python attributes={"classes": [], "id": "", "n": "20"}
np.arange?
```

As you can see, this brings up a sub-window in the browser with all the
information you need.

If, for whatever reason, you needed to print that info
into the notebook itself, you can use the following `help` function instead:

```python attributes={"classes": [], "id": "", "n": "21"}
help(np.arange)
```

### Control flow (a.k.a. `for` loops and `if` statements)

Although this book is not intend to be a comprehensive introduction to computer
programming or to general purpose Python (check the references for that, in
particular Allen Downey's
[book](http://www.greenteapress.com/thinkpython/thinkpython.html)), it is
important to be aware of two building blocks of almost any computer program:
`for` loops and `if` statements. It is possible that you will never require them
for this book. This book is mainly based on existing methods and
functions, but it is always useful to know how `for` and `if` work to be able to
recognize them. They can also come in very handy in cases where you some extra
functionality out of standard methods. So, let us have a look
at these two flow control methods. 

#### `for` loops

Loops allow you to repeat a particular action or set of actions. In this case, 
`for` loops allow you repeat an action over every element in a collection. For 
example, you could print your name ten times without having to type it yourself every single time:

```python attributes={"classes": [], "id": "", "n": "22"}
for i in np.arange(10):
    print('my name')
```

A couple of features in the loop are useful to note:

1. Loops are conducted *over* a sequence. In this
particular case, we loop over the sequence of ten numbers created by `np.arange(10)`.
1. In
every step, for every element of the sequence in this case, you repeat an
action. Here we are printing the same text, `my name`.
1. Each of the elements you loop over can be accessed inside of 
the loop as well. Although we did not use this feature in the loop above, this 
can be extremely useful in some cases. For example, we could iterate over a list of
names:

```python attributes={"classes": [], "id": "", "n": "23"}
names = ['Jane', 'Robert', 'Elise', 'Reilly']
for name in names:
    print("Whose turn is it?", name)
```

One more thing to note: it often makes sense to describe what you are iterating
over directly like we did in `for name in names`. But, remember, `name` is arbitrary, 
and we could have called it any valid Python variable name.

#### `if` statements

We have just seen how `for` loops allow you to repeat an
action over a sequence. However, sometimes you might not want to take an action
every single time. Often, we might want to run a bit of code only if some conditions
are satisfied. This "branching" behavior is provided by `if` statements, which
select or restrict actions to only run when a condition (or many conditions) are met.

For example, if you think of the loops written above,
we might want to skip players whose name begins with 'R':

```python attributes={"classes": [], "id": "", "n": "25"}
for name in names:
    if (name.startswith("R")):
        print("We are skipping ", name)
```

We can also take an action only when our `if` statement is *not* satisfied by 
using an `else` statement. 
Together, these are sometimes called "ifelse" statements. For example,
we could skip players whose names begin with "R", but allow everyone else to go:

```python attributes={"classes": [], "id": "", "n": "26"}
for name in names:
    if (name.startswith("R")):
        print("We are skipping ", name)
    else:
        print(name, "gets to go")
```

### Data structures

The standard python you can access without importing any additional libraries
contains a few core data structures that are very handy to know. Most of data
analysis is done on top of other structures specifically designed for the
purpose (`numpy` arrays and `pandas` dataframes, mostly. See the following sessions
for more details), but some understanding of these core Python structures is
very useful. In this context, we will look at three: values, lists, and
dictionaries.

#### Values
These are the most basic elements to organize data and information
in Python. You can think of them as numbers (integers or floats) or words
(strings). Typically, these are the elements that will be stored in lists and
dictionaries.

An integer is a whole number:

```python attributes={"classes": [], "id": "", "n": "27"}
i = 5
type(i)
```

A float is a number that allows for decimals:

```python attributes={"classes": [], "id": "", "n": "28"}
f = 5.2
type(f)
```

Note that a float can also not have decimals and still be stored as such:

```python attributes={"classes": [], "id": "", "n": "29"}
fw = 5.
type(fw)
```

However, they are different representations:

```python attributes={"classes": [], "id": "", "n": "30"}
f == fw
```

#### Lists
A list is an ordered sequence of values that can be of mixed types.
They are represented between squared brackets (`[]`) and, although not very
efficient in memory terms, are very flexible and useful to "put things
together".

For example, the following list of integers:

```python attributes={"classes": [], "id": "", "n": "31"}
l = [1, 2, 3, 4, 5]
l
```

```python attributes={"classes": [], "id": "", "n": "32"}
type(l)
```

Or the following mixed one:

```python attributes={"classes": [], "id": "", "n": "33"}
m = ['a', 'b', 5, 'c', 6, 7]
m
```

Lists can be queried and sliced. For example, the first element can be retrieved
by:

```python attributes={"classes": [], "id": "", "n": "34"}
l[0]
```

Or the second to the fourth:

```python attributes={"classes": [], "id": "", "n": "35"}
m[1:4]
```

Lists can be added:

```python attributes={"classes": [], "id": "", "n": "36"}
l + m
```

New elements added:

```python attributes={"classes": [], "id": "", "n": "37"}
l.append(4)
l
```

Or modified:

```python attributes={"classes": [], "id": "", "n": "38"}
l[1]
```

```python attributes={"classes": [], "id": "", "n": "39"}
l[1] = 'two'
l[1]
```

```python attributes={"classes": [], "id": "", "n": "40"}
l
```

#### Dictionaries
Dictionaries are unordered collections of "keys" and
"values". A key, which can be of any kind, is the element associated with a
"value", which can also be of any kind. Dictionaries are used when order is not
important but you need fast and easy lookup. They are expressed in curly
brackets, with keys and values being linked through columns.

For example, we
can think of a dictionary to store a series of names and the ages of the people
they represent:

```python attributes={"classes": [], "id": "", "n": "41"}
ages = {'Ana': 24, 'John': 20, 'Li': 27, 'Ivan': 40, 'Tali':33}
ages
```

```python attributes={"classes": [], "id": "", "n": "42"}
type(ages)
```

Dictionaries can then be queried and values retrieved using their
keys. For example, if we quickly want to know Li's age:

```python attributes={"classes": [], "id": "", "n": "43"}
ages['Li']
```

Similarly to lists, you can modify and assign new values:

```python attributes={"classes": [], "id": "", "n": "44"}
ages['Juan'] = 73
ages
```

Using this property, you can create entirely empty dictionaries and fill them 
with values later:

```python attributes={"classes": [], "id": "", "n": "45"}
newdict = {}
newdict['key1'] = 1
newdict['key2'] = 2
newdict
```

### Functions

The last part of this whirlwind tour on Python relates to functions, sometimes
also called methods. Functions are the most basic unit of re-usable code.
So far, we have only seen Python code that must be copied in its entirety in order
to be used again. However, as we will see in more detail later in
the course, one of the main reasons why you want to use Python for data
analysis instead of a point-and-click graphical interface like SPSS is that you 
can easily reuse code and re-run analyses easily. Code that does one thing can
be written *one time*, in one place, and run wherever and whenever it is needed.
Methods/functions help us accomplish this by encapsulating pieces of code that
perform a one specific task, protecting them from changes that we might make 
in other Python sessions or in other analyses. 
Thus, a function provides a clean, separate space to describe one small piece
of a larger analysis, and can simply be plugged into any other analysis. 

We have already *used*
methods here. When we call `np.arange`, we are using the `arange` method in `numpy`. 
Now, we will see how to *create* a method of our own that performs one specific task we want.
For example, let us create a very simple method to reproduce the first
loop we created above:

```python attributes={"classes": [], "id": "", "n": "46"}
def run_simple_loop():
    for i in np.arange(10):
        print(i)
    return None
```

Already with this simple method, there is a bunch of interesting things going
on:

* First, note how we define a bit of code is a method, as oposed to plain
Python: we use `def` followed by the name of our function (we have chosen
`run_simple_loop`, but we could have chosen any valid python name. 
* Second, we append `()`
after the name, and finish the line with a colon (`:`). This is necessary and
will allow us to specify requirements for the function (see below).
* Third,
realize that everything inside a function needs to be indented. This is a core
property of Python and, although some people find it odd, it enhances
readability greatly. 
* Fourth, the piece of code to do the task we want,
printing the sequence of numbers, is inside the function in the same way it was
outside, only properly indented. Everything that is indented is considered "within" the method,
and anything that is not indented will be considered "outside" of the method.
* Fifth, we finish the method with a `return` statement. In this case, we are 
returning a special value, `None`, but this will
change as methods become more sophisticated. Essentially, this is the part of
the method where you specify which elements you want it to return and save for
later use.

Once we have paid attention to these elements, we can see how the
method can be *called* and hence the code inside it executed:

```python attributes={"classes": [], "id": "", "n": "47"}
run_simple_loop()
```

This is the same way that we called `np.arange` before. Note how we do not
include the colon (`:`). Instead, when we call a method, we only use the name of
the method followed by the parenthesis.

This is a very simple method. It does not interact with the rest of our program.
We can just execute it without providing any additional information, and the
code prints what we want it to and returns `None`. The rest of this section 
relaxes these two aspects to allow us to build more complex, 
but also more useful, methods.

First, methods can take "arguments," additional information that the method will use. 
You can specify these arguments when the method is defined. Remember how
we called `np.arange` and provided a number that determined the length of the sequence we
wanted? We can do the same thing in our own function. The main aspect
to pay attention to in this context is that the arguments are themselves *variables*,
and not values. This is because they are simply names that *stand in* for whatever
we will eventually provide to the function.

For example, we can modify our method:

```python attributes={"classes": [], "id": "", "n": "48"}
def run_simple_loopX(x):
    for i in np.arange(x):
        print(i)
    return None
```

We have replaced the fixed length of the sequence (10) by a variable named `x`
that allows us to specify *any value we want* when we call the method:

```python attributes={"classes": [], "id": "", "n": "49"}
run_simple_loopX(3)
```

```python attributes={"classes": [], "id": "", "n": "50"}
run_simple_loopX(2)
```

Another way you can build more flexibility into a method is by allowing it to
return an output directly, instead of returning `None`. In the previous examples, our function
performs a computation (i.e. printing values on the screen), but it does not
return any value. This is in contrast with, for example, `np.arange` which does
return an output, the sequence of values:

```python attributes={"classes": [], "id": "", "n": "51"}
a = np.arange(10)
```

```python attributes={"classes": [], "id": "", "n": "52"}
a
```

Our function does not save anything:

```python attributes={"classes": [], "id": "", "n": "53"}
b = run_simple_loopX(3)
```

```python attributes={"classes": [], "id": "", "n": "54"}
b
```

We can modify this using the last line of a method. For example, let us assume
we want to return a sequence as long as the series of numbers we print on the
screen. One way to do this would be:

```python attributes={"classes": [], "id": "", "n": "55"}
def run_simple_loopXout(x):
    for i in np.arange(x):
        print(i)
    return np.arange(x)
```

Note that the main difference now is that instead of returning `None`, we return
the sequence we used in the `for` loop. We could be even more efficient, though,
by assigning the sequence to a new object *inside of the method*, and using it first 
in the loop and then returning it. The results are exactly the same, but there are less computations
performed, since we only build the sequence one time. More critically, 
we minimize the chances of making mistakes by referring to the same object every time:

```python attributes={"classes": [], "id": "", "n": "60"}
def run_simple_loopXout(x):
    seq = np.arange(x)
    for i in seq:
        print(i)
    return seq
```

Either of these two new versions of the method return an output:

```python attributes={"classes": [], "id": "", "n": "61"}
a = run_simple_loopX(3)
b = run_simple_loopXout(3)
```

```python attributes={"classes": [], "id": "", "n": "62"}
a
```

```python attributes={"classes": [], "id": "", "n": "63"}
b
```

The advantage of methods, as oposed to straight code, is that they force us to
think in a modular way, helping us identify the small components of what what we 
are doing in our analysis overall.  Encapsulating these little atoms of
functionality inside of methods allows us to write this functionality one time and
use them everywhere. This code reuse saves us time and headaches in the long run.

One final note on methods. It is important that, whenever you create a method,
you include some documentation about what arguments it requires, what the method
does more generally, and what values it returns. This is called the *docstring* 
of a function, and it looks like one big quote that comes immediately following the 
`def` statement. Although there are many ways formatting a docstring, one common format
is:

```python attributes={"classes": [], "id": "", "n": "64"}
def run_simple_loopXout(x):
    """
    Print out the values of a sequence of certain length
    ...
    
    Arguments
    ---------
    x     : int
            Length of the sequence to be printed out
    
    Returns
    -------
    seq   : np.array
            Sequence of values printed out
    """
    seq = np.arange(x)
    for i in seq:
        print(i)
    return seq
```

Docstrings, like other strings, is colored red in the notebook by default. Let us have
a look at the structure and components of a well-made docstring:

* It is encapsulated between triple commas (`"""`).
* It begins with
a short description of what the method does. The shorter the better, the more
concise, the even better.
* There is a section called "Arguments" that lists each
element that the function expects. 
* Each argument is then listed, followed by
its type. In this case it is an object `x` that, as we are told, needs to be an
integer.
* The arguments are followed by another section that specifies what the
function returns, and of what type the output is.

Docstrings are
very useful to remember what a function does. They also to force you to
write clearer code. A bonus is that, if you include documentation in this way,
it can be checked with the standard `help` or `?` systems reviewed above:

```python attributes={"classes": [], "id": "", "n": "65"}
run_simple_loopXout?
```

```python attributes={"classes": [], "id": "", "n": "66"}
help(run_simple_loopXout)
```

### Exercise to work on your own

Write a properly documented function that:

1. Has a single argument, an integer.
2. Creates a sequence of that length and an empty dictionary. 
3. Then, loops over that sequence and
  1. checks if each number is even or odd (Hint: `number % 2` will be `1` when `number` is odd, but zero when `number` is even.)
  2. If the number is even, it should be stored in the dictionary as a key, with a value of "even"
  3. If the number is odd, it should be stored in the dictionary as a key, with a value of "odd."
4. Returns the now-filled dictionary.


<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
