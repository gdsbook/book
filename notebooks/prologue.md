# Prologue

This book provides the tools, the methods, and the theory to meet the challenges of contemporary data science applied to geographic problems and data. Social media, new forms of data, and new computational techniques are revolutionizing social science. In the new world of pervasive, large, frequent, and rapid data, we have new opportunities to understand and analyze the role of geography in everyday life. The book provides the first comprehensive curriculum in geographic data science.

Geographic data is ubiquitous. On the whole, social processes, physical contexts, and individual behaviors show striking regularity in their geographic patterns, structures, and spacing. As data relating to these systems grows in scope, intensity, and depth, it becomes more important to extract meainingful insights from common geographical properties like location, but also how to leverage topological properties like relation that are less commonly-seen in standard data science.

This book introduces a new way of thinking about geographic challenges. Using geographical analysis and computational reasoning, it shows the reader how to unlock new insights hidden within data. The book is structured around the excellent data science environment available in Python, providing examples and worked analyses for the reader to replicate, adapt, extend, and improve.

## Motivation (SR)

### Why this book?
Writing a book like this is a major undertaking, and this suggests the authors must have some intrinsic motivations for taking on such a task. We do. Each of the authors is an active participants in both open source development of spatial analytical tools and academic geographic science. Through our research and teaching, we have come to recognize a need for book to fill a niche that sits at the intersection of GIS/Geography and the world of Data Science. We have seen the explosion of interest in all things Data Science on the one hand and, on the other, the longer standing and continued evolution of GIScience. This book represents our attempt at helping to emerge the intersection between these two fields. It is at that intersection where we believe the intellectual and methodological magic occurs.

### Who is this for?
In writing the book, we envisaged two community of readers who we want to bring together. The first are GIScientists and geographers who may be wondering what all the fuss is about Data Science, and  questioning whether they should engage with the methods, tools, and practices of this new field. Our response to such a reader is an emphatic "Yes!". We see so much to be gained and contributed by geographers who enter these new waters. The second community we have held in mind in writing this material are data scientists who are beginning to turn their attention to working with geographical data. Here we have encountered members of the data science community who are wondering what is so special about geographical data and problems? Data science currently has an impressive array of models and methods, surely these are all that geographers need? Our response to these questions is "No! There is a need for new forms of data science when working with geospatial data." Morever, we see the *collaboration between these two communities as critical to the development of these new advances*.


We also recognize that these two communities are not each a monolithic whole, but are in fact composed of individuals from different sectors, academic science, industry, public sector, and independent researchers, as well as at different career stages. We hope that we have succeeded in providing material that will be of interest to all of these readers.

### What this book isn't
Having described our motivation and intended audience for the book, it is important to point out what the book is not. First, we do not intend the work to be viewed as a GIS starter for data scientists. A number of excellent titles are available that serve that role GET CITES (Henrikki, others). Second, in a similar sense the book is not an introduction to Python programming for GIScientists. Again there are numerous offerings to choose from for the interested reader. GET CITES (Xiao other). Finally, we have conciously choosen breadth over depth in the selection of our topics. Each of the topics we cover are active areas of research of which our treatment should be viewed as providing an entry point to more advanced study. As the admonition goes:

"A couple of months in the laboratory can frequently save a couple of hours in the library." (Frank Westheimer[^1])

Speaking to our intended audiences, geographers new to data science and data scientists new to geography, we hope our book serves as a metaphorical library.

[^1]: Crampon, Jean E. 1988. Murphy, Parkinson, and Peter: Laws for librarians. Library Journal 113. no. 17 (October 15), p. 41.

## Content LJW

Every book reflects a combination of the authors' perspectives and the social and technological context in which the authors write. Thus, we see this book as a core component of the project of codifying *what a geographic data science does*, and (in turn) what kinds of knowledge are important for aspiring geographic data scientists. We also see the *medium* and *method* of writing this book as important for its purpose. So, let's discuss first the content, then the method and medium.

### Overview of content

This book delves throughly into a few core topics of geographic data science. From our background as academic geographers, we seek to present concepts in a more *geographic* way than a standard textbook on *data science*.  This means that we cover spatial data, mapping, and spatial statistics *right* off the bat, and talk at length about some concepts (such as clusters or outliers) in a geographic, not data-scientific, manner. But, as we hope is shown throughout the book, the difference in language and framing is superficial, but the concepts are foundational to both perspectives.

With that in mind, we discuss the central data structures and representations in geographic data science, and then move immediately to visualization and analysis of geographic data. We opt for descriptive *spatial statistics* that summarize the structure of maps, in order to give a sense of how to summarize geographic structure in data science problems. For the analysis sections, we opt for a presentation of a classic *subject* in spatial analysis (inequality), and then pivot to discussing important methods across geographic analysis, such as those that help understand when points are clustered in space, when geographic regions are latent within data, and when geographical spillovers complicate standard supervised learning approaches. The book closes with a discussion of how to use spatial principles to improve standard data scientific algorithms.

### What is not in the book

Despite the strong data science angle we take on the discussion in this book, there are many topics that we omit in our treatment. Every book must exhibit some kind of editorial discipline, but we used three principles to inform our own.

First, we avoided topics that get too complicated too quickly; instead, we sought to maximize the analytical benefits by focusing on simple but meaningful methods of analysis. This precludes many of the more complex but interesting topics and methods, like Bayesian inference or generative models (like cellular automata or agent-based models). GeoAI developments at the cutting edge of quantitative geographic analysis also were excised under this editorial rule. Further, many treatments of the geographical problem of *scale* and *uncertainty* fall in this category, since these questions generally pose issues that demand theoretical, not empirical, solutions.

Second, we sought to discuss things that were intellectually proximate to our own experiences and trainings in quantitative geography. The world of spatial statistics is vast, and very deep, but any one person only gains so much perspetive on it. This strongly informed our decisions of what to cover in the second and third sections, where we generally avoid more complex methods like Gaussian Process (geostatistical) models or geospatial knowledge graph methods.

Third, we sought to avoid topics that already have contemporary treatment in computational teaching. This includes spatial optimization problems (such as location allocation or coverage problems) as well as the generative and geostatistical models mentioned above.

Altogether, these three editorial principles help keep this book focused precisely on the set of techniques that give analysts the most benefit in the shortest space. It covers both methods to summarize and describe geographical pattern, correct analysis for the artefacts induced by geographical structure, and leverage geographical relationships to do better analyses.

### why a book this way

In addition to the content we cover (or omit) from the book, we strongly feel that writing the book *as we have* provides a novel and distinctive utility for our readers. The book is written to develop *everything* shown from scratch. Nearly every graphic has its code included in the book, and is developed directly within the narrative of the book itself. This approach helps illustrate a few things.

First, it facilitates pedagogy...

Geographic Data Science is a new field, but has many academic influences and precursors. Currently, curriculum development is forced to combine snippets of unpublished research code and vignettes together with various packages in the Python data science ecosystem. This book considers students as learners of both analytical and computational methods. We provide accessible, open computational examples with high-quality narrative exposition. This combination is design to satisfy the two main requirements from students:

Second, it provides learners with the narrative scaffolding *around* code that learners need to see in order to integrate their own code with analytical writing.
<!--- - As learners of analytical methods, students want the narrative scaffolding, careful explanation, and intelligible writing provided by typical introductory textbooks. --->

Third, it shows how tightly-coupled code is to analysis, thus demonstrating their strong mutual relationship.

<!--- - As learners of computational approaches, students need worked examples of code alongside an explanation on how to get started even doing analysis.--->

There are many hurdles over which students jump just to start the first example of many introductory textbooks. Our book provides a full view of Geographic Data Science, from setting up and organizing computational environments to preparing data, through to developing novel spatial insight and presenting it cogently to others. This kind of integrated approach is necessary for data science, and is particularly relevant in geographical settings where mapping is of central importance.


- launching research

- continuum from exploration-private (nb-dynamic) to communication-public (dead tree and static) (MacEachren)

<!--- Comments in emails about using jupyterbook

SJR:

- be sure to pin your writing enviornment to specific versions of the
  dependencies. jupyter-book and related has moved fast while whe have
  been writing, and some of the changes can break backwards
  compatibility

- converting from jupyter-book to latex is not fully automatic. a lot of
  tweaking has to happen before you have a final polished version. allow
  for time to do that.

- we simply said we wanted to publish the book online in the
  interactive form in addition to the dead-tree version. and we did that
  during the contract negotiations. the publisher agreed.
  we agreed to not make the tooling publically available to produce a pdf.

>
> Are there any benefits you have been surprised by?

- i think jupyter-book, at least for me, changes the way one
  writes. initially i wasn't crazy about writing with jupyter-book as it
  meant only writing in the notebook, which i find really painful
  compared to emacs with org or tex or markdown.

  but with things like jupytext you can write in markdown and have the
  notebooks derived/synced. this works well, when it does. when there
  are glitches it can cause some downtime getting conflicts sorted.

  writing code+narrative is different from writing say a normal academic
  paper. there is much more iteration between the code and the
  narrative. in this sense, the writing process is different.


DAB:
Technically speaking, Jupyter notebooks are not an ideal form to write long-form and version track. I’ve said it. I wish it wasn’t the case, but it is. We worked around with Jupytext and it made it a lot better, but Levi tried a few times to make us switch to .Rmd files and there were points where I was very close to jumping in… I don’t think it’s a deal breaker, and it’s definitely less and less as things get better (which they are), but I think ignoring it is setting yourself up for non-needed frustration…
The point Levi makes about the philosophical design choices of a book and a notebook is another one that I only realized quite into the book. Looking back, I actually don’t think I’d do it any other way, but I wish this distinction had been clearer to me at the beginning. A similar one is that we were writing effectively the same artifact for a PDF/paper book and a website. And the layout choices, what one allows and hinders can be contradicting. Sometimes, you need to choose one while knowing it’s hurting the other one. Again, I think it is still worth navigating that tension because you’re basically reaching two (different) audiences for (mostly) the price of one. But keeping it in mind is useful.
The point about the stack is an important one. We picked a development container and, although we haven’t made it perfectly (the container itself has changed a few times, we can’t always develop on it, etc.) I think overall has simplified the process. Find a way in which everyone is on comparable stacks because this stuff changes so fast and at times very notably and it can be a great distraction from actually writing the book…
I think final point is just to say the obvious, which is that the fact you can write a book from something like notebooks is such a cool thing and it is possible. Sure thing, it’s not perfect yet and there are parts that really feel uncomfortable. But, pedagogically speaking, I think it opens up very interesting possibilities that allow you to take the enterprise of writing a book into different domains (e.g., even before it’s published, we’ve done workshops pretty much on-the-fly with earlier drafts).

LJW:

Aside from the mechanics of using jupyter book (which, as Serge mentions, are getting better all the time), one thing that I think is useful to keep in mind here is the difference between writing notebooks for use and writing textbooks for use.

Notebooks are fundamentally iterative. There’s a lot of printing, plotting, and “show”ing that derive from their basic status as an enhanced REPL. Thus, narratives built using notebooks generally are oriented towards the building-up of many small, logically-connected components. Everything you show or do has to be built up from things that the reader can (generally) see. There is no deus ex machina when you’re inside the machine!

Books are not iterative in the same fashion. You can bring in things “for free” (like very complex visualizations of difficult or confusing concepts)… that is, you don’t have to build them along with your reader within the text. So… the idea of keeping the code close to the presentation is a big departure from how textbooks “work” as a medium. [1]

So, be mindful of how the medium will frame the message when you’re writing a book in notebooks. And, think/talk to each other about what ought to be built “in front of the reader” in your book before you start writing!

[1]: I know that Jupyterbook can do things like hiding cells (which kind of allow for this), but this was not available when we started our book, and the canonical representation of the notebook (the .ipynb itself) will always show the more complicated bits.

--->

## The book with the perspective of future


### acknowledgement/caveat emptor that the code will change in the future.

- Python for data is very fast-evolving landscape
- point about interoperability R v Python v Julia going to R+Python+Julia (polyglot analyses, cell magics that allow for multiple languages/reticulate)

### Concerns at the time of the book

- Writing this book as we have done it was impossible ten years ago
- It was barely possible when we started writing it
    - Why it was hard
    - Which things changed
    -
    - How did the ecosystem improved
- Hopefully in ten years, this will be standard, robust tooling with community consensus about how to do it properly

