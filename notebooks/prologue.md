# Prologue

This book provides the tools, the methods, and the theory to meet the challenges of contemporary data science applied to geographic problems and data. Social media, new forms of data, and new computational techniques are revolutionizing social science. In the new world of pervasive, large, frequent, and rapid data, we have new opportunities to understand and analyze the role of geography in everyday life. The book provides the first comprehensive curriculum in geographic data science.

Geographic data is ubiquitous. On the whole, social processes, physical contexts, and individual behaviors show striking regularity in their geographic patterns, structures, and spacing. As data relating to these systems grows in scope, intensity, and depth, it becomes more important to extract meainingful insights from common geographical properties like location, but also how to leverage topological properties like relation that are less commonly-seen in standard data science.

This book introduces a new way of thinking about geographic challenges. Using geographical analysis and computational reasoning, it shows the reader how to unlock new insights hidden within data. The book is structured around the excellent data science environment available in Python, providing examples and worked analyses for the reader to replicate, adapt, extend, and improve.

## Motivation (SR)

### Why this book?
Writing a book like this is a major undertaking, and this suggests the authors must have some intrinsic motivations for taking on such a task. We do. Each of the authors is an active participant in both open source development of spatial analytical tools and academic geographic science. Through our research and teaching, we have come to recognize a need for a book to fill the niche that sits at the intersection of GIS/Geography and the world of Data Science. We have seen the explosion of interest in all things Data Science on the one hand and, on the other, the longer standing and continued evolution of GIScience. This book represents our attempt at helping to emerge the intersection between these two fields. It is at that common ground where we believe the intellectual and methodological magic occurs.

### Who is this for?
In writing the book, we envisaged two communities of readers who we want to bring together. The first are GIScientists and geographers who may be wondering what all the fuss is about Data Science, and  questioning whether they should engage with the methods, tools, and practices of this new field. Our response to such a reader is an emphatic "Yes!". We see so much to be gained and contributed by geographers who enter these new waters. The second community we have held in mind in writing this material are data scientists who are beginning to turn their attention to working with geographical data. Here we have encountered members of the data science community who are wondering what is so special about geographical data and problems. Data science currently has an impressive array of models and methods, surely these are all that geographers need? Our response to these questions is "No! There is a need for new forms of data science when working with geospatial data." Morever, we see the *collaboration* between these two communities as *critical* to the development of these new advances.


We also recognize that neither of these two communities is a monolithic whole, but are in fact composed of individuals from different sectors, academic science, industry, public sector, and independent researchers, as well as at different career stages. We hope this book provides material that will be of interest to all of these readers.

### What this book isn't

Having described our motivation and intended audience for the book, we find it useful to also point out what the book is not. First, we do not intend the work to be viewed as a GIS starter for data scientists. A number of excellent titles are available that serve that role GET CITES (Henrikki, others). Second, in a similar sense the book is not an introduction to Python programming for GIScientists. Again there are numerous offerings to choose from for the interested reader. GET CITES (Xiao other). Finally, we have conciously choosen breadth over depth in the selection of our topics. Each of the topics we cover are active areas of research of which our treatment should be viewed as providing an entry point to more advanced study. As the admonition goes:

"A couple of months in the laboratory can frequently save a couple of hours in the library." (Frank Westheimer[^1])

Speaking to our intended audiences, geographers new to data science and data scientists new to geography, we hope our book serves as a metaphorical library.

[^1]: Crampon, Jean E. 1988. Murphy, Parkinson, and Peter: Laws for librarians. Library Journal 113. no. 17 (October 15), p. 41.

## Content (LJW)

Every book reflects a combination of the authors' perspectives and the social and technological context in which the authors write. Thus, we see this book as a core component of the project of codifying *what a geographic data science does* and, in turn, what kinds of knowledge are important for aspiring geographic data scientists. We also see the *medium* and *method* of writing this book as important for its purpose. Hence, let's discuss first the content, then the method and medium.

### Overview of content

This book delves throughly into a few core topics. From our background as academic geographers, we seek to present concepts in a more *geographic* way than a standard textbook on *data science*.  This means that we cover spatial data, mapping, and spatial statistics *right* off the bat, and talk at length about some concepts (such as clusters or outliers) in a geographic, not data-scientific, manner. But, as we hope is shown throughout the book, the difference in language and framing is superficial, while the concepts are foundational to both perspectives.

With that in mind, we discuss the central data structures and representations in geographic data science, and then move immediately to visualization and analysis of geographic data. We use descriptive *spatial statistics* that summarize the structure of maps in order to build the intuition of how spatial thinking can be embedded in data science problems. For the analysis sections, we opt for a presentation of a classic subject in spatial analysis -inequality-, and then pivot to discussing important methods across geographic analysis, such as those that help understand when points are clustered in space, when geographic regions are latent within data, and when geographical spillovers are present in standard supervised learning approaches. The book closes with a discussion of how to use spatial principles to improve standard data scientific algorithms.

### What is not in the book (/"editorial discipline"?)

Despite the "breath over depth" approach we take in this book, there are many topics that we omit in our treatment. Every book must exhibit some kind of editorial discipline, and we use three principles to inform our own.

First, we avoided topics that get "too complicated too quickly"; instead, we sought to maximize the analytical benefits by focusing on simple but meaningful methods of analysis. This precludes many of the interesting but more complex topics and methods, like Bayesian inference or generative models (like cellular automata or agent-based models). GeoAI developments (GET CITES, Gao et al. editorial) at the cutting edge of quantitative geographic analysis also were excised under this editorial rule. Further, many treatments of the geographical problem of *scale* and *uncertainty* fall in this category, since these questions generally pose issues that demand theoretical, not empirical, solutions.

Second, we sought to discuss things that were intellectually adjacent to our own experiences and training in quantitative geography. The world of spatial statistics is vast, and very deep, but any one person only gains so much perspetive on it. This strongly informed our decisions of what to cover in the second and third sections, where we generally avoid more complex methods like Gaussian Process (geostatistical) models or geospatial knowledge graph methods. We would like to emphasise these avoidances are not commentary on their merits, which we recognise, but on our own ability to present the topics with honesty, clarity, and effectiveness.

Third, we sought to avoid topics that already have contemporary treatment in computational teaching. This includes spatial optimization problems (such as location allocation or coverage problems) as well as the generative and geostatistical models mentioned above. With this book, we are trying to cover areas where we see a clear opportunity in (re)framing them in new ways for the benefit of the two communities we mention above. Where there is already a wheel, we have not reinvented it.

Altogether, these three editorial principles help keep this book focused precisely on the set of techniques we think give readers the most benefit in the shortest space. It covers both methods to summarize and describe geographical pattern, correct analysis for the artefacts induced by geographical structure, and leverage geographical relationships to do data analysis better.

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

## Time in the book, and the book in time (/"the book with the perspective of the future")

In this section, we consider some of the main trends that have shaped the
conception of the book. As mentioned, every project like this is in part a
reflection of the time in which it is conceived and created. In our case,
this "era effect" has had both very
tangible ramifications, as well as other ones that, though perhaps less
visible at first, signal mayor shifts of the ground on which geographic data
science stands. Some of them are unequivocally positive, others more of a
price to pay to be able to develop a project like this one. 

Start with the obvious (but powerful): writing the book in the way we have done
so _is_ possible. This is a statement we would have not been able to make
a mere ten years ago. What you are holding in your hands (or displaying on your
laptop) is an academic textbook released under an open license, entirely based
on open technology, and using a platform that treats both narrative _and_ code
as first-class citizens. It is as much a book as a software artifact, and its
form embodies many of the principles that inspire its content.

Though possible, the process has not been straightforward. Many of the
technologies we rely on heavily were _just_ available when we started writing
back in 2018 (CONFIRM). Computational notebooks were stable by then, but ways
of combining them and using them as the building block of long-form writing
were not. In particular, this book owes much of its current form to two
projects,
`jupyterbook` and `jupytext`, which make it possible to build complex
documents starting from Jupyter notebooks and to mirror their content
to other formats such as markdown, respectively. Both projects were in their
early days when we adopted them and, using them in production at the same time
they were being developed into a stable shape has not been without its
challenges. But this has also reminded us the very best of the open-source
ethos: their teams have been a phenomenal example of how a
an open, fast-paced project can bring together a community around it. Although
many of the changes broke things constantly, clear documentation,
signposting, and responsiveness to our questions made it all possible.

In effect, not only infrastruture-wise, the wider landscape of Python for
geographic data science evolves very fast. Our scientific stack has changed
significantly over the period of writing. New packages appear, existing ones
change, and some also loose support and maintenance to a point that they are
unusable. Writing a book that tries to set up the main tool set in this
context is challenging. In some ways, by the time this book is in print, some
of its parts will be outdated or even obsolete. We think this is a problem,
albeit a small and good one. It is small because the core value proposition of
the book is not as a technical guide teaching a set of specific computational
tools. It is rather a companion to help you think geographically when you work
with modern data, and get the most of state-of-the-art data technologies when
you work with geographic problems. It is also a _good_ problem to have,
because it is sign that the ecosystem is constantly getting better. New
packages only become significant if they do more, better, or both than the
existing ones. At any rate, this constantly and rapidly changing context made
us think more thoroughly about the computational infrastructure and, over
time, we learned to take it more as a feature rather than a bug (it also
inspired us to write [Chapter 3](03_spatial_data)!). 

Besides technical challenges, writing a texbook based on notebooks has also
unearthed more conceptual aspects we had not ancitipated. Writing computational
notebooks is qualitatively different from writing a traditional textbook. The
writing process changes when you weave code and narrative, and that takes
additional effort and explicit design choices. Furthermore, since one of our
hard requirements for this project was the content be available online as a
free site, we were effectively writing for both print _and_ the web in the
same document. This
often meant tricky tradeoff's and, sometimes, settling for the (smaller)
common shared subset of options and functionality. All in all, this book has
taught us in very practical ways that the medium often frames the message,
and that we were exploring a less-known medium that had its own rules.

Finally, we believe the book was written at an inflection point where the
computational landscape for data science and GISc has left its previous
steady state, but it is not quite clear yet what the new one fully looks like.
Perhaps, as the famous William Gibson's quote goes, the "future is already
here - it's just not evenly distributed". Scientific computing is open by
default and, more and more, very interoperable. Tooling to work with such
stack, from low-level components to the end-user, has improved enormously and
continues to do so. At the same time, we think some of these changes bring
about more substantial shifts that we have not fully accommodated yet. As we
mention above, we have only scratched the surface of what new media like
computational notebooks allow, and much of the social infrastructure around
science (e.g., publishing) has been largely detached from these changes. With
this book, we hope to demonstrate what is already possible in this new world,
but also "nudge the way" for the uneven bits of the future that are still not
here. We hope you enjoy it and inspires you to further nudge away!

