# master-thesis-scripts
Contains the scripts that are used in my master thesis about a computational model to study literary characters.

This repository hosts three scripts (plus a small common library) that are used in my master thesis, which aims to outline, both theoretically and technically, a computational method to obtain a model of the characters' traits and narratological role within a novel.
This is a brief overview on what each program does:
 - **traits.py**: this script is meant to operationalize concepts by the literary theorists Roland Barthes and Seymour Chatman, who, despite some theoretical differences, viewed the literary character as a **paradigm of traits** (either semantic or psychological), of which the text provides hints in the from of **distributed lexicon consistently related to a given character**. It is conceptually simple: it breaks down the text into sentences and paragraphs, storing those where the proper name of the selected literary character is found; then, by using TF-IDF, it finds those **words** that are **most distinctive of the vectorized sentences and paragraphs where the character is found** with respect to the complete text, and outputs them as **keywords** of the character. It is possible to perform the same analysis on segments of the text as well.
 - **intSentAnalyzer.py**: this script is meant to operationalize the narratological model provided by Vladimir Propp and Claude Bremond, who basically claimed that the **narratological role** of the character is the main drive of the development of the plot. By visualizing the **distribution of the selected characters** of the novel, and the segments of the text where they interact with **direct speech**, relating it to the **emotional arc of the story** obtained through sentiment analysis, it aims to **visualize how the characters affect the emotional trend of the plot**, and which interactions function as relevant plot junctures. To visualize interactions, it requires manual tagging as explained below.
 - **classchar.py**: this library provides some common functions and lists of stopwords.

**REQUIREMENTS:**
 - **Python 3.x**
 - Tested in **Windows 10** and **Linux** (Ubuntu, Fedora). Not guaranteed to work on other platforms.
 - Python libraries (built-in ones excluded): **sklearn**, **nltk** (with the packages "stopwords" and "punkt"), **vadersentiment**, **pandas**, **matplotlib**, **readline** (optional, Linux users only)
 - Novels in **.txt format** and proper encoding of escape characters: make sure **\n** is found **at the end of each paragraph** and not at the end of each line (the programs will work anyway but the results, especially the output of traits.py, will be inaccurate).

**INSTALLATION GUIDE:**
 - Run the usual "pip install library-name" in command-line to install the required libraries;
 - To install the required nltk packages, run in the Python IDE the command "nltk.download("package-name")" for both of them. An error output by nltk should tell you how to do that anyway.
 - Download the scripts and place them all in the same directory. The novels to be analyzed must be placed in the same directory of the scripts as well.
 - Run them as normal Python scripts, they will guide you at each step on what to insert and how.
