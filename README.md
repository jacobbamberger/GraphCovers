# GraphCovers dataset
Code accompanying the paper ["A Topological characterisation of Weisfeiler-Leman equivalence classes"](https://arxiv.org/abs/2206.11876) which will be presented at ICML's workshop Topology, Algebra and Geometry in ML on July 23rd 2022.

In the paper, we give a topological description of all graphs than are indisinguishable using the WL test, and therefore by most message passing Graph Neural Network. We use this description to generate many pairwise non-isomorphic graphs which are indistinguishable by the WL test. Here are a few examples: ![This is an image](https://github.com/jacobbamberger/GraphCovers/blob/main/covers.png)


## Code
In the code we focus on the generationg process, and give an efficient algorithm for testing if two generated graphs are isomorphic. Both are described in Section 3 of the paper. The codebase can be found in [covers.py](https://github.com/jacobbamberger/GraphCovers/blob/main/covers.py), but to get started we recommend checking out the [notebook](https://github.com/jacobbamberger/GraphCovers/blob/main/covers_notebook.ipynb). In the notebook we first generate some graphs, then visualize them, and finally try classifying them with several GNN architecures. Everything can be run on a CPU.

## Getting started
You only need numpy and networkx to generate the dataset, plotly to vizualise the covers, and pytorch and pytorch-geometric to train the models on the jupyter notebook. If you don't have this in your environment, one option is to do the following:

1. Start by installing the requirements:
```
python3 -m venv env
pip install -r requirements.txt
```
2. Run the [notebook](https://github.com/jacobbamberger/GraphCovers/blob/main/covers_notebook.ipynb)
3. Checkout the code in [covers.py](https://github.com/jacobbamberger/GraphCovers/blob/main/covers.py)

Hope you find it interesting!
Please let me know if you have any comments/questions.
