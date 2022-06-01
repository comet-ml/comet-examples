# Annoy integration with Comet.ml

Comet integrates with [Annoy](https://github.com/spotify/annoy).

Annoy ([Approximate Nearest Neighbors](http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor) Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are[mmapped] (https://en.wikipedia.org/wiki/Mmap) into memory so that many processes may  share the same data.

## Documentation

For more information on using and configuring Annoy integration, please see: https://www.comet.ml/docs/v2/integrations/third-party-tools/annoy/

## Setup

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the example

```bash
python annoy_example.py
```
