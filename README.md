# K-means++ Clustering — Python & C

Implementation of the **K-means++ initialization algorithm** in Python, integrated with a **C extension module** that performs the iterative K-means clustering computation.

The project combines Python's high-level data handling with a lower-level C implementation of the computational core, providing hands-on experience with numerical algorithms, memory management, and Python–C interoperability.

## Features

* Implements **K-means++ centroid initialization** in Python.
* Implements the iterative **K-means clustering algorithm** in C.
* Exposes the C implementation to Python through a custom extension module.
* Separates initialization, clustering logic, and Python/C integration into modular components.
* Supports local testing and validation of the implementation.

## Project Structure

```text
kmeans-python-c/
├── kmeans_pp.py        # K-means++ initialization and Python interface
├── kmeansmodule.c      # C extension implementing the K-means algorithm
├── setup.py            # Build configuration for the C extension
├── tests/              # Local tests
├── .gitignore
└── README.md
```

## Architecture

The project is divided into two main layers:

**Python layer — `kmeans_pp.py`**

Responsible for data handling, K-means++ centroid initialization, and interaction with the C extension.

**C layer — `kmeansmodule.c`**

Implements the computational part of the K-means algorithm and exposes it as a module that can be called directly from Python.

This structure keeps the high-level workflow in Python while moving the core numerical computation to C.

## Technologies

* Python
* C
* Python C API
* NumPy
* Git

## What I Practiced

This project provided practical experience with:

* Implementing numerical and clustering algorithms.
* Integrating C code with Python.
* Managing data transfer between Python and C.
* Working with memory and data structures in C.
* Structuring a project across multiple programming languages.
* Debugging and testing a native Python extension.

## Background

Developed as part of a Software Project course at Tel Aviv University.
