# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2016, Stanford University
# All rights reserved.

from __future__ import print_function, division, absolute_import

import os
import pickle
import re
import shutil
import stat
import warnings
from collections import defaultdict
from datetime import datetime

import mdtraj as md
import nbformat
import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

__all__ = ['backup', 'preload_top', 'preload_tops', 'load_meta', 'load_generic',
           'load_trajs', 'save_meta', 'save_generic', 'save_trajs']


def backup(fn):
    if not os.path.exists(fn):
        return

    backnum = 1
    backfmt = "{fn}.bak.{backnum}"
    trial_fn = backfmt.format(fn=fn, backnum=backnum)
    while os.path.exists(trial_fn):
        backnum += 1
        trial_fn = backfmt.format(fn=fn, backnum=backnum)

    warnings.warn("{fn} exists. Moving it to {newfn}"
                  .format(fn=fn, newfn=trial_fn))
    shutil.move(fn, trial_fn)


def chmod_plus_x(fn):
    st = os.stat(fn)
    os.chmod(fn, st.st_mode | stat.S_IEXEC)


def get_fn(base_fn, key):
    dfmt = "{}"
    ffmt = "{}.npy"
    if isinstance(key, tuple):
        paths = [dfmt.format(k) for k in key[:-1]]
        paths += [ffmt.format(key[-1])]
        return os.path.join(base_fn, *paths)
    return os.path.join(base_fn, ffmt.format(key))


def preload_tops(meta):
    """Load all topology files into memory.

    This might save some performance compared to re-parsing the topology
    file for each trajectory you try to load in. Typically, you have far
    fewer (possibly 1) topologies than trajectories

    Parameters
    ----------
    meta : pd.DataFrame
        The DataFrame of metadata with a column named 'top_fn'

    Returns
    -------
    tops : dict
        Dictionary of ``md.Topology`` objects, keyed by "top_fn"
        values.
    """
    top_fns = set(meta['top_fn'])
    tops = {}
    for tfn in top_fns:
        tops[tfn] = md.load(tfn)
    return tops


def preload_top(meta):
    """Load one topology file into memory.

    This function checks to make sure there's only one topology file
    in play. When sampling frames, you have to have all the same
    topology to concatenate.

    Parameters
    ----------
    meta : pd.DataFrame
        The DataFrame of metadata with a column named 'top_fn'

    Returns
    -------
    top : md.Topology
        The one topology file that can be used for all trajectories.
    """
    top_fns = set(meta['top_fn'])
    if len(top_fns) != 1:
        raise ValueError("More than one topology is used in this project!")
    return md.load(top_fns.pop())


def load_meta(meta_fn='meta.pandas.pickl'):
    """Load metadata associated with a project.

    Parameters
    ----------
    meta_fn : str
        The filename

    Returns
    -------
    meta : pd.DataFrame
        Pandas DataFrame where each row contains metadata for a
        trajectory.
    """
    return pd.read_pickle(meta_fn)


def save_meta(meta, meta_fn='meta.pandas.pickl'):
    """Save metadata associated with a project.

    Parameters
    ----------
    meta : pd.DataFrame
        The DataFrame of metadata
    meta_fn : str
        The filename
    """
    backup(meta_fn)
    pd.to_pickle(meta, meta_fn)


def save_generic(obj, fn):
    """Save Python objects, including msmbuilder Estimators.

    This is a convenience wrapper around Python's ``pickle``
    serialization scheme. This protocol is backwards-compatible
    among Python versions, but may not be "forwards-compatible".
    A file saved with Python 3 won't be able to be opened under Python 2.
    Please read the pickle docs (specifically related to the ``protocol``
    parameter) to specify broader compatibility.

    If a file already exists at the given filename, it will be backed
    up.

    Parameters
    ----------
    obj : object
        A Python object to serialize (save to disk)
    fn : str
        Filename to save the object. We recommend using the '.pickl'
        extension, but don't do anything to enforce that convention.
    """
    backup(fn)
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)


def load_generic(fn):
    """Load Python objects, including msmbuilder Estimators.

    This is a convenience wrapper around Python's ``pickle``
    serialization scheme.

    Parameters
    ----------
    fn : str
        Load this file

    Returns
    -------
    object : object
        The object.
    """
    with open(fn, 'rb') as f:
        return pickle.load(f)


def save_trajs(trajs, fn, meta):
    """Save trajectory-like data

    Data is stored in individual numpy binary files in the
    directory given by ``fn``.

    This method will automatically back up existing files named ``fn``.

    Parameters
    ----------
    trajs : dict of (key, np.ndarray)
        Dictionary of trajectory-like ndarray's keyed on ``meta.index``
        values.
    fn : str
        Where to save the data. This will be a directory containing
        one file per trajectory
    meta : pd.DataFrame
        The DataFrame of metadata
    """
    backup(fn)
    os.mkdir(fn)
    for k in meta.index:
        v = trajs[k]
        np.save(get_fn(fn, k), v)


def load_trajs(fn, meta='meta.pandas.pickl'):
    """Load trajectory-like data

    Data is expected to be stored as if saved by ``save_trajs``.

    This method finds trajectories based on the ``meta`` dataframe.
    If you remove a file (trajectory) from disk, be sure to remove
    its row from the dataframe. If you remove a row from the dataframe,
    be aware that that trajectory (file) will not be loaded, even if
    it exists on disk.

    Parameters
    ----------
    fn : str
        Where the data is saved. This should be a directory containing
        one file per trajectory.
    meta : pd.DataFrame or str
        The DataFrame of metadata. If this is a string, it is interpreted
        as a filename and the dataframe is loaded from disk.

    Returns
    -------
    meta : pd.DataFrame
        The DataFrame of metadata. If you passed in a string (filename)
        to the ``meta`` input, this will be the loaded DataFrame. If
        you gave a DataFrame object, this will just be a reference back
        to that object
    trajs : dict
        Dictionary of trajectory-like np.ndarray's keyed on the values
        of ``meta.index``.
    """
    if isinstance(meta, str):
        meta = load_meta(meta_fn=meta)
    trajs = {}
    for k in meta.index:
        trajs[k] = np.load(get_fn(fn, k))
    return meta, trajs


class ProjectTemplate(object):
    """Construct a set of scripts to serve as a template for a new project.

    Parameters
    ----------
    flavor : str
        The type of project to set up. Current options are {'generic',
        'fah'}.
    steps : list of int
        Which steps to write out. If None, write all steps (default)
    ipynb : bool
        Write IPython Notebooks where applicable.

    """

    flavors = {'generic', 'fah'}

    def __init__(self, flavor='generic', steps=None, ipynb=False):
        self.write_funcs = defaultdict(lambda: self.write_generic)
        self.write_funcs.update({
            'py': self.write_python,
            'sh': self.write_shell,
        })

        if ipynb:
            self.write_funcs['py'] = self.write_ipython

        if flavor not in self.flavors:
            raise ValueError("Unknown flavor {}. Please choose one of {}"
                             .format(flavor, self.flavors))

        self.flavor = flavor
        self.steps = steps

    def get_header(self):
        return '\n'.join([
            "msmbuilder autogenerated template version 1",
            'created {}'.format(datetime.now().isoformat()),
            "please cite msmbuilder in any publications"
        ])

    def write_ipython(self, templ_fn, rendered):
        templ_ipynb_fn = templ_fn.replace('.py', '.ipynb')

        cell_texts = [templ_ipynb_fn] + re.split(r'## (.*)\n', rendered)
        cells = []
        for heading, content in zip(cell_texts[:-1:2], cell_texts[1::2]):
            cells += [new_markdown_cell("## " + heading.strip()),
                      new_code_cell(content.strip())]
        nb = new_notebook(
            cells=cells,
            metadata={'kernelspec': {
                'name': 'python3',
                'display_name': 'Python 3'
            }})
        backup(templ_ipynb_fn)
        with open(templ_ipynb_fn, 'w') as f:
            nbformat.write(nb, f)

    def write_python(self, templ_fn, rendered):
        backup(templ_fn)
        with open(templ_fn, 'w') as f:
            f.write(rendered)

    def write_shell(self, templ_fn, rendered):
        backup(templ_fn)
        with open(templ_fn, 'w') as f:
            f.write(rendered)
        chmod_plus_x(templ_fn)

    def write_generic(self, templ_fn, rendered):
        backup(templ_fn)
        with open(templ_fn, 'w') as f:
            f.write(rendered)

    def get_templates(self, fns):
        keys = set((fn.split('.')[0], fn.split('.')[-1]) for fn in fns)
        templates = {}
        for k, ext in keys:
            if self.steps is not None:
                try:
                    step = int(k.split('-')[0])
                except ValueError:
                    continue
                if step not in self.steps:
                    continue

            trial_fn = ('{k}.{flavor}.{ext}'
                        .format(k=k, flavor=self.flavor, ext=ext))
            if trial_fn in fns:
                templates[k, ext] = trial_fn
            else:
                trial_fn = "{k}.{ext}".format(k=k, ext=ext)
                if trial_fn in fns:
                    templates[k, ext] = trial_fn
                else:
                    continue
        return templates

    def render_iter(self):
        env = Environment(loader=PackageLoader('msmbuilder',
                                               'project_templates'))
        templ_dict = self.get_templates(env.list_templates())
        for (k, ext), templ_fn in templ_dict.items():
            template = env.get_template(templ_fn)
            rendered = template.render(
                header=self.get_header(),
                topology_fn='data/fs-peptide.pdb',
                timestep=10,
                date=datetime.now().isoformat(),
            )
            yield k, ext, rendered

    def render_all(self):
        return {"{k}.{ext}".format(k=k, ext=ext): rendered
                for k, ext, rendered in self.render_iter()}

    def write_all(self):
        for k, ext, rendered in self.render_iter():
            out_fn = "{k}.{ext}".format(k=k, ext=ext)
            self.write_funcs[ext](out_fn, rendered)
