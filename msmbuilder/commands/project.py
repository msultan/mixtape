"""Set up a new MSMBuilder project

"""
# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2016, Stanford University
# All rights reserved.

from __future__ import print_function, division, absolute_import

import os
import re
import stat
import textwrap
from collections import defaultdict
from datetime import datetime

import nbformat
from jinja2 import Environment, PackageLoader
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from ..cmdline import Command, argument
from ..dataset2 import backup


def chmod_plus_x(fn):
    st = os.stat(fn)
    os.chmod(fn, st.st_mode | stat.S_IEXEC)


class SetUpProject(Command):
    _group = '0-Support'
    _concrete = True
    description = __doc__

    ipynb = argument('--ipynb', default=False, action='store_true',
                     help="Write IPython / Jupyter notebooks instead of "
                          "Python scripts")

    flavor = argument('--flavor', choices=['generic', 'fah'],
                      default='generic',
                      help="Which flavor of scripts to write.")

    steps = argument('--steps', nargs='+', type=int, default=[],
                     help="Only make files for these steps in the process.")

    def __init__(self, args):
        # Functions by extension
        self.write_funcs = defaultdict(lambda: self.write_generic)
        self.write_funcs.update({
            'py': self.write_python,
            'sh': self.write_shell,
        })

        if args.ipynb:
            # TODO: modify plotting boilerplate (matplotlib inline, xdg-open)
            self.write_funcs['py'] = self.write_ipython

        self.flavor = args.flavor
        self.steps = args.steps

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
        nb = new_notebook(cells=cells,
                          metadata={'kernelspec': {'name': 'python3',
                                                   'display_name': 'Python 3'}})
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
            if len(self.steps) > 0:
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

    def start(self):
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
            out_fn = "{k}.{ext}".format(k=k, ext=ext)
            self.write_funcs[ext](out_fn, rendered)

        print('\n'.join(textwrap.wrap(
            "Ok, I wrote out a bunch of Python files that can guide you "
            "through analyzing a system with MSMBuilder. I implore you to "
            "look at the scripts before you start blindly running them. "
            "You will likely have to change some (hyper-)parameters or "
            "filenames to match your particular project."
        )))
        print()
        print('\n'.join(textwrap.wrap(
            "More than that, however, it is important that you understand "
            "exactly what the scripts are doing. Each protein system is "
            "different, and it is up to you (the researcher) to hone in on "
            "interesting aspects. This very generic pipeline may not give "
            "you any new insight for anything but the simplest systems."
        )))
