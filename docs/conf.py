# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../scanreader/'))

# -- Project information -----------------------------------------------------

project = 'scanreader'
copyright = '2024, Elizabeth R. Miller Brain Observatory (MBO) | The Rockefeller University. All Rights Reserved.'
author = 'Flynn OConnell'

# The full version, including alpha/beta/rc tags
release = 'v1.0.0'

myst_url_schemes = ("http", "https", "mailto")
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

autodoc2_packages = [
    "../scanreader",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.myst': 'myst-nb',
    '.ipynb': 'myst-nb',
}
# autodoc2_docstring_parser_regexes = [
#     # this will render all docstrings as Markdown
#     (r".*", "myst"),
#     # this will render select docstrings as Markdown
#     (r"mypackage\.mymodule\..*", "myst"),
# ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'exclude']
extensions = ["autodoc2",
              "sphinxcontrib.images",
              "sphinxcontrib.video" ,
              "myst_nb",
              "sphinx_copybutton",
              "numpydoc",
              "sphinx.ext.intersphinx",
              "sphinx.ext.viewcode",
              "sphinx.ext.napoleon",
              "sphinx.ext.autosectionlabel",
              "sphinx_togglebutton",
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '__pycache__/']
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
    'mbo': ('https://millerbrainobservatory.github.io/',None),
    'lbmpy': ('https://millerbrainobservatory.github.io/LBM-CaImAn-Python/', None),
    'lbmmat': ('https://millerbrainobservatory.github.io/LBM-CaImAn-MATLAB/', None)
}

html_logo = '_static/scanreader.svg'
html_short_title = "scanreader"
html_theme = 'sphinx_book_theme'
html_title = ""
html_css_files = [ 'custom.css' ]
html_favicon = "_static/mbo_icon_dark.ico"
html_static_path = ['_static']

# html_context = {
#     "github_user": "https://github.com/MillerBrainObservatory/",
#     "github_repo": "https://github.com/MillerBrainObservatory/scanreader/",
#     "doc_path": "docs"
# }


# sphinxcontrib.images config
images_config = dict(backend='LightBox2',
                     default_image_width='100%',
                     default_show_title='True',
                     default_group='default')


sphinx_book_options = {
    "external_links": [
        {"name": "MBO.io", "url": "https://millerbrainobservatory.github.io/index.html"},
        {"name": "LBM.Mat", "url": "https://millerbrainobservatory.github.io/LBM-CaImAn-MATLAB/index.html"},
        {"name": "LBM.Py", "url": "https://millerbrainobservatory.github.io/LBM-CaImAn-Python/index.html"},
    ],
}

html_theme_options = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/executablebooks/MyST-Parser",
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    # "announcement": "<b>v3.0.0</b> is now out! See the Changelog for details",
}

html_theme_options = sphinx_book_options
