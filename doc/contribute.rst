====================
Become a contributor
====================

How to contribute to celer?
---------------------------

``celer`` is an open source project and hence rely on community efforts to evolve.
No matter how small your contribution is, we highly valuate it. Your contribution
can come in three forms

- **Bug report**

``celer`` runs continuously unit test on the code base to prevent bugs. Help us tighten these tests by reporting 
any bug that you encountered while using ``celer``. To do so, use the
`issue section <https://github.com/mathurinm/celer/issues>`_
available on the ``celer`` repository.

- **Feature request**

We are constantly improving ``celer`` and we would like to align that with our user needs.
Hence, we highly appreciate any suggestion to extend or add new features to ``celer``.
You can use the `issue section <https://github.com/mathurinm/celer/issues>`_ to make suggestions.


- **Pull request**

If you fixed a bug, added new features, or even corrected a small typo in the documentation.
You can submit a `pull request <https://github.com/mathurinm/celer/pulls>`_ to integrate your changes
and we will reach out to you as soon as possible.



Setup ``celer`` on your local machine
---------------------------------------

Here are key steps to help you setup ``celer`` on your local machine in case you wanted to
contribute with code or documentation to celer.

1. Fork the repository and afterwards run the following command to clone it on your local machine

.. code-block:: shell

    $ git clone https://github.com/{YOUR_GITHUB_USERNAME}/celer.git


2. ``cd`` to ``celer`` directory and install it in edit mode by running

.. code-block:: shell

    $ cd celer
    $ pip install -e .


3. To run the gallery examples and build the documentation, run the followings

.. code-block:: shell

    $ cd doc
    $ pip install -r doc-requirements.txt
    $ make html


.. note::
    You should have a `gcc compiler <https://gcc.gnu.org/>`_ 
    installed in your local machine since ``celer`` uses Cython.