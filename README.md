# DyPy: A general-purpose backward dynamic programming solver for Python

DyPy is currently in early development. as of 4/2/2019, the code here is not ready for production use unless you are
willing to invest time in ensuring your solution is correct and that the solver is working correctly. Please be cautioned
against using this solver. Your contributions to make this solver production-ready are welcome though, so please [get in touch](https://watershed.ucdavis.edu/user/64/contact)
if you are interested.

DyPy’s goal is to provide an interface to backward dynamic programming that supports the following priorities (in order):

1. Ease of learning and use
2. Flexible/adaptable to new problems
3. Speed (but only after 1 and 2 are satisfied - anecdotally, it's decently fast)

It achieves this approach with an extensive set of classes that are meant to be built on for new problems, where
as much as possible just works, but you can tweak things or provide options to handle your specific case. New contributions
that follow this philosophy are welcome!

Documentation for the in-development dypy can be found at https://nickrsan.github.io/dypy

If our disclaimer is big and scary (it should be) and you have a need for a library supporting dynamic programming, you 
might be interested in [ProDyn](https://prodyn.readthedocs.io/en/latest/index.html), which seems complete but is likely less flexible.