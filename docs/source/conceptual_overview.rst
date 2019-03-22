DyPy Core Concepts and Classes
==============================
DyPy's goal is to provide an interface to backward dynamic programming that supports the following priorities (in order):

1. Ease of learning and use
2. Flexible/adaptable to new problems
3. Speed (but only after 1 and 2 are satisfied)

In support of these goals, this document describes the core classes and how you might use them to build a dynamic program
with DyPy. While it outlines the core classes, and some of this document will be redundant to the API documentation, usage
options are included here to support reuse and extension of the package.

DynamicProgram
--------------
The Dynamic Program class is the core of DyPy. Each problem you wish to solve will involve creating an instance of this class
and attaching the classes below to it in ways that tell it how to solve your problem. One important design consideration
for DyPy is that it should be able to handle problems with multiple state variables

DynamicProgram manages all data and the flow of the optimization. By default, it will build all the stages and manage their
tables, but this part of the process can be customized as well (see `Stage`_ below for more)

Objective Functions
+++++++++++++++++++
The objective function will do some of the heavy lifting for your dynamic program, and must be created by the user for each
specific optimization problem. DyPy will call the objective function for every combination of state variables and stage
variables in each stage of the optimization and the objective function will need to return the cost or benefit value for
that set of inputs. The objective function will be provided access to the `Stage`_ object for the stage it is currently
evaluating, as well as the *values* of all of the state variables and the decision variable. These will be provided as
keyword arguments to the objective function. See the :ref:`examples` section for more information.

Stage
-----

StateVariable
-------------

DecisionVariable
----------------

Prior
-----

Reducer
-------

