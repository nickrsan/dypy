DyPy Core Concepts and Classes
==============================
DyPy's goal is to provide an interface to backward dynamic programming that supports the following priorities (in order):

1. Ease of learning and use
2. Flexible/adaptable to new problems
3. Speed (but only after 1 and 2 are satisfied)

In support of these goals, this document describes the core classes and how you might use them to build a dynamic program
with DyPy. While it outlines the core classes, and some of this document will be redundant to the API documentation, usage
options are included here to support reuse and extension of the package. Specific usage information will be included
in the `api` section. Note that many elements of the API that are otherwise not needed to use the package
are included as documented, public methods in order to aid subclassing and development of more complex models.

DynamicProgram
--------------
The Dynamic Program class is the core of DyPy. Each problem you wish to solve will involve creating an instance of this class
and attaching the classes below to it in ways that tell it how to solve your problem. One important design consideration
for DyPy is that it should be able to handle problems with multiple state variables

DynamicProgram manages all data and the flow of the optimization. By default, it will build all the stages and manage their
tables, but this part of the process can be customized as well (see `Stage`_ below for more).

API documentation for this class is here: :class:`dypy.DynamicProgram`

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
A stage provides the set of potential states and decisions at each modeled point sequential moment in the dynamic program.
The Stage class provides most of the heavy lifting and data management of this package and most classes ultimately
tie to either the `DynamicProgram`_ class or the Stage class. By default, the DynamicProgram class builds and manages
stages for you, but you may override this behavior for more complicated scenarios.

StateVariable
-------------
StateVariable objects provide options for potential future conditions. A `DynamicProgram`_ can involve
multiple state variables, in which case, all permutations of all state variable values are evaluated. Be careful because
the solution space can quickly grow in size as you add more state variables with more options. A StateVariable should have
a name and a set of potential values. By default, the potential values can be generated for you if you provide
a minimum value, a maximum value, and the discretization size of steps in between.

DecisionVariable
----------------
DecisionVariables describe potential choices that can be made at each stage. Similar to `StateVariable`_ objects,
they have a name and values, though they are managed slightly differently. Currently only a single decision variable is
supported in DyPy, though conceptually, it might be possible to implement multiple decision variables (with increasing
complexity to both code and solution time). Both DecisionVariables and StateVariables are provided to the
objective function to determine the value of each potential decision when the system is in a certain state.

Prior
-----
Priors in DyPy are used in two ways, each time referencing data from another stage that should be incorporated in the
current stage. This need arises during the backward matrix formulation as well as the forward calculation of the best
path. The :class:`dypy.Prior` class and subclasses provide different ways of applying future stage values to earlier stages.
:class:`dypy.SimplePrior` includes an implementation that will work for most single variable problems, but which may not work for
multi-variable problems. This class can be subclassed and have the apply method overridden to provide
a different implementation. The apply method should return the new matrix.

By default, the Prior class to be used should be provided to the `DynamicProgram`_ upon creation, but they
can also be overridden per-stage in case of a need to apply priors differently at different stages.

Reducer
-------
Reducers are still to be implemented, but provide a tool for turning multi-state variable problems into
single state variable problems before calculating the best path. In the case of a stochastic dynamic program,
one state might but based on your choices where other states are based on probabilistic future scenarios. Reducers
can help reduce the probabilistic states so that a single state variable reflecting the needs of the decision
can be used for the forward optimal path calculation.

Use of reducers is *not* required, and those with need for a true
stochastic dynamic program may wish to implement branching behavior reflecting the uncertainty in future stages.
The `Stage`_ and `Prior`_ classes would then need to be overriddent to provide such behavior in lieu of using reducers

