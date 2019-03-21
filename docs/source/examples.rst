DyPy Examples
=============

Programming Effort
------------------
You're a programmer working hard to triage bugs in your software
before a big release coming up. In your bug tracker, bugs are split
into 5 categories: "Core", "UX/UI", "Network", "Database", and "API".
The release is in 12 days, but your team manager wants to make sure
to get some fixes in each, so they ask you to spend no more
than 5 days on any single area and at least one day on each, but other than that, they just want you
to fix the most bugs possible. Often, you get some benefit from continued work
on one area, so you fix more bugs the longer the time you spend on one task,
but you also know some areas a bit better than others and burn out a bit if
you spend too much time in any one area. Based on that, and the prioritization
of tickets in the tracker, you estimate the number
of bugs you fix in each category as a function of time as follows:

.. list-table:: Bugs closed by days of effort in each area
   :widths: 40, 12, 12, 12, 12, 12
   :header-rows: 1

   * - Category
     - 1 Day
     - 2 Days
     - 3 Days
     - 4 Days
     - 5 Days
   * - Core
     - 2
     - 5
     - 7
     - 8
     - 10
   * - UI/UX
     - 3
     - 6
     - 8
     - 9
     - 9
   * - Network
     - 1
     - 2
     - 4
     - 7
     - 9
   * - Database
     - 2
     - 3
     - 6
     - 8
     - 11
   * - API
     - 4
     - 6
     - 8
     - 9
     - 10

Build a dynamic program that recommends the appropriate days of effort for each category.

Solution
++++++++
Using DyPy, we'll set up each category of work as a stage since we'll do them sequentially,
and then build an objective function that provides the number of bugs closed for a given
number of days of effort.

.. code-block:: python

    # we have 12 days available to us for studying
    state_variable = dypy.StateVariable("Days Spent On Category", values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # but we can only spend between 1 and 4 days studying for any single course
    decision_variable = dypy.DecisionVariable("Time on Category", options=[1, 2, 3, 4, 5])

    def objective_function(stage, days_spent_on_category, time_on_category):
        """
            When the objective is called, the solver makes the stage, state variables,
            and decision variable available to it. The keyword argument names here
            match the names of the state variables and decision variables (which will be
            passed as keyword arguments, so order isn't important, but name is). The name
            is automatically derived by lowercasing the name of the variable and replacing
            spaces with underscores. It will remove digits from the front if they are present
            so that the name is a valid python identifier. You can also manually provide a
            kwarg name - if you wish to do this, see the API documentation.

            In this example, we can think of these variables as providing:
                1. stage - this will give access to the DyPy.stage object
                2. state - this just provides access to the state value being assessed, not the object
                3. decision - this just provides access to the decision value being assessed, not the object
        """

        # we are given benefits, so defining here - each category is a row, each column is
        # number of days, and the value is how much benefit we get from working on that category
        # for that long
        benefit_list = [
            # column 0 means no time spent on it, so we get no benefit
            [0, 2, 5, 7, 8, 10],
            [0, 3, 6, 8, 9, 9],
            [0, 1, 2, 4, 7, 9],
            [0, 2, 3, 6, 8, 11],
            [0, 4, 6, 8, 9, 10],
        ]

        if time_on_category > days_spent_on_category:
            return stage.parent_dp.exclusion_value  #
        else:
            return benefit_list[stage.number][time_on_category]

    # define the dynamic program and tell it we have four timesteps
    dynamic_program = dypy.DynamicProgram(timestep_size=1, time_horizon=5,
                                          objective_function=objective_function, calculation_function=dypy.MAXIMIZE,
                                          prior=dypy.SimplePrior)
    dynamic_program.exclusion_value = -1  # set it to -1 since we know nothing else will be negative here - lets us visualize arrays better
    dynamic_program.decision_variable = decision_variable
    dynamic_program.add_state_variable(state_variable)

    # each category will be a stage, in effect - tell it to create all five stages as empty
    dynamic_program.build_stages(name_prefix="Category")  # assigns names by default, but we can override them
    stage_names = ["Core", "UI/UX", "Network", "Database", "API"]
    for i, stage in enumerate(dynamic_program.stages):
        dynamic_program.stages[i].name = stage_names[i]  # these need to match the order

    dynamic_program.run()