# placebo_evaluation
The repository contains the project that evaluates various _ex ante_ and _ex post_ methods of counterfactual estimation for REDD+ projects (Reducing Emissions from Deforestation and forest Degradation in Developing countries), using a set of placebo projects randomly selected across the wet tropics with comparable characterics to existing REDD+ projects. Because of the absence of project activities, project and counterfactual deforestation should follow the identical trend in these placebo projects, allowing us to evaluate the predictive performance of a method by comparing predictions against observed deforestation rates in the placebo projects.

As input, the code ingests the output of the implementation code of the Canopy PACT 2.0 methodology for tropical forest carbon accreditation (https://github.com/quantifyearth/tmf-implementation), in the format of parquet files of sampled pixels in the project area and parquet files of matched pixels, which can be used to constrcut counterfactuals. It then estimates annual compound deforestation rates (%) of the placebo projects using one of the three _ex ante_ forecasting methods as well as an _ex post_ estimation method.

The project contains the following scripts, all written in Python:
1. rates.py: estimates annual counterfactual deforestation rates (%) using the four methods and comparing against observed deforestation rates in placebo projects
2. graphs.py: plots results from script 1
3. yearly_rates.py: calculates the predictive performance of the four methods for near-future time intervals of different durations
4. yearly_graphs.py: plots results from script 3
