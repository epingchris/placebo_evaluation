# Evaluation of methods of counterfactual estimation for REDD+ projects via a placebo approach
 
The repository contains the project that evaluates various _ex ante_ and _ex post_ methods of counterfactual estimation for REDD+ projects (Reducing Emissions from Deforestation and forest Degradation in Developing countries), using a set of placebo projects randomly selected across the wet tropics with comparable characterics to existing REDD+ projects. Because of the absence of project activities, project and counterfactual deforestation should follow the identical trend in these placebo projects, allowing us to evaluate the predictive performance of a method by comparing predictions against observed deforestation rates in the placebo projects.

## Input

The code ingests the parquet files of sampled pixels in the project area and parquet files of matched pixels. The matching procedures were conducted using the Tropical Moist Forest Accreditation Methodology Implementation code[[1]](#1), which implements the Canopy PACT 2.0 methodology[[2]](#1)[[3]](#2). The matched pixels are used to constrcut counterfactuals using one of the three _ex ante_ forecasting methods as well as an _ex post_ estimation method. The code then estimates annual compound deforestation rates (%) of the placebo projects, and compares it against observed deforestation rates.

## Structure

The project contains the following scripts, all written in Python:
1. rates.py: estimates annual counterfactual deforestation rates (%) using the four methods and comparing against observed deforestation rates in placebo projects
2. graphs.py: plots results from script 1
3. yearly_rates.py: calculates the predictive performance of the four methods for near-future time intervals of different durations
4. yearly_graphs.py: plots results from script 3

## System requirements

This project is developed under Python 3.12.2, and uses the libraries _panda_, _numpy_, _scipy_, and _matplotlib_.

On the Sherwood cluster (Dept of Comp Sci, University of Cambridge), the easiest way to run a script is by:

```
tmfpython3 -m [script name]
```

## References
<a id="1">[1]</a> 
Dales M, Ferris P, Message R, Holland J, and Williams A (2023).
GitHub Repository: Tropical Moist Forest Accreditation Methodology Implementation, 
https://github.com/quantifyearth/tmf-implementation. commit:7f15246

<a id="2">[2]</a> 
Balmford, A _et al._ (2023). 
PACT Tropical Moist Forest Accreditation Methodology v2.0. _Cambridge Open Engage_, 
https://www.cambridge.org/engage/coe/article-details/657c8b819138d23161bb055f.

<a id="3">[3]</a> 
Swinfield, T and Balmford, A (2023). 
Cambridge Carbon Impact: Evaluating carbon credit claims and co-benefits. _Cambridge Open Engage_, 
https://www.cambridge.org/engage/coe/article-details/6409c345cc600523a3e778ae.