# placebo_evaluation
The repository contains the project that evaluates various _ex ante_ and _ex post_ methods of counterfactual estimation for REDD+ projects (Reducing Emissions from Deforestation and forest Degradation in Developing countries), using a set of placebo projects randomly selected across the wet tropics with comparable characterics to existing REDD+ projects. Because of the absence of project activities, project and counterfactual deforestation should follow the identical trend in these placebo projects, allowing us to evaluate the predictive performance of a method by comparing predictions against observed deforestation rates in the placebo projects.

As input, the code ingests the parquet files of sampled pixels in the project area and parquet files of matched pixels. The matching procedures were conducted using the Tropical Moist Forest Accreditation Methodology Implementation code[[1]](#1), which implements the Canopy PACT 2.0 methodology[[2]](#1)[[3]](#2). The matched pixels are used to constrcut counterfactuals using one of the three _ex ante_ forecasting methods as well as an _ex post_ estimation method. The code then estimates annual compound deforestation rates (%) of the placebo projects, and compares it against observed deforestation rates.

The project contains the following scripts, all written in Python:
1. rates.py: estimates annual counterfactual deforestation rates (%) using the four methods and comparing against observed deforestation rates in placebo projects
2. graphs.py: plots results from script 1
3. yearly_rates.py: calculates the predictive performance of the four methods for near-future time intervals of different durations
4. yearly_graphs.py: plots results from script 3

## References
<a id="1">[1]</a> 
Dales M, Ferris P, Message R, Holland J, Williams A (2023).
GitHub Repository: Tropical Moist Forest Accreditation Methodology Implementation
https://github.com/quantifyearth/tmf-implementation. commit:7f15246

<a id="2">[2]</a> 
Balmford, A _et al._ (2023). 
PACT Tropical Moist Forest Accreditation Methodology v2.0
Communications of the ACM, 11(3), 147-148.
```bibtex
@ARTICLE{Balmford2023-zd,
  title    = "{PACT} Tropical Moist Forest Accreditation Methodology",
  author   = "Balmford, Andrew and Coomes, David and Hartup, James and Jaffer,
              Sadiq and Keshav, Srinivasan and Lam, Miranda and Madhavapeddy,
              Anil and Rau, E-Ping and Swinfield, Thomas and Wheeler, Charlotte",
  journal  = "Cambridge Open Engage",
  abstract = "This draft document describes the methodology developed by the
              Cambridge Center for Carbon Credits (4C) for estimating the number
              of credits to be issued to a project in the tropical moist forest
              (TMF) biome. It expands on the methodology outlined in (Swinfield
              and Balmford, 2023). We welcome comments and suggestions in the
              associated online document at https://tinyurl.com/cowgreport.",
  month    =  jun,
  year     =  2023,
  keywords = "Carbon credits;Methodology;Cambridge Center for Carbon Credits",
  language = "en"
}
```

<a id="3">[3]</a> 
```bibtex
@ARTICLE{Swinfield2023-pc,
  title    = "Cambridge Carbon Impact: Evaluating carbon credit claims and
              co-benefits",
  author   = "Swinfield, Tom and Balmford, Andrew",
  journal  = "Cambridge Open Engage",
  abstract = "While decarbonising economies is a global imperative, offsetting
              residual emissions remains an essential interim step in avoiding
              climate breakdown. Carbon credits can be purchased on the
              voluntary market, but genuine concerns exist as to whether
              credits, in their current state, will enable the transition to
              net-zero by 2050. The Cambridge Carbon Impact project seeks to
              rebuild trust through an independent assessment of carbon credits,
              using the best methods and data available. By evaluating carbon
              credits across all dimensions of additionality and risk, and
              reporting performance through inter-comparable metrics, we aim to
              clarify concerns, address them where possible and flag them where
              not, and so allow prospective buyers to make offsets that have the
              kind of impacts they want. We report the number of credits needed
              to have the same climate impact as one tonne of CO2 sequestered
              into geological storage, which we dub one Permanent Additional
              Carbon Tonne (PACT).",
  month    =  mar,
  year     =  2023,
  keywords = "Carbon credits;Natural climate solutions;Tropical
              forest;Peatland;Woodland;Restoration;Conservation;Direct air
              capture and storage",
  language = "en"
}
```