Welcome to Rexis\!
================

## The Machine Learning Technique Recommendation System

Rexis is an R package and Shiny application that serve as a holistic
approach to the algorithm selection problem for machine learning
classification problems. Rexis reads in a .csv data file with binary
targets in the first column. Column headers are encouraged for ease of
use. Rexis then predicts the best algorithms from its built in taxonomy
of algorithms. Next, Rexis performs all recommended algorithms for the
problem and reports their perforance. According to research performed at
the Air Force Institute of Technology, Rexis predicts the best algorithm
in 78% of problems.

### Getting Started

To get started, install the Rexis package from the Rexis
project:

``` install
devtools::install_github("marcchale/Rexis", INSTALL_opts=c("--no-multiarch"))
library(Rexis)
```

### Using Rexis

Now you’re ready to use Rexis. Run the following code and browse to the
desired data set. The “Heart”heart.csv" data set is one example included
in the “Rexis/data/” folder. Type ResultsVar into your consolde to see
the performance of the analysis\!

``` use
ResultsVar <- rexecute()
ResultsVar
```

If you prefer the Shiny interface, run the code below in the R console
and select a .csv \!\!

``` use
run_my_app("RexisApp")
```

<img src="inst/images/HeartScreenshot.PNG" alt="Screenshot Example">

### Help with Rexis

You can access additional help documentation for the rexicute and the
rexicuteshiny functions in your R IDE.

    ? rexicute
    ? rexicuteshiny

### Additional Resources

If you would like to read the AFIT Thesis that used the Rexis software,
please email <marc.chale@afit.edu>

If you are searching for information on Rexis, Pennsyvania, please see
the following link [Rexis,
PA](https://en.wikipedia.org/wiki/Rexis,_Pennsylvania)

If you are searching for information on Regis Philbin, please see the
following link [Regis](https://en.wikipedia.org/wiki/Regis_Philbin)
