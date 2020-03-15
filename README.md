Welcome to Rexis\!
================

## The Machine Learning Technique Recommendation System

Rexis is an R package and Shiny application that serve as a holistic
approach to the algorithm selection problem for machine learning
classification problems. Rexis reads in a .csv data with binary targets
in the first column. Column headers are encouraged for ease of use.
Rexis then predicts the best algirthms from its built in taxonomy of
algorithms. Next, Rexis performs all recommended algorithms for the
problem and reports their perforance.

If you are searching for information on Rexis,Pennsyvania, please see
the following link [Rexis,
PA](https://en.wikipedia.org/wiki/Rexis,_Pennsylvania)

If you are searching for information on Regis Philbin, please see the
following link [Regis](https://en.wikipedia.org/wiki/Regis_Philbin)

To get started, install the Rexis package from the Rexis
project:

``` install
devtools::install_github("marcchale/Rexis", INSTALL_opts=c("--no-multiarch"))
```

Now you’re ready to use Rexis. Run the following code and browse to the
desired data set. The Heart data set is one example included in the
“data” folder. Type ResultsVar into your consolde to see the
performance of the analysis\!

``` use
ResultsVar <- rexecute()
```

If you prefer the Shiny interface, run the

``` use
run_my_app("RexisApp")
```

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
