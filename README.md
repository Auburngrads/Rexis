Welcome to Rexis\!
================

## Welcome to Rexis, the Machine Learning Technique Recommendation System\!

To get started, install the Rexis package from the Rexis project:

``` install
install.packages("Rexis")
```

Next, open Rexis

``` open
library(Rexis)
```

Now you’re ready to use Rexis. Run the following code and browse to the
desired data set. The Heart data set is included in the package folder
“data”. Type ResultsVar into your consolde to see the performance of
the analysis\!va

``` use
ResultsVar <- Rexis:::execute_py()
```

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
