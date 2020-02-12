#' @title Computes the values
#' 
#' @description This is the description of the function that says the function computes an output value for every input
#' @export
#' @param FV \code{numeric} The future value of the thing
#' @param r \code{numeric} vector of legth l for the rate
#' @param n the number of years that the rate \code{r} is applied
#' @examples
#' \dontrun{
#' pv(FV = 1000, r = .08, n = 5)
#'[1] 680.58
#'pv(1000, .08, 5)
#'[1] 680.58
#'
#'pv(r = .08, FV = 1000, n = 5)
#'[1] 680.58
#' }

pv <- function(FV, r, n = 5) {
  if(!is.atomic(FV)) {
    stop('FV must be an atomic vector')
  }
  
  if(!is.numeric(FV) | !is.numeric(r) | !is.numeric(n)){
    stop('This function only works for numeric inputs!\n', 
         'You have provided objects of the following classes:\n', 
         'FV: ', class(FV), '\n',
         'r: ', class(r), '\n',
         'n: ', class(n))
  }
  
  if(r < 0 | r > .25) {
    message('The input for r exceeds the normal\n',
            'range for interest rates (0-25%)')
  }
  
  present_value <- FV / (1 + r)^n
  round(present_value, 2)
}
