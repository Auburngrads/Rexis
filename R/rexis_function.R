#'@title rexicute 
#'@description This function allows user to employ Rexis from the R console. 
#' @examples rexicute()
#'    A new window will allow user to select a .csv file. Navigate to Rexis/data/heart.csv
#'    #'@import reticulate
#'@export
#remove.packages("Rexis")


  rexicute <- function(datafile_string){
  datafile_string<-choose.files(default = "C:\\Users\\mchale\\OneDrive\\Documents\\AFIT\\WInter 20\\OPER 782 Data Science Programs\\Rexis\\data\\heart.csv", caption = "Select files",
                        multi = TRUE, filters = Filters,
                         index = nrow(Filters))
  #datafile_string<-"C:\\Users\\mchale\\OneDrive\\Documents\\AFIT\\WInter 20\\OPER 782 Data Science Programs\\Rexis\\data\\heart.csv"
  R_path<- "C:\\Users\\mchale\\Desktop"
  #R_path<-choose.dir(default =  "C:\\Users\\mchale\\Desktop", caption = "Select folder")
  out=py$Thesis_13_func_R(filepath_str=datafile_string, save_path=R_path)
  
  fig<-R_plot_func(out)
  
  return(fig)  #changed this line
}

  
  #'@title Rexicute for Shiny
  #'@description This R function provides a framework to launch the Rexis R Shiny app. 
  #' @param () there are no inputs to this function. An action button in the app will prompt user for a .csv file. 
  #' @examples rexicute()
  #'    click "browse" and navigate to Rexis/data/heart.csv
  #'@import reticulate
  #'@export
  #remove.packages("Rexis")
  
  
  rexicuteshiny <- function(datafile_string){
    #datafile_string<-choose.files(default = "C:\\Users\\mchale\\OneDrive\\Documents\\AFIT\\WInter 20\\OPER 782 Data Science Programs\\Rexis\\data\\heart.csv", caption = "Select files",
    # multi = TRUE, filters = Filters,
    #index = nrow(Filters))
    #datafile_string<-"C:\\Users\\mchale\\OneDrive\\Documents\\AFIT\\WInter 20\\OPER 782 Data Science Programs\\Rexis\\data\\heart.csv"
    R_path<- "C:\\Users\\mchale\\Desktop"
    #R_path<-choose.dir(default =  "C:\\Users\\mchale\\Desktop", caption = "Select folder")
    out=py$Thesis_13_func_R(filepath_str=datafile_string, save_path=R_path)
    
    fig<-R_plot_func(out)
    
    return(fig)  #changed this line
  }  
  
  
  

#execute_py <- function(){
# py$Thesis_13_func(filename_str, filesave_str, pltsave_str)
#}:
