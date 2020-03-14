#'@title rexicute 
#'@description this is the sample description for rexicute
#' @param datafile_string this is a text example
#' @examples rexicute ('C:/Users/mchale/OneDrive/Documents/AFIT/WInter 20/OPER 782 Data Science Programs/Rexis/data/heart.csv')
#'@import reticulate
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

  
  #'@title rexicute Shiny
  #'@description this is the sample description for rexicute
  #' @param datafile_string this is a text example
  #' @examples rexicute ('C:/Users/mchale/OneDrive/Documents/AFIT/WInter 20/OPER 782 Data Science Programs/Rexis/data/heart.csv')
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
