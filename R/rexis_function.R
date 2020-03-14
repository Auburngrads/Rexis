#'
#'@import reticulate
#remove.packages("Rexis")


execute_py <- function(){
  datafile_string<-choose.files(default = "C:\\Users\\mchale\\OneDrive\\Documents\\AFIT\\WInter 20\\OPER 782 Data Science Programs\\Rexis\\data\\heart.csv", caption = "Select files",
                         multi = TRUE, filters = Filters,
                         index = nrow(Filters))
  #datafile <- readr::read_csv(file.choose())
  R_path<- "C:\\Users\\mchale\\Desktop"
  #R_path<-choose.dir(default =  "C:\\Users\\mchale\\Desktop", caption = "Select folder")
  out=py$Thesis_13_func_R(filepath_str=datafile_string, save_path=R_path)
  
  fig<-R_plot_func(out)
  
  out_list <- list(fig, out)
  
  return(out_list)  #changed this line
}


#execute_py <- function(){
# py$Thesis_13_func(filename_str, filesave_str, pltsave_str)
#}:
