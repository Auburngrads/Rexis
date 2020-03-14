# nocov start

.onLoad<- function(libname, pkgname){
  reticulate::import_from_path(module="Thesis_13_func_R", path=system.file("python", package="Rexis"))
  py_file = system.file("python", "Thesis_13_func_R.py", package = "Rexis")
  
  reticulate::source_python(py_file) #bridges R and python
}

.onUnload<-function(libpath){
  library.dynam.unload("Rexis", libpath)
}