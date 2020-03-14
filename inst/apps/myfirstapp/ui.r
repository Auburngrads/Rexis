
                if (interactive()) {
                  
                  ui <- fluidPage(title = 'My First App!', 
                                  theme = shinythemes::shinytheme('flatly'),
                    titlePanel("Rexis"),
                    sidebarLayout(
                      sidebarPanel(
                        fileInput("file1", "Choose CSV File",
                                  accept = c(
                                    "text/csv",
                                    "text/comma-separated-values,text/plain",
                                    ".csv")
                        ),
                        tags$hr()
                        #checkboxInput("header", "Header", TRUE),
                        
                        
                                              ),
                      
                      
                      
                      mainPanel(
                        plotOutput("plot1"),
                       DT::dataTableOutput("contents"))))
                      
                  
                  }