server <- function(input, output, session) {
  dataSetInput <- eventReactive (input$file1,  {
    infile <- input$file1
    df <- read.csv(infile$datapath, header =  TRUE)
    return(df)
    })
  dataSetPath <- eventReactive (input$file1,  {
    infile <- input$file1
    path <- infile$datapath
    return(path)
  })
  
  output$contents <- DT::renderDataTable(dataSetInput())
  
  output$plot1 <- renderPlot({
    Rexis:::rexicuteshiny(dataSetPath())
  })
  #output$plot1<-renderPlot(fig)
  
  }
  # The shinyjs function call in the above app can be replaced by
  # any of the following examples to produce similar Shiny apps
  #onclick("rec", toggle("element"))
  #onclick(expr = text("element", date()), id = "btn")
  #{onclick("rec", info(date())); onclick("btn", info("Another message"), TRUE)}
  
  #onclick("rec", fig<-Rexis:::execute_py())
  
  
  #PlotAction<-eventReactive()
  