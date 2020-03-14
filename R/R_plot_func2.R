R_plot_func <- function(out){
  #import ggplot
  library(ggplot2)
  #Read in plotly

  #Plot Bar Chart
  
  #Create Data Frame with plot info        
  df<-out
  df$Technique<-rownames(out)
  #Determine Hit and Hit Vectors, ect
  TopObsVect<- df$ObservedRanks==1
  TopRecVect<- df$RecRanks==1
  TopRecIndx<-which(df$RecRanks==1)
  TopRec<-df$Technique[TopRecIndx]
  HitVect<- TopObsVect*TopRecVect
  if (sum(HitVect>=1)) {hit<-TRUE} else {hit<-FALSE}
  
  
  
  fig <- ggplot(df, aes(Technique, MeanRecall, fill=TopRecVect)) + 
    geom_col()+ 
    ggtitle("Mean Recall")+
    theme(plot.title = element_text(hjust = 0.5))+
    ylim(0,1.09)+
    geom_label(x=1.2, y=1, label=paste("hit = ",hit))+
    geom_label(x=1.2, y=1.1, label=paste("Top Rec = ", df$Technique[TopRecIndx]) )+
    theme(legend.position="none") #removes legend
  
  fig
  
  return(fig)
}