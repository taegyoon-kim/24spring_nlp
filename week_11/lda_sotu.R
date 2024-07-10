###############################################################################   
#### HSS 510 Guide Coding: Topic Model                                     ####
#### 2024 May 8, Taegyoon Kim                                              ####
#### This tutorial is prepared by Jaehong Kim                              ####
#### (https://sociology.kaist.ac.kr/members/jaehong-kim)                   ####  
#### (https://ladal.edu.au/topicmodels.html)                               ####
###############################################################################



###############################################################################
################################### Set Up ####################################
###############################################################################

### packages -------------------------

# install pacman

if (!require("pacman", quietly = TRUE)) {
  install.packages("pacman") # check if "pacman" is installed and install if not
  }

# load the pacman package

library(pacman)

# use p_load to install and load the necessary packages

p_load(tm, slam, ldatuning, topicmodels, lda,
       ggplot2, dplyr, tidytext, furrr, tidyverse, 
       wordcloud, Rtsne, rsvd, geometry, NLP, lubridate)



###############################################################################
######################### Latent Dirichlet Allocation #########################
###############################################################################


### load and pre-process data -------------------------

# load and examine data (State of the Union Addresses by US presidents)

df <- readRDS(url("https://slcladal.github.io/data/sotu_paragraphs.rda", "rb"))
glimpse(df)
View(df)

# create corpus object

corpus <- Corpus(DataframeSource(df)) # from the tm package

# load stop words (unlikely these signal topics/themes)

english_stopwords <- readLines(
  "https://slcladal.github.io/resources/stopwords_en.txt", 
  encoding = "UTF-8")
length(english_stopwords)
head(english_stopwords)

# pre-processing chain

corpus_processed <- tm_map(corpus, content_transformer(tolower)) 
corpus_processed <- tm_map(corpus_processed, removeWords, english_stopwords) 
corpus_processed <- tm_map(corpus_processed, removePunctuation, 
                           preserve_intra_word_dashes = TRUE) 
corpus_processed <- tm_map(corpus_processed, removeNumbers) 
corpus_processed <- tm_map(corpus_processed, stemDocument, language = "en") 
corpus_processed <- tm_map(corpus_processed, stripWhitespace) 

# DTM (Document-Term Matrix)

dtm_no_thres <- DocumentTermMatrix(corpus_processed) 
dim(dtm_no_thres)

# another DTM

min_count <- 5

dtm <- DocumentTermMatrix(
  corpus_processed, 
  control = list(bounds = list(global = c(min_count, Inf))))

dim(dtm) # have a look at the number of documents and terms in the matrix

# remove rows for documents with no words 

sel_idx <- row_sums(dtm) > 0 
dtm <- dtm[sel_idx, ]

# update our data frame too

df <- df[sel_idx, ]


### determining the number of topics -------------------------

# fit topics models

result <- FindTopicsNumber(
  dtm,
  topics = seq(from = 10, to = 40, by = 10),
  metrics = c("Griffiths2004", # perplexity
              "CaoJuan2009"), # coherence
  method = "Gibbs",
  verbose = TRUE
  )

# see also for more diagnostics 
# https://github.com/doug-friedman/topicdoc
# https://github.com/doug-friedman/topicdoc?tab=readme-ov-file

# export and load "result"

save(result, file = "/Users/taegyoon/result.RData")
result <- get(load(file = "/Users/taegyoon/result.RData"))

# visualization

FindTopicsNumber_plot(result)


### estimate a model with K = 40 -------------------------

# number of topics

K <- 40

# set random number generator seed

set.seed(9161)

# estimate the LDA model, inference via 1000 iterations of Gibbs sampling

lda_40 <- LDA(
  dtm, 
  K, 
  method = "Gibbs", 
  control = list(iter = 1000, verbose = 25)) 

# posterior distributions 

tmResult <- posterior(lda_40)
tmResult$terms
tmResult$topics

# 40 topics (probability distributions over |V| = 4278)

beta <- tmResult$terms # get beta from results
dim(beta) # K distributions over nTerms (DTM) terms
rowSums(beta) # rows in beta sum to 1         

# 8810 documents (probability distribution over K = 20)

theta <- tmResult$topics 
dim(theta) # nDocs(dtm) distributions over K topics

# 10 most likely terms within the term probabilities beta of the inferred topics

rep_terms <- terms(lda_40, 20)

# 5 most representative documents for each topic

top_n <- 5
most_rep_docs <- apply(theta, 2, function(x) order(x, decreasing = TRUE)[1:top_n])

rep_docs_list <- list()

for (k in 1:K) {
  rep_docs_list[[k]] <- data.frame(
    Topic = k,
    Document = most_rep_docs[, k],
    Probability = theta[most_rep_docs[, k], k]
  )
}

rep_docs_df <- do.call(rbind, rep_docs_list)

print(rep_docs_df)
