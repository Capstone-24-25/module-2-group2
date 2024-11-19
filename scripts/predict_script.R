# Load libraries
library(tidyverse)
library(Matrix)

# Load models
binary_model <- load("../results/log_reg_binary.RData")
multi_model <- load("../results/log_reg_multi.RData")

# Load test data
load("../data/claims-test.RData")  # Update with the actual test data file path

# Ensure test data is preprocessed to match training data
## Clean the test data
claims_test_clean <- claims_test %>%
  parse_data()
  
  claims_test_clean %>% head(5)

## Tokenize
# Set seed for reproducibility
set.seed(1234)
# Tokenize into unigrams
test_unigrams <- claims_test_clean %>% 
  select(.id, text_clean) %>% 
  unnest_tokens(output = word, 
                input = text_clean, 
                token = 'words', 
                stopwords = str_remove_all(stop_words$word,'[[:punct:]]'))

## Convert to TF-IDF
# Count unigrams and compute TF-IDF
test_unigrams_tfidf <- test_unigrams %>% 
  count(.id, word, name = 'n') %>% 
  bind_tf_idf(term = word,
              document = .id, 
              n = n) %>% 
  filter(n>=5) %>%
  pivot_wider(id_cols = c(.id), 
              names_from = word,
              values_from = tf_idf,
              values_fill = 0)

### Dimensionality Reduction of Test Data
test_unigrams_tfidf_mtx <- test_unigrams_tfidf %>% select(-.id)

# PCA projection for training unigram data
test_dtm_unigrams_sparse <- test_unigrams_tfidf_mtx %>% 
  as.matrix() %>%
  as('sparseMatrix')


svd_out_unigrams <- sparsesvd(test_dtm_unigrams_sparse, rank=173)

# project test data onto PCs
test_dtm_projected<- reproject_fn1(.dtm = test_unigrams_tfidf_mtx, svd_out = svd_out_unigrams)


# Binary classification predictions
binary_preds <- predict(binary_model, newdata = test_dtm_projected, type = "response")

# Multi-class classification predictions
multi_preds <- predict(multi_model, newdata = test_dtm_projected, type = "response")

# Combine predictions
pred_df <- tibble(
  .id = test_unigrams_tfidf$.id,  # Ensure `.id` is accessible
  bclass.pred = factor(binary_preds, levels = c(1, 2), labels = c("N/A", "Relevant claim content")),
  mclass.pred = factor(multi_preds, levels = 1:5, labels = colnames(preds_multi_actual))
)


# Print example output
print(pred_df %>% head(5))
