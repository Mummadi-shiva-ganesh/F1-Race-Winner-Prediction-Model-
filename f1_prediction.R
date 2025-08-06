library(tidyverse)
library(tidytext)
library(readr)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
# Read CSV (adjust path if needed)
resumes <- read_csv("UpdatedResumeDataSet.csv")
# Unnest resume text into individual words
resumes_clean <- resumes %>%
  unnest_tokens(word, Resume) %>%
  anti_join(stop_words) %>%
  filter(!str_detect(word, "^[0-9]+$"))  # Remove numbers
ggplot(resumes, aes(x = fct_infreq(Category), fill = Category)) +
  geom_bar() +
  coord_flip() +
  labs(title = "Resume Count per Category", x = "Job Category", y = "Count") +
  theme_minimal()
word_freq <- resumes_clean %>%
  count(word, sort = TRUE)

wordcloud(words = word_freq$word,
          freq = word_freq$n,
          max.words = 100,
          colors = brewer.pal(8, "Dark2"),
          random.order = FALSE)
top_words <- resumes_clean %>%
  count(Category, word, sort = TRUE) %>%
  group_by(Category) %>%
  slice_max(order_by = n, n = 5)

print(top_words)
