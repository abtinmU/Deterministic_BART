# Load required libraries
library(caret)
library(e1071)
library(mice)
library(dplyr)
library(readr)
library(utils)
library(xgboost)
library(neuralnet)
library(randomForest)
library(caTools)
library(nnet)
library(car)
library(glmnet)
library(tidyr)
library(ggplot2)
library(gridExtra)


'
The Balloon Analogue Risk Task (BART) is a popular tool for assessing risk-taking behavior. In this task, participants see a balloon on a computer screen, which they can inflate by pressing a key to earn a reward. However, each pump carries the risk of the balloon bursting, which would result in losing the reward. Participants must decide when to bank their earnings to maximize their total reward.

The data provided for this project comes from an experiment investigating the impact of alcohol consumption on risk-taking. This experiment used a within-subjects design, where participants were given three different doses of alcohol before completing the BART. Each participant went through three blocks of trials, each with a different probability of the balloon bursting. The blocks, consisting of 30 trials each, were presented in random order, and participants were informed of the burst probability before each block. The reward for each pump was a percentage of the money earned so far. In this project report, we conduct an analysis on the elements that effect the participnts` risk-taking and their reward.

The experiment data is shared in a folder containing `name-id.txt` files for each participant separately. Experiment design consists of 3 sessions of participants with different conditions (sober, tipsy, drunk), 3 different burst probabilities i different sessions (0.1, 0.15, 0.2), and a total of 3 trials per block. Meaning that each participant has a total of 90 rows of data in their data file. 

The relationship between sessions and conditions is claimed to be randomized, and we are provided a mapping matrix between participants and their condition in each session. The condition identifier matrix below indicates, for example, that for participant 1, session 1 was the drunk condition, session 2 was the tipsy condition, and session 3 was the sober condition. For participant 3, session 1 was tipsy, session 2 was drunk, and session 3 was sober.

[3 1 2 2 1 3 2 3 1 2 2 3 2 1 1 1 3 3]

[2 2 3 3 3 1 1 2 2 1 1 1 3 3 2 3 2 1]

[1 3 1 1 2 2 3 1 3 3 3 2 1 2 3 2 1 2]
'
set.seed(123)
knitr::opts_chunk$set(
  echo = FALSE,
  message = FALSE,
  warning = FALSE
)
# the condition_info array
condition_info <- matrix(c(3, 1, 2, 2, 1, 3, 2, 3, 1, 2, 2, 3, 2, 1, 1, 1, 3, 3,
                           2, 2, 3, 3, 3, 1, 1, 2, 2, 1, 1, 1, 3, 3, 2, 3, 2, 1,
                           1, 3, 1, 1, 2, 2, 3, 1, 3, 3, 3, 2, 1, 2, 3, 2, 1, 2),
                         nrow = 3, byrow = TRUE)


## Data preperation
'
The first step of the project is to combine all the data files into a single, organized data frame. In this step, I extracted the data of different patients from their corresponding txt files. Then I organized all the information provided in the data and generated the following 10 variables:

- participant: identifies the participant. Although its numerical, its essentially categorical since it represents different people.
- condition: represents the condition of each participant in each session
- p_burst: represents the probability of a balloon bursting in each session
- trial: represents the trial number of each participant in each session
- pumps: the number of times a balloon is pumped for each participant in each point of a session
- cash: the amount of money gained in each trial
- total: total reward of a block
- session: an integer representing a session (1, 2, or 3)
- block: indicates which block the trial belongs to
- explosion: indicates if the balloon burst
'

# List and sort the text files
files <- list.files(path = "bart-data\\bart-data", pattern = "\\.txt$", full.names = TRUE)
files <- sort(files)

# Initialize an empty dataframe
tidy_data <- data.frame(participant = integer(),
                        condition = character(),
                        p_burst = integer(),
                        trial = integer(),
                        pumps = integer(),
                        cash = numeric(),
                        total = numeric(),
                        session = integer(),
                        stringsAsFactors = FALSE)

participanti <- 0

dfs <- list()

for (file in files) {
  
  # Read the data
  data <- read_tsv(file, skip = 1, col_names = c('pres.bl', 'block', 'gr.fact', 'prob.', 'trial', 'pumps', 'cash', 'total'),
                   col_types = cols(
                     `pres.bl` = col_double(),
                     `block` = col_double(),
                     `gr.fact` = col_double(),
                     `prob.` = col_double(),
                     `trial` = col_double(),
                     `pumps` = col_double(),
                     `cash` = col_double(),
                     `total` = col_double()
                   ))
  
  # Extract session number
  session_str <- strsplit(basename(file), "_")[[1]][2]
  session <- as.numeric(sub("\\.txt$", "", session_str)) - 1
  
  # Determine participant
  participant <- (participanti) %/% 3
  
  # Determine condition
  condition <- condition_info[session + 1, participant + 1]
  
  condition <- ifelse(condition == 1, 'sober',
                      ifelse(condition == 2, 'tipsy', 'drunk'))
  
  # Determine p_burst
  p_burst <- ifelse(data$block == 1, 10, 
                    ifelse(data$block == 2, 15, 
                           ifelse(data$block == 3, 20, NA)))
  
  df <- data.frame(participant = participant + 1,
                   condition = condition,
                   p_burst = p_burst,
                   trial = data$trial,
                   pumps = data$pumps,
                   cash = data$cash,
                   total = data$total,
                   session = session + 1,
                   block = data$block,
                   stringsAsFactors = FALSE)
  
  dfs <- append(dfs, list(df))
  
  participanti <- participanti + 1
}

# Concatenate all dataframes into one
final_df <- bind_rows(dfs)

final_df <- final_df %>%
  mutate(explosion = cash == 0)

# Concatenate all dataframes into one
final_df <- bind_rows(dfs)

final_df <- final_df %>%
  mutate(explosion = cash == 0)

# Define a standardize function
standardize <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

print(sample_n(final_df, 10))


final_df$participant <- as.factor(final_df$participant)
final_df$condition <- as.factor(final_df$condition)
final_df$session <- as.factor(final_df$session)
final_df$block <- as.factor(final_df$block)
final_df$trial <- as.factor(final_df$trial)
final_df$p_burst <- as.factor(final_df$p_burst/100)


## Missingness
'
Considering that our data does not have any missing data, we generate them by ourselves. For this purpose, I randomly deleted 25% of the data from each variable. Then, I decided on a suitable imputation method to restore the missing values. In this step, I directly used Predictive Mean Matching  for numerical variables and Polynomial Regression imputation for factors. The reason I didnt try Single Imputation is that my data was specifically designed with different participant and different conditions of participants in mind. These differences are critical to have the best analysis of the data later. If we just use Single Imputation and replace the missing values of all variables with the same mean/median values, these differences would be lost. At the same time, NOCB  and LOCF were not suitable because they would have reslted the data of different participants and different conditions being mixed together.

Now, if we collapse the 4060 rows (90 rows * 17 participants) of our generated above data with respect to the trials in a block, we would be able to use a new version of this data with 162 rows (3 * 3 * 17) and we can generate 6 new useful variables:
  
- mNOP: mean number of pumps across trials, within a block
- pC: proportion of cashed trials in a block
- pE: proportion of explosions in a block (with respect to the total number of pumps in a block)
- C: the total number of times the participant cashed the balloons in each block
- pE: proportions of cashed trials in each block for each participant
- NOE: number of explosions in each block for each participant

In the end, we set the suitable type for each variable (factor/numeric) and check the summary of our new imputed data and new variables.
'

# Create a copy of final_df to introduce missing values
df_missing <- final_df

# Function to introduce NAs randomly
introduce_nas <- function(x, perc) {
  n <- length(x)
  n_na <- round(n * perc)
  na_indices <- sample(seq_len(n), size = n_na)
  x[na_indices] <- NA
  return(x)
}

# Apply the function to introduce 25% NAs in each column
df_missing <- df_missing %>%
  mutate(across(everything(), ~ introduce_nas(., 0.05)))

print("Data with missing information:")
print(head(df_missing))

# Check the proportion of NAs
sapply(df_missing, function(x) sum(is.na(x)) / length(x))

# Specify the imputation methods for each variable
methods <- c(participant = "polyreg",
             condition = "polyreg",
             p_burst = "polyreg",
             trial = "polyreg",
             pumps = "pmm",
             cash = "pmm",
             total = "pmm",
             session = "polyreg",
             block = "",
             explosion = "pmm")

# Perform the imputation using the mice package
imputed_data <- mice(df_missing, method = methods, m = 1, maxit = 5, seed = 123)

# Extract the complete dataset after imputation
complete_data <- complete(imputed_data, 1)

# Seperate imputation for the block variable to avoid logged events error
methods_block <- c(
  '',          # participant
  '',          # condition
  '',          # p_burst
  '',          # trial
  '',          # pumps
  '',          # cash
  '',          # total
  '',          # session
  'polyreg',   # block
  ''           # explosion
)

imputed_block <- mice(complete_data, m = 1, method = methods_block, maxit = 5, seed = 123)
complete_data$block <- complete(imputed_block)$block
head(complete_data)
final_df = complete_data

head(final_df)
str(final_df)

# Determine the type of each variable in the dataframe
variable_types <- sapply(complete_data, function(x) if(is.numeric(x)) "numeric" else "factor")

# Iterate over all pairs of variables
for (i in 1:(ncol(complete_data) - 1)) {
  for (j in (i + 1):ncol(complete_data)) {
    var1 <- names(variable_types)[i]
    var2 <- names(variable_types)[j]
    
    # Check if either variable is a factor and convert to numeric ranks if so
    if (variable_types[i] == "factor") {
      x <- as.numeric(as.factor(complete_data[[var1]]))
    } else {
      x <- complete_data[[var1]]
    }
    
    if (variable_types[j] == "factor") {
      y <- as.numeric(as.factor(complete_data[[var2]]))
    } else {
      y <- complete_data[[var2]]
    }

    # Use Spearman's correlation since we're using ranks now
    correlation <- cor(x, y, method = "spearman", use = "complete.obs")
    method_used <- "Spearman"

    # Print each result in the specified format
    print(paste(var1, "and", var2, "correlation using", method_used, "is", round(correlation, 4)))
  }
}


## EDA and CDA
'
Now we visualize and apply suitable statistical tests to analyze the data. The focus of my analysis would be the following research question:
  
  - How does the participants` condition effect their maximum risk taking (maximum number of times that participants pump a balloon)?
  - How does the participants` condition effect their average risk taking?
  - How does the participants` prior on burst probability of balloons effect their maximum risk taking?
  - How does the participants` prior on burst probability of balloons effect their average risk taking?
  - What about the pumps that lead to explosions? Does increase risk-taking in low burst probabilities result in a significantly higher number of explosions?
  - Do different burst probabilities have any effects on total rewards?
  - Do different conditions have any effects on total rewards?
  - How do individual participants differ in their mean total rewards across trials?
  - How do individual participants differ in their mean number of pumps in different conditions?
'

fina_df = complete_data

# Calculate the max number of pumps for each participant, condition, and p_burst
max_pumps_df <- final_df %>%
  group_by(participant, condition, p_burst) %>%
  summarise(max_pumps = max(pumps, na.rm = TRUE), .groups = 'drop') %>%
  ungroup()

# Create the plot
g <- ggplot(max_pumps_df, aes(x = participant, y = max_pumps, color = condition, group = condition)) +
  geom_point(position = position_dodge(width = 0.75)) +
  geom_line(position = position_dodge(width = 0.75)) +
  facet_grid(p_burst ~ condition, scales = "free_y") +
  labs(x = 'Participant', y = 'Max Number of Pumps', title = 'Max Number of Pumps by Participant, Condition, and Burst Probability') +
  theme_bw() +
  scale_color_manual(values = c('drunk' = 'red', 'tipsy' = 'blue', 'sober' = 'green')) +
  theme(strip.text = element_text(size = 10),
        axis.text.x = element_text(angle = 90, hjust = 1),
        legend.position = "bottom")

# Print the plot
print(g)

'
We can observe a general trend here: although different alcohol condition doesnt have much of an effect of the maximum risk-taking of participants, the differences in probability of burst seems to have a rather significant effect. We can see that higher probability of burst values have resulted participants to have a smaller maximum pump number. We check this using a two-way anova test.
'

# Calculate the max number of pumps for each participant, condition, and p_burst
final_df <- final_df %>%
  mutate(p_burst = as.factor(p_burst),
         condition = as.factor(condition))

# Run the two-way ANOVA
anova_result <- aov(max_pumps ~ p_burst * condition, data = max_pumps_df)
summary(anova_result)

'
The p-value is extremely small. It is clear that we can reject the null hypothesis and conclude a significant effect from burst probability on maximum number of pumps. While the condition has a big p-value and we fail to find any significant effect from condition to maximum risk-taking of participants. And the result of the Tukey multiple comparisons of means confirms that as burst probability increases the maximum pump number decreases. 
'

# Post-hoc test for p_burst effect
posthoc_p_burst <- TukeyHSD(anova_result, "p_burst")
print(posthoc_p_burst)

'
Now we check if the average number of pumps is also as significantly effected as the maximum number of pumps. Especially, we need to check if the condition of participants shows any significant effect on their average risk-taking.
'

# Calculate the mean number of pumps for each participant, condition, and p_burst
mean_pumps_df <- final_df %>%
  group_by(participant, condition, p_burst) %>%
  summarise(mean_pumps = mean(pumps, na.rm = TRUE), .groups = 'drop') %>%
  ungroup()

# Create the plot
g <- ggplot(mean_pumps_df, aes(x = participant, y = mean_pumps, color = condition, group = condition)) +
  geom_point(position = position_dodge(width = 0.75)) +
  geom_line(position = position_dodge(width = 0.75)) +
  facet_grid(p_burst ~ condition, scales = "free_y") +
  labs(x = 'Participant', y = 'Mean Number of Pumps', title = 'Burst Probability: {col_name}') +
  theme_bw() +
  scale_color_manual(values = c('drunk' = 'red', 'tipsy' = 'blue', 'sober' = 'green')) +
  theme(strip.text = element_text(size = 10),
        axis.text.x = element_text(angle = 90, hjust = 1),
        legend.position = "bottom")

# Print the plot
print(g)

'
Similar to the previous part, we observe an effect from burst probability but no effect from the condition. We use another two-way anova text and check this.
'

# Calculate the mean number of pumps for each participant, condition, and p_burst
mean_pumps_df <- final_df %>%
  group_by(participant, condition, p_burst) %>%
  summarise(mean_pumps = mean(pumps, na.rm = TRUE), .groups = 'drop') %>%
  ungroup()

# Convert p_burst and condition to factors
mean_pumps_df <- mean_pumps_df %>%
  mutate(p_burst = as.factor(p_burst),
         condition = as.factor(condition))

# Run the two-way ANOVA
anova_result <- aov(mean_pumps ~ p_burst * condition, data = mean_pumps_df)
summary(anova_result)

'
We again observe a significant effect from p_burst but no effect from condition.
'

# Post-hoc test for p_burst effect
posthoc_p_burst <- TukeyHSD(anova_result, "p_burst")
print(posthoc_p_burst)

'
Applying Tukey multiple comparisons of means confirms shows us that higher burst probabilities decrease participants average risk-taking.

The following 2 plots allow us to further confirm the effect of burst probability on risk-taking. But we also observe a new observation: although lower burst probabilities show a higher risk-taking, they also show a higher variance. Indicating that although in low burst probabilities some participants take a lot of risks, but there are also some participants that still remain careful.
'

# Calculate the grand mean of the number of pumps
grand_mean_pumps <- mean(final_df$pumps, na.rm = TRUE)
print(sprintf("Grand Mean of Number of Pumps: %.2f", grand_mean_pumps))

# Calculate the means and standard deviations by burst probability and trial
summary_burst <- final_df %>%
  group_by(trial, p_burst) %>%
  summarise(mean_pumps = mean(pumps, na.rm = TRUE),
            sd_pumps = sd(pumps, na.rm = TRUE), .groups = 'drop') %>%
  ungroup()

# Create the plot
plot1 <- ggplot(summary_burst, aes(x = trial, y = mean_pumps, color = p_burst, group = p_burst)) +
  geom_line(linewidth = 1.5) +
  geom_ribbon(aes(ymin = mean_pumps - sd_pumps, ymax = mean_pumps + sd_pumps), alpha = 0.2) +
  geom_hline(yintercept = grand_mean_pumps, linetype = "dashed", color = "black") +
  labs(title = "Number of Pumps by Burst Probability",
       x = "Trial",
       y = "Number of Pumps",
       color = "Burst Probability") +
  scale_color_brewer(palette = "Dark2") +
  theme_minimal() +
  theme(legend.position = "top")

# Calculate the means and standard deviations by condition and trial
summary_condition <- final_df %>%
  group_by(trial, condition) %>%
  summarise(mean_pumps = mean(pumps, na.rm = TRUE),
            sd_pumps = sd(pumps, na.rm = TRUE), .groups = 'drop')

# Create the plot
plot2 <- ggplot(summary_condition, aes(x = trial, y = mean_pumps, color = condition, group = condition)) +
  geom_line(size = 1.5) +  # Thicker line for the main plot
  geom_ribbon(aes(ymin = mean_pumps - sd_pumps, ymax = mean_pumps + sd_pumps), alpha = 0.2) +
  geom_hline(yintercept = grand_mean_pumps, linetype = "dashed", color = "black") +
  labs(title = "Number of Pumps by Condition",
       x = "Trial",
       y = "Number of Pumps",
       color = "Condition") +
  theme_minimal() +
  theme(legend.position = "top")

grid.arrange(plot1, plot2, ncol = 2, widths=c(10,10))

'
To test this, we apply Levenes Test and observe a significant difference in the variances of the mean number of pumps across different burst probabilities. The p-value is extremely small whic means that the null hypothesis (that the variances are equal) can be rejected with high confidence. 
'

# Perform Levene's Test
levene_test_result <- leveneTest(pumps ~ as.factor(p_burst), data = final_df)

# Print the results of Levene's Test
print(levene_test_result)

'
We already observed the surprising result that the condition of participants, in general, doesnt have an effect on the number of pumps. Now, we analyze this across different participants. In the following plot, we observe that different participants seem to have many differences in this regard.
'

# Number of Pumps by Participant and Condition
g <- ggplot(final_df, aes(x = participant, y = pumps, fill = condition)) +
  geom_bar(stat = "summary", fun = "mean", position = "dodge") +
  labs(x = "Participant", y = "Number of Pumps", title = "Number of Pumps by Participant and Condition") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")

print(g)

'
This shows the reason behind the surprising conclusiojn we had on the effect of condition on pumps before. It seems the reason we failed to find a significant effect before is that alcohol (condition) has vastly different effects on different participants. It increases the risk-taking of some participants, decreases the risk-taking of another group of participant, and it doesnt effect some participants. This is why  testing the grand average between all participants gave us the conclusion that "condition doesnt effect risk-taking"! Because the effect of alcohol on different participants could be the total opposite of each other. Now, we conduct independents anova tests for different participants and test this:
'

# Loop through participants 1 to 17
for (participant_id in 1:17) {
  
  # Filter the data for the current participant
  participant_data <- final_df %>% filter(participant == participant_id)
  
  # Perform ANOVA to check if different conditions affect the average number of pumps
  anova_result <- aov(pumps ~ condition, data = participant_data)
  
  # Get the p-value from the ANOVA summary
  p_value <- summary(anova_result)[[1]][["Pr(>F)"]][1]
  
  # Print the result for the current participant
  if (p_value < 0.05) {
    cat("Participant", participant_id, ": Significant effect (p-value =", p_value, ")\n")
  } else {
    cat("Participant", participant_id, ": No significant effect (p-value =", p_value, ")\n")
  }
}

'
Now the results sound more logical. We observe that alcohol has no significant effect on the number of pumps of 9 participants. But it has significant effects on the other 8 participants.

Now, we check if different burst probabilities have a significant effect on the number of explosions.
'

# Calculate the mean number of pumps by participant, condition, p_burst, and explosion
mean_pumps_explosion <- final_df %>%
  group_by(participant, condition, p_burst, explosion) %>%
  summarise(mean_pumps = mean(pumps, na.rm = TRUE), .groups = 'drop')

# Calculate the mean number of pumps by p_burst, condition, and explosion for the plot
mean_pumps_explosion_for_plot <- final_df %>%
  group_by(p_burst, condition, explosion) %>%
  summarise(mean_pumps = mean(pumps, na.rm = TRUE), .groups = 'drop')

#print(nrow(mean_pumps_explosion))  # 18 * 3 * 3 * 2 = 324
#print(nrow(mean_pumps_explosion_for_plot)) # 3 * 3 * 2 = 18

# Create the plot
g <- ggplot(mean_pumps_explosion_for_plot, aes(x = p_burst, y = mean_pumps, color = condition, group = condition)) +
  geom_point(position = position_dodge(width = 0.3)) +
  geom_line(position = position_dodge(width = 0.3)) +
  facet_wrap(~ explosion, scales = "free_y", labeller = labeller(explosion = c(`FALSE` = "Explosion: FALSE", `TRUE` = "Explosion: TRUE"))) +
  labs(x = "Burst Probability", y = "Mean Number of Pumps", color = "Condition") +
  theme_minimal() +
  theme(legend.position = "top")

# Print the plot
print(g)

'
We observe a clear connection between them. We also observe that the higher risk average pump number comes from 0.1 burst probability from the cases that balloon has not exploded. For better p_burst vs explosion interpretation, we check the frequency table and odds ratio.
'

odds.ratio = function(x, conf.level=0.95) {
  OR = x[1,1] * x[2,2] / ( x[2,1] * x[1,2] )
  SE = sqrt(sum(1/x))
  CI = exp(log(OR) + c(-1,1) * qnorm(0.5*(1-conf.level), lower.tail=F) * SE )
  list(estimator=OR,
       SE=SE,
       conf.interval=CI,
       conf.level=conf.level)
}

expl_mar=table(final_df$p_burst,final_df$explosion)

print(expl_mar)
odds.ratio(expl_mar)

'
We can observe that there is a statistically significant association p_burst and explosion. The odds of an explosion are lower in the higher p_burst values.The chi-square test of independence also confirms the dependence of explosion on p_burst.
'
chisq.test(expl_mar)

mosaicplot(table(final_df$explosion, final_df$p_burst),main='Mosaic plot for Burst Probability vs Explosion',col=c("darkgreen","royalblue","orange"))

# Calculate the mean and standard error for each combination of trial, condition, and p_burst
mean_se_total_per_trial <- final_df %>%
  group_by(trial, condition, p_burst) %>%
  summarise(
    mean_total = mean(total, na.rm = TRUE),
    se_total = sd(total, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )

# Create the plot using facet_wrap
plot <- ggplot(mean_se_total_per_trial, aes(x = trial, y = mean_total, color = p_burst, group = p_burst)) +
  geom_line(size = 0.5) +  # Decrease line width
  geom_point(size = 1.5) +  # Add points with decreased size
  geom_ribbon(aes(ymin = mean_total - se_total, ymax = mean_total + se_total, fill = p_burst), alpha = 0.2) +
  facet_wrap(~ condition, scales = "free_y") +
  scale_color_viridis_d() +
  scale_fill_viridis_d() +
  labs(x = "Trial", y = "Total Reward", color = "Burst Probability", fill = "Burst Probability",
       title = "Total Reward by Trial, Condition, and Burst Probability with CI") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Print the plot
print(plot)


# Calculate the average total rewards per participant
avg_total_per_participant <- final_df %>%
  group_by(participant, condition, p_burst, trial) %>%
  summarise(mean_total = mean(total, na.rm = TRUE), .groups = 'drop')

# Filter the first 18 participants
subset_first_five <- avg_total_per_participant %>%
  filter(participant %in% 1:18)

# Create the plot using facet_grid
plot <- ggplot(subset_first_five, aes(x = as.numeric(trial), y = mean_total, color = as.factor(participant))) +
  geom_line(size = 0.5) +  # Decrease line width
  geom_point(size = 0.5) +  # Decrease point size
  facet_grid(condition ~ p_burst, labeller = labeller(p_burst = label_both, condition = label_both)) +
  labs(x = "Trial", y = "Mean Total Rewards", color = "Participant",
       title = "Mean Total Rewards by Condition and Burst Probability") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Print the plot
print(plot)


# Load necessary libraries

# Calculate the average total rewards per participant
avg_total_per_participant <- final_df %>%
  group_by(participant, condition, p_burst, trial) %>%
  summarise(mean_total = mean(total, na.rm = TRUE), .groups = 'drop')

# Fit the repeated measures ANOVA model
# Convert relevant columns to factors
avg_total_per_participant$participant <- as.factor(avg_total_per_participant$participant)
avg_total_per_participant$condition <- as.factor(avg_total_per_participant$condition)
avg_total_per_participant$p_burst <- as.factor(avg_total_per_participant$p_burst)
avg_total_per_participant$trial <- as.factor(avg_total_per_participant$trial)

# Define the model
aov_model <- aov(mean_total ~ condition * p_burst * trial + Error(participant/(condition * p_burst * trial)), data = avg_total_per_participant)

# Print the summary of the ANOVA model
summary(aov_model)

# Check assumptions
# Normality of residuals
qqnorm(residuals(aov_model[[1]]))
qqline(residuals(aov_model[[1]]))

# Homogeneity of variances (if needed)
leveneTest(mean_total ~ condition * p_burst * trial, data = avg_total_per_participant)


# Pivot the data to get vectors for participants, conditions, and burst probabilities
participant_vectors <- final_df %>%
  select(trial, participant, total) %>%
  group_by(trial, participant) %>%
  summarize(total = mean(total)) %>%
  pivot_wider(names_from = participant, values_from = total)

condition_vectors <- final_df %>%
  select(trial, condition, total) %>%
  group_by(trial, condition) %>%
  summarize(total = mean(total)) %>%
  pivot_wider(names_from = condition, values_from = total)

p_burst_vectors <- final_df %>%
  select(trial, p_burst, total) %>%
  group_by(trial, p_burst) %>%
  summarize(total = mean(total)) %>%
  pivot_wider(names_from = p_burst, values_from = total)

colnames(p_burst_vectors)[2:4] <- c("0.1", "0.15", "0.2")

# Combine these vectors into a single data frame
vectors_df <- bind_cols(participant_vectors, condition_vectors[-1], p_burst_vectors[-1])


# Calculate the grand average for each trial
grand_average <- final_df %>%
  group_by(trial) %>%
  summarise(total = mean(total, na.rm = TRUE))

# Function to calculate mean squared deviation
mean_squared_deviation <- function(x) {
  grand_mean <- mean(x, na.rm = TRUE)
  mean((x - grand_mean)^2, na.rm = TRUE)
}


# Calculate the mean squared deviation for each column
scores <- sapply(vectors_df[-1], mean_squared_deviation)

# Define the index ranges and variable names
iloc_indices <- list(2:4, 5:7, 8:25)
var_names <- c('pBurst', 'condition', 'participant')

# Initialize a list to hold the plots
plot_list <- list()

# Loop through each variable type
for (i in 1:length(iloc_indices)) {
  var_idx <- iloc_indices[[i]]
  var_scores <- scores[var_idx]
  max_score <- max(var_scores)
  idx_max <- which.max(var_scores)
  max_score_val <- colnames(vectors_df)[var_idx[idx_max]]
  var_vals <- colnames(vectors_df)[var_idx]
  sub_df <- vectors_df %>% select(trial, all_of(var_vals))
  
  sub_df_long <- pivot_longer(sub_df, -trial, names_to = "variable", values_to = "value")
  
  # Convert 'trial' to numeric
  sub_df_long$trial <- as.numeric(sub_df_long$trial)
  
  # Create the plot
  p <- ggplot() +
    geom_line(data = grand_average, aes(x = as.numeric(trial), y = total, group = 1), color = 'green', size = 1.5, alpha = 0.8) +
    geom_line(data = sub_df_long %>% filter(variable == max_score_val), aes(x = trial, y = value, group = variable), color = 'red', size = 1) +
    geom_line(data = sub_df_long %>% filter(variable != max_score_val), aes(x = trial, y = value, group = variable), color = 'black', size = 0.5) +
    labs(title = paste('Max deviant', var_names[i], ':', max_score_val),
         x = 'Trials',
         y = 'Score') +
    theme_minimal()
  
  plot_list[[i]] <- p
}

# Combine the plots into a single figure
grid.arrange(grobs = plot_list, ncol = 3, top = "Deviant Analysis")


## Statistical Modeling

# Fit the logistic regression model
logistic_model <- glm(explosion ~ condition + p_burst + pumps + session + participant, data = final_df, family = binomial)

# Print the summary of the model
summary(logistic_model)

# Convert p_burst to numeric and cube it
final_df$pb <- as.numeric(as.character(final_df$p_burst))
final_df$pb <- final_df$pb^(2)

# Normalize the 'pumps' variable
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
final_df$pumps_n <- normalize(final_df$pumps)

# Create interaction terms
final_df$condition_pb <- as.numeric(final_df$condition) * final_df$pb

# One-hot encode the 'session' variable
session_dummies <- model.matrix(~ session - 1, data = final_df)
final_df <- cbind(final_df, session_dummies)

# Fit the logistic regression model
logistic_model <- glm(explosion ~ condition_pb * session3 + pb * pumps+pb +participant, 
                      data = final_df, family = 'binomial')

# Print the summary of the model
summary(logistic_model)

# Calculate VIF to check for multicollinearity
vif(logistic_model)


### Logistic Regression

#final_df$explosion <- ifelse(final_df$explosion==TRUE, 1, 0)
set.seed(123)  
split <- sample.split(final_df$explosion, SplitRatio = 0.7)
train_data <- subset(final_df, split == TRUE)
test_data <- subset(final_df, split == FALSE)

# Define the control for cross-validation
train.control = trainControl(method = "repeatedcv", number = 10)  # 10-fold cross-validation

# Train the logistic regression model using k-fold cross-validation
model_kfold = train(
  explosion ~ condition_pb * session3 + pb * pumps + pb + participant, 
  data = train_data, 
  method = "glm", 
  family = "binomial", 
  trControl = train.control
)

# Print cross-validation results
print(model_kfold)

# Predict on the test set
test_predictions <- predict(model_kfold, newdata = test_data)

# Convert probabilities to binary outcomes (using 0.5 as the threshold)
final_predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)

# Generate the confusion matrix for the test set
final_cm <- confusionMatrix(as.factor(final_predicted_classes), as.factor(test_data$explosion))
print("Confusion Matrix for Test Set:")
print(final_cm)


### Neural Networks

# Normalize the data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization to the relevant columns in train and test sets
train_data_normalized <- as.data.frame(lapply(train_data[, sapply(train_data, is.numeric)], normalize))
test_data_normalized <- as.data.frame(lapply(test_data[, sapply(test_data, is.numeric)], normalize))

# Add non-numeric columns back to the normalized data
train_data_normalized$explosion <- train_data$explosion
train_data_normalized$condition_pb <- train_data$condition_pb
train_data_normalized$session3 <- train_data$session3
train_data_normalized$pb <- train_data$pb
train_data_normalized$participant <- train_data$participant

test_data_normalized$explosion <- test_data$explosion
test_data_normalized$condition_pb <- test_data$condition_pb
test_data_normalized$session3 <- test_data$session3
test_data_normalized$pb <- test_data$pb
test_data_normalized$participant <- test_data$participant

# Grid search for tuning the neural network model
layer_options <- 1:6
neuron_options <- c(1, 2, 4, 6, 8)
results <- data.frame(Neurons = integer(), Accuracy = numeric())

for (neurons in neuron_options) {
  # Train the neural network model with the current configuration
  nn_model <- nnet(explosion ~ condition_pb * session3 + pb * pumps + pb + participant, 
                   data = train_data_normalized, size = neurons, decay = 0.1, maxit = 200, linout = TRUE)
  
  # Predict on the test set
  nn_predictions <- predict(nn_model, newdata = test_data_normalized, type = 'raw')
  
  # Convert probabilities to binary outcomes (using 0.5 as the threshold)
  nn_predicted_classes <- ifelse(nn_predictions > 0.5, 1, 0)
  
  # Ensure the factor levels match
  nn_predicted_classes <- factor(nn_predicted_classes, levels = c(0, 1))
  actual_classes <- factor(test_data$explosion, levels = c(0, 1))
  
  # Generate the confusion matrix
  nn_confusion_matrix <- confusionMatrix(nn_predicted_classes, actual_classes)
  
  # Calculate accuracy
  accuracy <- nn_confusion_matrix$overall['Accuracy']
  
  # Store the results
  results <- rbind(results, data.frame(Neurons = neurons, Accuracy = accuracy))
}
# Print the results
print(results)

# Identify the best configuration
best_config <- results[which.max(results$Accuracy),]
print(best_config)

# Train the best model separately
best_neurons <- best_config$Neurons

# Fit the neural network model with the best configuration
best_nn_model <- nnet(explosion ~ condition_pb * session3 + pb * pumps + pb + participant, 
                      data = train_data_normalized, size = best_neurons, decay = 0.1, maxit = 200, linout = TRUE)

# Predict on the test set with the best model
best_nn_predictions <- predict(best_nn_model, newdata = test_data_normalized, type = 'raw')

# Convert probabilities to binary outcomes (using 0.5 as the threshold)
best_nn_predicted_classes <- ifelse(best_nn_predictions > 0.5, 1, 0)

# Generate the confusion matrix for the best model
best_nn_confusion_matrix <- confusionMatrix(as.factor(best_nn_predicted_classes), as.factor(test_data$explosion))

# Print the confusion matrix
print(best_nn_confusion_matrix)


### random forest       

# Fine-tuning the Random Forest model using cross-validation
# Define the tuning grid
tune_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))

# Train control for cross-validation
train_control <- trainControl(method = "cv", number = 5)

# Train the Random Forest model with fine-tuning
tuned_rf_model <- train(explosion ~ condition_pb * session3 + pb * pumps + pb + participant, 
                        data = train_data, method = "rf", trControl = train_control, tuneGrid = tune_grid)

# Print the best tuning parameters
print(tuned_rf_model$bestTune)

# Predict on the test set with the tuned model
tuned_rf_predictions <- ifelse(predict(tuned_rf_model, newdata = test_data) > 0.5, 1, 0)

# Generate the confusion matrix for the tuned model
tuned_rf_confusion_matrix <- confusionMatrix(as.factor(tuned_rf_predictions), as.factor(test_data$explosion))
print(tuned_rf_confusion_matrix)

# Check variable importance
variable_importance <- varImp(tuned_rf_model)

# Print variable importance
print(variable_importance)

# Convert variable importance to a data frame for plotting
importance_df <- as.data.frame(variable_importance$importance)
importance_df$Variable <- rownames(importance_df)
importance_df <- importance_df[order(importance_df$Overall, decreasing = TRUE), ]

# Select top 10 important variables
top_10_importance_df <- head(importance_df, 10)

# Create a color vector, with the most important variable in red
top_10_importance_df$color <- c("red", rep("black", nrow(top_10_importance_df) - 1))

# Plot the top 10 variable importance using ggplot2
ggplot(top_10_importance_df, aes(x = reorder(Variable, Overall), y = Overall, fill = color)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_identity() +
  xlab("Variables") +
  ylab("Importance") +
  ggtitle("Top 10 Variable Importance in Random Forest Model")


### XGBoost
# Define the tuning grid
tune_grid <- expand.grid(
  nrounds = c(50, 150),
  eta = c(0.01, 0.2),
  max_depth = c(3, 9),
  gamma = c(0, 5),
  colsample_bytree = c(0.5, 1.0),
  min_child_weight = c(1, 10),
  subsample = c(0.8, 1.0)
)

# Train control for cross-validation
train_control <- trainControl(method = "cv", number = 3, verboseIter = TRUE)

# Train the XGBoost model with fine-tuning
tuned_xgb_model <- train(
  explosion ~ condition_pb * session3 + pb * pumps + pb + participant, 
  data = train_data, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = tune_grid
)

# Print the best tuning parameters
print(tuned_xgb_model$bestTune)

# Predict on the test set with the tuned model
tuned_xgb_predictions <- predict(tuned_xgb_model, newdata = test_data)

# Ensure the predicted classes are factors with the same levels as the test set
tuned_xgb_predictions <- factor(ifelse(tuned_xgb_predictions > 0.5, 1, 0))

# Generate the confusion matrix for the tuned model
tuned_xgb_confusion_matrix <- confusionMatrix(tuned_xgb_predictions, as.factor(test_data$explosion))
print(tuned_xgb_confusion_matrix)


### SVM

# Define the tuning grid
tune_grid <- expand.grid(
  C = c(0.1, 1, 10),
  sigma = c(0.01, 0.1, 1)
)

# Train control for cross-validation
train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

# Train the SVM model with fine-tuning
tuned_svm_model <- train(
  explosion ~ condition_pb * session3 + pb * pumps + pb + participant, 
  data = train_data, 
  method = "svmRadial", 
  trControl = train_control, 
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")  # Preprocess to center and scale the data
)

# Print the best tuning parameters
print(tuned_svm_model$bestTune)

# Predict on the test set with the tuned model
tuned_svm_predictions <- as.factor(ifelse(predict(tuned_svm_model, newdata = test_data) > 0.5, 1, 0))

# Generate the confusion matrix for the tuned model
tuned_svm_confusion_matrix <- confusionMatrix( tuned_svm_predictions, as.factor(test_data$explosion))
print(tuned_svm_confusion_matrix)


# Split the data into training and testing sets
trainIndex <- createDataPartition(final_df$explosion, p = 0.7, list = FALSE)
train_data <- final_df[trainIndex, ]
test_data <- final_df[-trainIndex, ]

# Prepare the training data
x_train <- model.matrix(explosion ~ condition_pb * session3 + condition_pb + pb * pumps+pb *participant, data = train_data)[, -1]
y_train <- train_data$explosion

# Prepare the test data
x_test <- model.matrix(explosion ~ condition_pb * session3 + condition_pb + pb * pumps+pb *participant, data = test_data)[, -1]
y_test <- test_data$explosion

# Define the grid for lambda
grid <- 10^seq(10, -10, length = 100)

# Perform Ridge Regularization (alpha = 0 for Ridge)
cv_ridge <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0, lambda = grid)

# Get the optimal lambda
opt_lambda_ridge <- cv_ridge$lambda.min
print(paste("Optimal Lambda for Ridge:", opt_lambda_ridge))

# Fit the model with the optimal lambda
ridge_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0, lambda = opt_lambda_ridge)

# Predict on the test data
pred_probs <- predict(ridge_model, s = opt_lambda_ridge, newx = x_test, type = "response")
predictions <- ifelse(pred_probs > 0.5, 1, 0)

# Calculate accuracy on the test set
accuracy <- mean(predictions == y_test)
print(paste("Test Set Accuracy:", accuracy))


train_data2 <- subset(train_data, p_burst != 0.15)
test_data2<- subset(test_data, p_burst != 0.15)

# Drop unused factor levels in the target variable
train_data2$p_burst <- droplevels(train_data2$p_burst)
test_data2$p_burst <- droplevels(test_data2$p_burst)

# Define the control for cross-validation
train.control = trainControl(method = "cv", number = 10)

# Train the multinomial logistic regression model using k-fold cross-validation
model_kfold <- train(
  p_burst ~ explosion + condition + pumps_n * explosion * participant + cash, 
  data = train_data2, 
  method = "multinom",
  trControl = train.control
)

# Print cross-validation results
print(model_kfold)

# Predict on the test set
test_predictions <- predict(model_kfold, newdata = test_data2)

# Generate the confusion matrix for the test set
final_cm <- confusionMatrix(test_predictions, test_data2$p_burst)
print("Confusion Matrix for Test Set:")
print(final_cm)


### Neural Networks

set.seed(123)
# Remove rows where p_burst is 0.15
train_data2 <- subset(train_data, p_burst != 0.15)
test_data2 <- subset(test_data, p_burst != 0.15)

# Drop unused factor levels in the target variable
train_data2$p_burst <- droplevels(train_data2$p_burst)
test_data2$p_burst <- droplevels(test_data2$p_burst)

# Subset the data to include only the specified variables
vars_to_include <- c("p_burst", "explosion", "condition", "pumps_n", "participant", "cash")
train_data2 <- train_data2[, vars_to_include]
test_data2 <- test_data2[, vars_to_include]

# One-hot encode factor variables, excluding the target variable
dummy_train <- dummyVars(" ~ .", data = train_data2[, -which(names(train_data2) == "p_burst")])
train_data_encoded <- data.frame(predict(dummy_train, newdata = train_data2[, -which(names(train_data2) == "p_burst")]))
train_data_encoded$p_burst <- train_data2$p_burst

dummy_test <- dummyVars(" ~ .", data = test_data2[, -which(names(test_data2) == "p_burst")])
test_data_encoded <- data.frame(predict(dummy_test, newdata = test_data2[, -which(names(test_data2) == "p_burst")]))
test_data_encoded$p_burst <- test_data2$p_burst

# Normalize the data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization to the relevant columns in train and test sets
train_data_normalized <- as.data.frame(lapply(train_data_encoded[, sapply(train_data_encoded, is.numeric)], normalize))
test_data_normalized <- as.data.frame(lapply(test_data_encoded[, sapply(test_data_encoded, is.numeric)], normalize))

# Add the target variable back to the normalized data
train_data_normalized$p_burst <- as.vector(train_data_encoded$p_burst)
test_data_normalized$p_burst <- as.vector(test_data_encoded$p_burst)

train_data_normalized <- as.data.frame(lapply(train_data_normalized, as.numeric))
test_data_normalized <- as.data.frame(lapply(test_data_normalized, as.numeric))

# Define the predictors
predictors <- setdiff(names(train_data_normalized), "p_burst")

# Define advanced grid search parameters using a named list
layer_options <- list(
  option1 = c(2, 3, 2),     
  option2 = c(3, 3, 3), 
  option3 = c(4, 5, 4),     
  option4 = c(5, 6, 5),     
  option5 = c(6, 7, 6),     
  option6 = c(2, 5, 5, 2),  
  option7 = c(3, 4, 4, 3),   
  option8 = c(3, 4, 5, 4, 3)
)
results <- data.frame(Layers = character(), Kappa = numeric())
confusion_matrices <- list()

for (option in names(layer_options)) {
  layers <- layer_options[[option]]
  
  # Convert layer configuration to string for storing in results
  layers_str <- paste(layers, collapse = ", ")
  
  # Train the neural network model with the current configuration
  nn_model <- neuralnet(as.vector(p_burst) ~ ., data = train_data_normalized, threshold = 0.001, hidden = neurons, linear.output = FALSE, stepmax =100000, act.fct = "logistic")
  
  # Predict on the test set
  nn_predictions <- predict(nn_model, test_data_normalized)
  
  # Convert probabilities to binary outcomes
  nn_predicted_classes <- ifelse(nn_predictions > 0.25, 1, 0)
  
  # Ensure the factor levels match
  nn_predicted_classes <- factor(nn_predicted_classes, levels = c(0, 1))
  actual_classes <- factor(test_data_normalized$p_burst, labels = c(0, 1))
  
  # Generate the confusion matrix
  nn_confusion_matrix <- confusionMatrix(nn_predicted_classes[1:962], actual_classes)
  
  # Calculate accuracy
  kappa <- nn_confusion_matrix$overall['Kappa']
  
  # Store the results
  results <- rbind(results, data.frame(Layers = layers_str, Kappa = kappa))
  confusion_matrices[[layers_str]] <- nn_confusion_matrix
  
  print('one closer to the end!')
  print(layers_str)
}


# Print the results
print(results)

# Identify the best configuration
best_config <- results[which.max(results$Kappa),]
print(best_config)

# Retrieve and print the confusion matrix for the best model
best_layers_str <- as.character(best_config$Layers)
best_nn_confusion_matrix <- confusion_matrices[[best_layers_str]]

# Print the confusion matrix
print(best_nn_confusion_matrix)


### random forest       

# Fine-tuning the Random Forest model using cross-validation
# Define the tuning grid
tune_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))

# Train control for cross-validation
train_control <- trainControl(method = "cv", number = 5)

# Train the Random Forest model with fine-tuning
tuned_rf_model <- train(p_burst ~ explosion + condition + pumps_n * explosion * participant + cash, 
                        data = train_data2, method = "rf", trControl = train_control, tuneGrid = tune_grid)

# Print the best tuning parameters
print(tuned_rf_model$bestTune)

# Predict on the test set with the tuned model
tuned_rf_predictions <- predict(tuned_rf_model, newdata = test_data2)

# Generate the confusion matrix for the tuned model
tuned_rf_confusion_matrix <- confusionMatrix(tuned_rf_predictions, factor(test_data2$p_burst, labels=c(0.1,0.2)))
print(tuned_rf_confusion_matrix)


### SVM

# Define the tuning grid
tune_grid <- expand.grid(
  C = c(0.1, 1, 10),
  sigma = c(0.01, 0.1, 1)
)

# Train control for cross-validation
train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

# Train the SVM model with fine-tuning
tuned_svm_model <- train(
  p_burst ~ explosion + condition + pumps_n * explosion * participant + cash, 
  data = train_data2, 
  method = "svmRadial", 
  trControl = train_control, 
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")  # Preprocess to center and scale the data
)

# Print the best tuning parameters
print(tuned_svm_model$bestTune)

# Predict on the test set with the tuned model
tuned_svm_predictions <- predict(tuned_svm_model, newdata = test_data2)

# Generate the confusion matrix for the tuned model
tuned_svm_confusion_matrix <- confusionMatrix( tuned_svm_predictions, as.factor(test_data2$p_burst))
print(tuned_svm_confusion_matrix)



### XGBoost

# Define the tuning grid
tune_grid <- expand.grid(
  nrounds = c(50, 150),
  eta = c(0.01, 0.2),
  max_depth = c(3, 9),
  gamma = c(0, 5),
  colsample_bytree = c(0.5, 1.0),
  min_child_weight = c(1, 10),
  subsample = c(0.8, 1.0)
)

# Train control for cross-validation
train_control <- trainControl(method = "cv", number = 3, verboseIter = TRUE)

# Train the XGBoost model with fine-tuning
tuned_xgb_model <- train(
  p_burst ~ explosion + condition + pumps_n * explosion * participant + cash, 
  data = train_data2, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = tune_grid
)

# Print the best tuning parameters
print(tuned_xgb_model$bestTune)

# Predict on the test set with the tuned model
tuned_xgb_predictions <- predict(tuned_xgb_model, newdata = test_data2)

# Generate the confusion matrix for the tuned model
tuned_xgb_confusion_matrix <- confusionMatrix(tuned_xgb_predictions, test_data2$p_burst)
print(tuned_xgb_confusion_matrix)

# Predict on the test set with the tuned model
tuned_xgb_predictions <- predict(tuned_xgb_model, newdata = test_data2)

# Generate the confusion matrix for the tuned model
tuned_xgb_confusion_matrix <- confusionMatrix(tuned_xgb_predictions, test_data2$p_burst)
print(tuned_xgb_confusion_matrix)

# Convert the trained model to xgb.Booster object for further analysis
xgb_model <- xgb.Booster(model = tuned_xgb_model$finalModel)

# Plot the tree structure (example: plotting the first tree)
xgb.plot.tree(model = xgb_model, trees = 0)

# Get feature importance
importance_matrix <- xgb.importance(model = xgb_model)

# Print feature importance
print(importance_matrix)

# Plot the feature importance
xgb.plot.importance(importance_matrix, top_n = 10)

# Highlight the most important variable
top_10_importance_df <- head(importance_matrix, 10)
top_10_importance_df$Feature <- factor(top_10_importance_df$Feature, levels = top_10_importance_df$Feature[order(top_10_importance_df$Gain, decreasing = TRUE)])
top_10_importance_df$color <- c("red", rep("black", nrow(top_10_importance_df) - 1))

# Custom plot with ggplot2
ggplot(top_10_importance_df, aes(x = Feature, y = Gain, fill = color)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_identity() +
  xlab("Features") +
  ylab("Importance (Gain)")


final_df$pumps_n <- (final_df$pumps - min(final_df$pumps)) / (max(final_df$pumps) - min(final_df$pumps))

# Split the data into training and testing sets
trainIndex <- createDataPartition(final_df$condition, p = 0.7, list = TRUE)
train_data <- final_df[trainIndex$Resample1, ]
test_data <- final_df[-trainIndex$Resample1, ]

# Define 10-fold cross-validation setup
folds <- createFolds(train_data$condition, k = 10)
cv_results <- vector("list", length = 10)

# Perform 10-fold cross-validation
for(i in seq_along(folds)) {
  fold_train <- train_data[-folds[[i]], ]
  fold_test <- train_data[folds[[i]], ]
  
  # Fit the multinomial logistic regression model
  fold_model <- multinom(condition ~ explosion + pb + pumps_n + participant + cash, data = fold_train)
  
  # Predict on the validation set
  fold_predictions <- predict(fold_model, newdata = fold_test, type = "class")
  
  # Calculate accuracy for this fold
  fold_accuracy <- mean(fold_predictions == fold_test$condition)
  cv_results[[i]] <- fold_accuracy
}

# Calculate average accuracy over all folds
mean_accuracy <- mean(unlist(cv_results))
print(paste("Cross-Validated Accuracy:", mean_accuracy))

# Fit final model on full training data
final_multinom_model <- multinom(condition ~ pb + pumps_n * explosion * participant + cash, data = train_data)

# Predict on the test data
final_predictions <- predict(final_multinom_model, newdata = test_data, type = "class")

# Calculate accuracy on the test set
final_accuracy <- mean(final_predictions == test_data$condition)
print(paste("Test Set Accuracy:", final_accuracy))


### Stage 2: SVM, RF, ANN, XGBoost
'
We first load the data preprocessed from python. The python preprocessing was done useing the following code:
'
# we first load the data preprocessed from python. The python preprocessing was done useing the following code:
# d = df.groupby(['participant', 'condition', 'p_burst']).agg({
#         'total': 'max',
#         'pumps': ['sum', 'mean'],
#         'explosion': 'sum',
#         'session': 'max',
#         'block': 'max'
#     }).reset_index()
# d.columns = [idx[0] if idx[1]=='' else '_'.join(idx) for idx in d.columns.to_flat_index()]
# d.rename(columns={'session_max':'session', 'block_max':'block', 'total_max':'total'}, inplace=True)
# n_participant = len(d.participant.unique())
# factorize_condition = d["condition"].factorize()
# d['a'] = factorize_condition[0]
# a_labels = factorize_condition[1]
# d['pC'] = 1 - d['explosion_sum'] / 30 # proportion of cashed
# d['pE'] = d['explosion_sum'] / d['pumps_sum'] # proportion of explosions
# d['mNOP'] = standardize(d['pumps_mean'])
# d['NOE'] = standardize(d['explosion_sum'])
# d['TR'] = standardize(d['total'])
# d['PB'] = standardize(d['p_burst'])
# d['pC_s'] = standardize(d['pC'])
# d['C'] = 30 - d['explosion_sum']

completed_data <- read.csv('data_preprocessed2.csv', header = T)
completed_data$block <- NULL
completed_data$session <- as.factor(completed_data$session)
completed_data$participant <- as.factor(completed_data$participant)
completed_data$condition <- as.factor(completed_data$condition)

participant_dummies <- model.matrix(~ participant - 1, data = completed_data)
session_dummies <- model.matrix(~ session - 1, data = completed_data)

participant_dummies <- data.frame(participant_dummies)
session_dummies <- data.frame(session_dummies)
session3 <- session_dummies$session3

completed_data <- completed_data %>% select(-participant, -session)
completed_data <- cbind(completed_data, participant_dummies)
completed_data <- cbind(completed_data, session3)

str(completed_data)


#### ANN

# Data Preparation
# Convert all factors to dummy variables
numeric_data <- model.matrix(~ . - 1, data = completed_data)  # Create dummy variables for factors
TR_vector <- completed_data$TR  # Save the target variable

# Scale the numeric data, excluding the target variable 'TR'
completed_data_scaled <- scale(numeric_data)
completed_data_scaled <- data.frame(completed_data_scaled, TR = TR_vector)

# Split data into training and test sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(completed_data_scaled$TR, p = 0.8, list = FALSE)
train_data <- completed_data_scaled[train_index, ]
test_data <- completed_data_scaled[-train_index, ]

# Define predictors and the formula
completed_data_scaled <- completed_data_scaled[, !names(completed_data_scaled) %in% "TR.1"]
predictors <- setdiff(names(completed_data_scaled), "TR")

# Cross-validation settings
folds <- createFolds(train_data$TR, k = 10)

# Remove the TR.1 column from train_data and test_data
train_data <- train_data[, !names(train_data) %in% "TR.1"]
test_data <- test_data[, !names(test_data) %in% "TR.1"]


# Explore different configurations of neural networks
best_rmse <- Inf
best_config <- NULL
configurations <- expand.grid(layers = 1:10, neurons = c(3, 4, 6, 8, 10))

# Loop over configurations
for(config in seq(nrow(configurations))) {
  layer_config <- rep(configurations[config, "neurons"], configurations[config, "layers"])
  
  cv_rmse <- vector("numeric", length = 10)
  i <- 1
  for(fold_index in folds) {
    fold_train <- train_data[fold_index, ]
    fold_test <- train_data[-fold_index, ]
    
    nn_model <- neuralnet(TR ~ ., data = fold_train, hidden = layer_config, linear.output = TRUE, rep = 2)
    
    predictions <- compute(nn_model, fold_test[, predictors])
    actual_values <- fold_test$TR
    predicted_values <- predictions$net.result
    cv_rmse[i] <- sqrt(mean((predicted_values - actual_values)^2))
    
    i <- i + 1
  }
  
  average_rmse <- mean(cv_rmse)
  print('mse is: ')
  print(average_rmse)
  print('current configuration is: ')
  print(layer_config)
  if(average_rmse < best_rmse) {
    best_rmse <- average_rmse
    best_config <- layer_config
  }
}

# Output the best RMSE and configuration found
print(paste("Best RMSE from CV:", best_rmse))
print(paste("Best configuration:", toString(best_config)))

# You might need to run the following code twice to avoid receiving any errors
final_model <- neuralnet(TR~., data = train_data, hidden = best_config, linear.output = TRUE, rep = 1)

# Evaluate on the test set
final_predictions <- compute(final_model, test_data[, predictors])
final_actual_values <- test_data$TR
final_predicted_values <- final_predictions$net.result
final_test_rmse <- sqrt(mean((final_predicted_values - final_actual_values)^2))

# Print final RMSE on test set
print(paste("Final RMSE on test set:", final_test_rmse))


### SVM

# Data Preparation
# Convert all factors to dummy variables
numeric_data <- model.matrix(~ . - 1, data = completed_data)
TR_vector <- completed_data$TR

# Scale the numeric data, excluding the target variable 'TR'
completed_data_scaled <- scale(numeric_data)
completed_data_scaled <- data.frame(completed_data_scaled, TR = TR_vector)

# Split data into training and test sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(completed_data_scaled$TR, p = 0.8, list = FALSE)
train_data <- completed_data_scaled[train_index, ]
test_data <- completed_data_scaled[-train_index, ]

# Define training control for SVM
train_control <- trainControl(
  method = "cv", 
  number = 10,
  savePredictions = "final", 
  search = "grid" 
)

# Define the grid for parameter tuning
svm_grid <- expand.grid(
  sigma = 2^(-1:1),  # Range for the sigma parameter
  C = 2^(2:4)        # Range for the cost parameter
)

# Train the SVM model
svm_model <- train(
  TR ~ .,
  data = train_data,
  method = "svmRadial",  # Radial basis function kernel
  trControl = train_control,
  tuneGrid = svm_grid,
  preProcess = "scale"  # Ensure data is scaled
)

# Print out the best model's parameters and RMSE
print(svm_model$bestTune)
print(min(svm_model$results$RMSE))

# Evaluate the model on the test set
test_predictions <- predict(svm_model, newdata = test_data)
test_rmse <- sqrt(mean((test_predictions - test_data$TR)^2))

# Print final RMSE on the test set
print(paste("Final RMSE on test set:", test_rmse))


#### Linear Regression

# Remove the TR.1 column from train_data and test_data if such a column exists
train_data <- train_data[, !names(train_data) %in% "TR.1"]
train_data <- train_data[, !names(train_data) %in% "pumps_sum"]
test_data <- test_data[, !names(test_data) %in% "pumps_sum"]
test_data <- test_data[, !names(test_data) %in% "TR.1"]

# Define training control using 10-fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 10,  # Number of folds in cross-validation
  savePredictions = "final",
  search = "grid"
)

# Define the grid for parameter tuning
tuning_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),  # Mix between Ridge (0) and Lasso (1)
  lambda = 10^seq(-3, 1, length = 10)  # Regularization parameter
)

# Train the Elastic Net model with cross-validation
elastic_net_model <- train(
  TR ~ . + PB*pumps_mean,
  data = train_data,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = tuning_grid,
  metric = "RMSE",
  maximize = FALSE  # Set to FALSE because lower RMSE is better
)

# Output the best model's parameters and associated RMSE
print(elastic_net_model$bestTune)
print(min(elastic_net_model$results$RMSE))

# Evaluate the model on the test set
test_predictions <- predict(elastic_net_model, newdata = test_data)
test_rmse <- sqrt(mean((test_predictions - test_data$TR)^2))

# Print final RMSE on test set
print(paste("Final RMSE on test set:", test_rmse))



# Get variable importance
variable_importance <- varImp(elastic_net_model)
print(variable_importance)

# Extract the coefficients for the best model
best_lambda <- elastic_net_model$bestTune$lambda
best_alpha <- elastic_net_model$bestTune$alpha
final_model <- glmnet(
  as.matrix(train_data[, !names(train_data) %in% c("TR", "TR_shifted", "TR_bc")]), 
  train_data$TR_bc, 
  alpha = best_alpha, 
  lambda = best_lambda
)

# Get the coefficients
coefficients <- coef(final_model)
print(coefficients)

# Combine importance and coefficients
importance_and_direction <- data.frame(
  Variable = rownames(coefficients),
  Coefficient = as.numeric(coefficients)
)

# Remove the intercept for clarity
importance_and_direction <- importance_and_direction[importance_and_direction$Variable != "(Intercept)", ]

# Sort by importance (absolute value of coefficient)
importance_and_direction <- importance_and_direction[order(abs(importance_and_direction$Coefficient), decreasing = TRUE), ]

# Transform the coefficients to reduce the impact of very large values
importance_and_direction$TransformedCoefficient <- log(abs(importance_and_direction$Coefficient) + 1) * sign(importance_and_direction$Coefficient)

# Display the top 6 variables
top_vars <- head(importance_and_direction, 6)
top_vars$color <- ifelse(top_vars$Coefficient > 0, "blue", "red")

# Plot the top 6 variables with their transformed coefficients
ggplot(top_vars, aes(x = reorder(Variable, TransformedCoefficient), y = TransformedCoefficient, fill = color)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_identity() +
  xlab("Variables") +
  ylab("Transformed Coefficient") +
  ggtitle("Top 6 Variable Importance and Direction in Elastic Net Model") +
  theme(axis.text.x = element_text(angle = 0, hjust = 1))



### Random Forest

# Define training control using 10-fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 10,  # Number of folds in cross-validation
  savePredictions = "final",
  search = "grid"
)

# Define a grid for mtry tuning
mtry_grid <- round(seq(sqrt(ncol(train_data)), ncol(train_data) / 3, length.out = 5))

# Define other parameters for tuning
ntree_grid <- c(100, 200, 500)  # Number of trees
nodesize_grid <- c(1, 5, 10)  # Minimum size of terminal nodes
maxnodes_grid <- c(10, 20, 30)  # Maximum number of terminal nodes

# Initialize a data frame to store results
results <- data.frame()

# Manual tuning
for (ntree in ntree_grid) {
  for (nodesize in nodesize_grid) {
    for (maxnodes in maxnodes_grid) {
      
      # Define the grid for mtry tuning
      tuning_grid <- expand.grid(
        mtry = mtry_grid
      )
      
      # Train the Random Forest model with cross-validation
      rf_model <- train(
        TR ~ .,
        data = train_data,
        method = "rf",
        trControl = train_control,
        tuneGrid = tuning_grid,
        metric = "RMSE",
        maximize = FALSE,
        ntree = ntree,
        nodesize = nodesize,
        maxnodes = maxnodes
      )
      
      # Collect the results
      best_result <- rf_model$results[which.min(rf_model$results$RMSE),]
      best_result$ntree <- ntree
      best_result$nodesize <- nodesize
      best_result$maxnodes <- maxnodes
      
      results <- rbind(results, best_result)
      
    }
  }
}

# Find the best combination
best_combination <- results[which.min(results$RMSE),]
print(best_combination)

# Train the final model with the best combination
final_rf_model <- randomForest(
  TR ~ .,
  data = train_data,
  mtry = best_combination$mtry,
  ntree = best_combination$ntree,
  nodesize = best_combination$nodesize,
  maxnodes = best_combination$maxnodes
)

# Evaluate the final model on the test set
test_predictions <- predict(final_rf_model, newdata = test_data)
test_rmse <- sqrt(mean((test_predictions - test_data$TR)^2))

# Print final RMSE on test set
print(paste("Final RMSE on test set:", test_rmse))

# Variable importance plot
varImpPlot(final_rf_model)

# Print the model summary
print(summary(final_rf_model))


#### XGBoost

numeric_data <- model.matrix(~ . - 1, data = completed_data)
TR_vector <- completed_data$TR

# Combine numeric data with target variable
completed_data_scaled <- data.frame(numeric_data, TR = TR_vector)

# Split data into training and test sets
set.seed(123)
train_index <- createDataPartition(completed_data_scaled$TR, p = 0.8, list = FALSE)
train_data <- completed_data_scaled[train_index, ]
test_data <- completed_data_scaled[-train_index, ]

# Remove the TR.1 column from train_data and test_data
train_data <- train_data[, !names(train_data) %in% "TR.1"]
test_data <- test_data[, !names(test_data) %in% "TR.1"]

# Define training control and tuning grid for XGBoost
train_control <- trainControl(
  method = "cv",
  number = 2,  # Number of folds in cross-validation
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",  # Saves all resampling results
  savePredictions = "final",
  classProbs = FALSE,  # Set TRUE for classification problems
)

# Define the grid for parameter tuning
tuning_grid <- expand.grid(
  nrounds = c(50, 150),  # Number of boosting rounds
  eta = c(0.01, 0.1),  # Learning rate
  max_depth = c(3, 9),  # Max depth of trees
  gamma = c(0, 0.1, 0.2),  # Minimum loss reduction required for further partition
  colsample_bytree = c(0.5, 0.7, 1),  # Subsample ratio of columns when constructing each tree
  min_child_weight = c(1, 5, 10),  # Minimum sum of instance weight (hessian) needed in a child
  subsample = c(0.5, 1)  # Subsample percentage of the training instance
)

# Train the XGBoost model
xgb_model <- train(
  TR ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tuning_grid,
  metric = "RMSE",
  maximize = FALSE
)

# Output the best model's parameters and associated RMSE
print(xgb_model$bestTune)
print(min(xgb_model$results$RMSE))

# Evaluate the model on the test set
test_predictions <- predict(xgb_model, newdata = test_data)
test_rmse <- sqrt(mean((test_predictions - test_data$TR)^2))

# Print final RMSE on test set
print(paste("Final RMSE on test set:", test_rmse))
