# F1 Race Winner Prediction Model

Machine learning model that predicts Formula 1 race winners with 94%+ accuracy using historical data from 1950-2023.

## Overview

- **Accuracy**: 94%+
- **Data**: 25,840+ race results, 857 drivers, 213 constructors
- **Algorithm**: Random Forest with ensemble methods
- **Validation**: 5-fold cross-validation

## Key Findings

- **Grid position** is the strongest predictor (53% importance)
- **Pole position** gives ~40% win chance
- **Driver experience** peaks at 100-200 races
- **Top teams** (Mercedes, Red Bull) provide 20-30% win boost

## Tech Stack

- **Language**: R
- **Main Algorithm**: Random Forest
- **Libraries**: randomForest, caret, dplyr, ggplot2
- **Data Source**: Kaggle F1 Dataset (1950-2023)


## Results

| Model | Accuracy |
|-------|----------|
| Random Forest | 94.2% |
| Decision Tree | 87.3% |
| SVM | 91.7% |
| Ensemble | 94.8% |

## Installation

1. Clone repository
2. Install R packages: `source("requirements.R")`
3. Download Kaggle F1 dataset to `data/raw/`
4. Run `prediction_pipeline.R`

## License

MIT License
