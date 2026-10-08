# Freight Rate Prediction

See `freight-rate-ml-assessment.pdf` for the assessment.

## Setup
pip install -r requirements.txt

Put the four data files in a folder named `data/`:
train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv

## Run
python training.py --data-dir data --out-dir .
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

Use `--no-cv` to skip the cross-validation (it takes a few minutes).

## Outputs
- validation_predictions.csv: predictions for the 12,000 validation loads
- data/december_chart_inputs.csv: December predictions
- scorer_results/candidate_december.png: chart made by score.py

## Approach
Time-based validation (train on past months, test on the next month),
gradient boosting on log(rate), corrupted prices removed from training.