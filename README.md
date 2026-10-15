# expense-classifier-ml

Sorts expense descriptions into categories such as food, transport, bills, and subscriptions with Naive Bayes.

## Data

Deterministic synthetic data is generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, a template-grouped holdout kept each base description out of training. On 120 expenses across six categories, accuracy was 0.4167 and macro F1 was 0.3611. This baseline generalizes poorly across description templates.

## Limitations

The synthetic descriptions are limited and do not capture the variety of real transaction records. A practical model would need appropriately sourced, labeled descriptions and validation on a separate sample.
