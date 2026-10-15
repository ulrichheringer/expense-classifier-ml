# expense-classifier-ml

Categoriza descrições de despesas em grupos como alimentação, transporte, contas e assinaturas com Naive Bayes.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: holdout agrupado por descrição-base (templates de validação não aparecem no treino), 120 despesas em seis categorias — acurácia **0,4167**, F1 macro **0,3611**. Este baseline mostra generalização fraca entre templates; amplie e valide com descrições históricas rotuladas antes de uso real.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
