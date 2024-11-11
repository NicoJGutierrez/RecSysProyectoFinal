## Fuente:
https://github.com/barnap/group-recommenders-offline-evaluation

# Documentación de Uso

Este proyecto implementa varias estrategias de agregación y evaluadores de métricas para recomendaciones grupales. A continuación se describe cómo utilizar las diferentes partes del código.

## 0. Requisitos

Instala las dependencias necesarias utilizando pip:

pip install pandas numpy scipy

## 1. Estrategias de Agregación

Las estrategias de agregación se encuentran en el archivo `aggregators.py`. Estas estrategias se utilizan para generar recomendaciones grupales basadas en diferentes métodos de agregación.

### Ejemplo de Uso:

```python
from aggregation_strategies.aggregators import AggregationStrategy

# Datos de ejemplo
group_ratings = pd.DataFrame({
    'user': [1, 1, 2, 2],
    'item': [101, 102, 101, 103],
    'predicted_rating': [4.5, 3.0, 4.0, 5.0]
})
recommendations_number = 2

# Obtener una estrategia de agregación
strategy = AggregationStrategy.getAggregator("ADD")
recommendations = strategy.generate_group_recommendations_for_group(group_ratings, recommendations_number)
print(recommendations)

```

## 2. Funciones Utilitarias
Las funciones utilitarias se encuentran en el archivo utility_functions.py. Estas funciones se utilizan para preprocesar datos y generar recomendaciones para todos los grupos.

### Ejemplo de Uso:

```python
from utility_functions import generate_group_recommendations_forall_groups

# Datos de ejemplo
test_df = pd.DataFrame({
    'user': [1, 1, 2, 2],
    'item': [101, 102, 101, 103],
    'predicted_rating': [4.5, 3.0, 4.0, 5.0]
})
group_composition = {
    'group1': {
        'group_size': 2,
        'group_similarity': 0.8,
        'group_members': [1, 2]
    }
}
recommendations_number = 2

# Generar recomendaciones para todos los grupos
group_recommendations = generate_group_recommendations_forall_groups(test_df, group_composition, recommendations_number)
print(group_recommendations)
```

## 3. Evaluadores de Métricas
Los evaluadores de métricas se encuentran en el archivo metric_evaluators.py. Estas clases se utilizan para evaluar la calidad de las recomendaciones grupales.

### Ejemplo de Uso:
```python
from evaluation_metrics.metric_evaluators import MetricEvaluator

# Datos de ejemplo
group_ground_truth = pd.DataFrame({
    'user': [1, 1, 2, 2],
    'item': [101, 102, 101, 103],
    'rating': [5, 4, 4, 5]
})
group_recommendation = [101, 103]
group_members = [1, 2]
propensity_per_item = pd.DataFrame({
    'item': [101, 102, 103],
    'propensity_score': [0.8, 0.9, 0.7]
}).set_index('item')
per_user_propensity_normalization_term = pd.Series([0.5, 0.6], index=[1, 2])
evaluation_ground_truth = "GROUP_CHOICES"
binarize_feedback_positive_threshold = 4
binarize_feedback = True
feedback_polarity_debiasing = 0.0

# Obtener un evaluador de métricas
evaluator = MetricEvaluator.getMetricEvaluator("NDCG")
evaluation_results = evaluator.evaluateGroupRecommendation(
    group_ground_truth,
    group_recommendation,
    group_members,
    propensity_per_item,
    per_user_propensity_normalization_term,
    evaluation_ground_truth,
    binarize_feedback_positive_threshold,
    binarize_feedback,
    feedback_polarity_debiasing
)
print(evaluation_results)
```