import ast
import numpy as np
import pandas as pd

def parse_group_data(groups_df):
    parsed_data = []
    
    for _, row in groups_df.head(300).iterrows():
        group_members = ast.literal_eval(row['group_members'])
        recommendations = ast.literal_eval(row['recommendations'])
        parsed_data.append((group_members, recommendations))
    
    print(f"\nTotal: {len(parsed_data)}")
    return parsed_data

def calculate_relevance(train_df, user_id, item_id):
    rating = train_df[
        (train_df['Username'] == user_id) & 
        (train_df['BGGId'] == item_id)
    ]['Rating']
    
    return rating.iloc[0] if not rating.empty else 0

def calculate_group_metrics(parsed_data, train_df, k=5, threshold=5):
    total_precision = 0
    total_recall = 0
    total_ndcg = 0
    total_relevance = 0
    total_diversity = 0
    total_groups = len(parsed_data)
    total_hits = 0
    total_possible_hits = 0
    
    all_recommended_items = set()
    total_recommendations = 0
    
    total_novelty = 0
    item_popularity = {}
    total_items = len(train_df['BGGId'].unique())
    for item_id in train_df['BGGId'].unique():
        item_popularity[item_id] = len(train_df[train_df['BGGId'] == item_id]) / len(train_df)
    
    total_fairness = 0
    total_serendipity = 0
    
    for i, (group_members, recommendations) in enumerate(parsed_data):
        print(f"\rProcesando {i+1}/{total_groups}", end="")
        recommended_items = recommendations[:k]
        
        all_recommended_items.update(recommended_items)
        total_recommendations += len(recommended_items)
        
        group_relevance = []
        group_hits = 0
        for item_id in recommended_items:
            member_ratings = []
            for user_id in group_members:
                rating = calculate_relevance(train_df, user_id, item_id)
                if rating > 0:
                    member_ratings.append(rating)
                    group_hits += 1
                total_possible_hits += 1
            avg_rating = np.mean(member_ratings) if member_ratings else 0
            group_relevance.append(avg_rating)
        
        total_hits += group_hits
        
        binary_relevance = [1 if rel >= threshold else 0 for rel in group_relevance]
        
        total_relevant = 0
        unique_items = train_df['BGGId'].unique()
        for item_id in unique_items:
            member_ratings = []
            for user_id in group_members:
                rating = calculate_relevance(train_df, user_id, item_id)
                if rating > 0:
                    member_ratings.append(rating)
            avg_rating = np.mean(member_ratings) if member_ratings else 0
            if avg_rating >= threshold:
                total_relevant += 1
        
        precision = sum(binary_relevance) / k if k > 0 else 0
        recall = sum(binary_relevance) / total_relevant if total_relevant > 0 else 0
        
        dcg = sum((2 ** rel - 1) / np.log2(idx + 2) 
                  for idx, rel in enumerate(group_relevance))
        ideal_relevance = sorted(group_relevance, reverse=True)
        idcg = sum((2 ** rel - 1) / np.log2(idx + 2) 
                   for idx, rel in enumerate(ideal_relevance))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        total_precision += precision
        total_recall += recall
        total_ndcg += ndcg
        total_relevance += np.mean(group_relevance) if group_relevance else 0
        
        
        group_novelty = 0
        for item_id in recommended_items:
            pop = item_popularity.get(item_id, 0)
            if pop > 0:
                group_novelty += -np.log2(pop)
        group_novelty /= len(recommended_items) if recommended_items else 1
        total_novelty += group_novelty
        
        # Calculate fairness (variance of satisfaction among group members)
        member_satisfaction = []
        for user_id in group_members:
            user_satisfaction = 0
            for item_id in recommended_items:
                rating = calculate_relevance(train_df, user_id, item_id)
                user_satisfaction += rating
            user_satisfaction /= len(recommended_items) if recommended_items else 1
            member_satisfaction.append(user_satisfaction)
        
        # Lower variance means higher fairness
        fairness = 1 / (1 + np.var(member_satisfaction)) if member_satisfaction else 0
        total_fairness += fairness
        
        # Calculate serendipity (unexpected relevant recommendations)
        serendipity = 0
        for item_id in recommended_items:
            # Item is considered serendipitous if it's both relevant and novel
            relevance = np.mean([calculate_relevance(train_df, user_id, item_id) 
                               for user_id in group_members])
            popularity = item_popularity.get(item_id, 0)
            unexpectedness = -np.log2(popularity) if popularity > 0 else 0
            serendipity += relevance * unexpectedness
        serendipity /= len(recommended_items) if recommended_items else 1
        total_serendipity += serendipity
    
    total_diversity = len(all_recommended_items) / total_recommendations
    
    avg_metrics = {
        "average_precision": total_precision / total_groups,
        "average_recall": total_recall / total_groups,
        "average_ndcg": total_ndcg / total_groups,
        "average_relevance": total_relevance / total_groups,
        "average_diversity": total_diversity,
        "average_novelty": total_novelty / total_groups,
        "average_fairness": total_fairness / total_groups,
        "average_serendipity": total_serendipity / total_groups
    }
    return avg_metrics

train_df = pd.read_csv('train.csv')
# df = pd.read_csv('RecSysProyectoFinal/random_group_recommendations.csv', sep=';')
# df.to_csv('RecSysProyectoFinal/random_group_recommendations.csv', index=False, quoting=1)
groups_df = pd.read_csv('RecSysProyectoFinal/random_group_recommendations.csv')
parsed_data = parse_group_data(groups_df)
metrics = calculate_group_metrics(parsed_data, train_df, k=5)
print("Random LightFM")
print(f"Precision: {metrics['average_precision']:.6f}")
print(f"Recall: {metrics['average_recall']:.6f}")
print(f"nDCG: {metrics['average_ndcg']:.6f}")
print(f"Relevance Score: {metrics['average_relevance']:.2f}")
print(f"Diversity: {metrics['average_diversity']:.2f}")
print(f"Novelty: {metrics['average_novelty']:.2f}")
print(f"Fairness: {metrics['average_fairness']:.2f}")
print(f"Serendipity: {metrics['average_serendipity']:.2f}")


# df = pd.read_csv('RecSysProyectoFinal/similar_group_recommendations.csv', sep=';')
# df.to_csv('RecSysProyectoFinal/similar_group_recommendations.csv', index=False, quoting=1)
groups_df = pd.read_csv('RecSysProyectoFinal/similar_group_recommendations.csv')
parsed_data = parse_group_data(groups_df)
metrics = calculate_group_metrics(parsed_data, train_df, k=5)
print("Similar LightFM")
print(f"Precision: {metrics['average_precision']:.6f}")
print(f"Recall: {metrics['average_recall']:.6f}")
print(f"nDCG: {metrics['average_ndcg']:.6f}")
print(f"Relevance Score: {metrics['average_relevance']:.2f}")
print(f"Diversity: {metrics['average_diversity']:.2f}")
print(f"Novelty: {metrics['average_novelty']:.2f}")
print(f"Fairness: {metrics['average_fairness']:.2f}")
print(f"Serendipity: {metrics['average_serendipity']:.2f}")

train_df = pd.read_csv('train.csv')
# df = pd.read_csv('RecSysProyectoFinal/svd_random_group_recommendations.csv', sep=';')
# df.to_csv('RecSysProyectoFinal/svd_random_group_recommendations.csv', index=False, quoting=1)
groups_df = pd.read_csv('RecSysProyectoFinal/svd_random_group_recommendations.csv')
parsed_data = parse_group_data(groups_df)
metrics = calculate_group_metrics(parsed_data, train_df, k=5)
print("Random SVD")
print(f"Precision: {metrics['average_precision']:.6f}")
print(f"Recall: {metrics['average_recall']:.6f}")
print(f"nDCG: {metrics['average_ndcg']:.6f}")
print(f"Relevance Score: {metrics['average_relevance']:.2f}")
print(f"Diversity: {metrics['average_diversity']:.2f}")
print(f"Novelty: {metrics['average_novelty']:.2f}")
print(f"Fairness: {metrics['average_fairness']:.2f}")
print(f"Serendipity: {metrics['average_serendipity']:.2f}")

# df = pd.read_csv('RecSysProyectoFinal/svd_similar_group_recommendations.csv', sep=';')
# df.to_csv('RecSysProyectoFinal/svd_similar_group_recommendations.csv', index=False, quoting=1)
groups_df = pd.read_csv('RecSysProyectoFinal/svd_similar_group_recommendations.csv')
parsed_data = parse_group_data(groups_df)
metrics = calculate_group_metrics(parsed_data, train_df, k=5)
print("Similar SVD")
print(f"Precision: {metrics['average_precision']:.6f}")
print(f"Recall: {metrics['average_recall']:.6f}")
print(f"nDCG: {metrics['average_ndcg']:.6f}")
print(f"Relevance Score: {metrics['average_relevance']:.2f}")
print(f"Diversity: {metrics['average_diversity']:.2f}")
print(f"Novelty: {metrics['average_novelty']:.2f}")
print(f"Fairness: {metrics['average_fairness']:.2f}")
print(f"Serendipity: {metrics['average_serendipity']:.2f}")

train_df = pd.read_csv('train.csv')
# df = pd.read_csv('RecSysProyectoFinal/mostpopular_similar_group_recommendations.csv', sep=';')
# df.to_csv('RecSysProyectoFinal/mostpopular_similar_group_recommendations.csv', index=False, quoting=1)
groups_df = pd.read_csv('RecSysProyectoFinal/mostpopular_similar_group_recommendations.csv')
parsed_data = parse_group_data(groups_df)
metrics = calculate_group_metrics(parsed_data, train_df, k=5)
print("Similar Most Popular")
print(f"Precision: {metrics['average_precision']:.6f}")
print(f"Recall: {metrics['average_recall']:.6f}")
print(f"nDCG: {metrics['average_ndcg']:.6f}")
print(f"Relevance Score: {metrics['average_relevance']:.2f}")
print(f"Diversity: {metrics['average_diversity']:.2f}")
print(f"Novelty: {metrics['average_novelty']:.2f}")
print(f"Fairness: {metrics['average_fairness']:.2f}")
print(f"Serendipity: {metrics['average_serendipity']:.2f}")

# df = pd.read_csv('RecSysProyectoFinal/mostpopular_random_group_recommendations.csv', sep=';')
# df.to_csv('RecSysProyectoFinal/mostpopular_random_group_recommendations.csv', index=False, quoting=1)
groups_df = pd.read_csv('RecSysProyectoFinal/mostpopular_random_group_recommendations.csv')
parsed_data = parse_group_data(groups_df)
metrics = calculate_group_metrics(parsed_data, train_df, k=5)
print("Random Most Popular")
print(f"Precision: {metrics['average_precision']:.6f}")
print(f"Recall: {metrics['average_recall']:.6f}")
print(f"nDCG: {metrics['average_ndcg']:.6f}")
print(f"Relevance Score: {metrics['average_relevance']:.2f}")
print(f"Diversity: {metrics['average_diversity']:.2f}")
print(f"Novelty: {metrics['average_novelty']:.2f}")
print(f"Fairness: {metrics['average_fairness']:.2f}")
print(f"Serendipity: {metrics['average_serendipity']:.2f}")