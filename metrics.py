import ast
import numpy as np
import pandas as pd

def parse_group_data(groups_df):
    parsed_data = []
    
    for _, row in groups_df.head(300).iterrows():
        group_members = ast.literal_eval(row['group_members'])
        recommendations = ast.literal_eval(row['recommendations'])
        parsed_data.append((group_members, recommendations))
    
    print(f"\nTotal groups parsed: {len(parsed_data)}")
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
    
    # Keep track of all recommended items
    all_recommended_items = set()
    total_recommendations = 0
    
    for i, (group_members, recommendations) in enumerate(parsed_data):
        print(f"\rProcessing Group {i+1}/{total_groups}", end="")
        recommended_items = recommendations[:k]
        
        # Add to global set of recommended items
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
    
    # Calculate global diversity
    total_diversity = len(all_recommended_items) / total_recommendations
    
    avg_metrics = {
        "average_precision": total_precision / total_groups,
        "average_recall": total_recall / total_groups,
        "average_ndcg": total_ndcg / total_groups,
        "average_relevance": total_relevance / total_groups,
        "average_diversity": total_diversity,  # This is now global diversity
    }
    
    return avg_metrics

train_df = pd.read_csv('train.csv')
groups_df = pd.read_csv('RecSysProyectoFinal/group_recommendations.csv')
parsed_data = parse_group_data(groups_df)
metrics = calculate_group_metrics(parsed_data, train_df, k=5)
print(f"Precision: {metrics['average_precision']:.6f}")
print(f"Recall: {metrics['average_recall']:.6f}")
print(f"nDCG: {metrics['average_ndcg']:.6f}")
print(f"Relevance Score: {metrics['average_relevance']:.2f}")
print(f"Diversity: {metrics['average_diversity']:.2f}")

train_df = pd.read_csv('train.csv')
groups_df = pd.read_csv('RecSysProyectoFinal/svd_group_recommendations.csv')
parsed_data = parse_group_data(groups_df)
metrics = calculate_group_metrics(parsed_data, train_df, k=5)
print(f"Precision: {metrics['average_precision']:.6f}")
print(f"Recall: {metrics['average_recall']:.6f}")
print(f"nDCG: {metrics['average_ndcg']:.6f}")
print(f"Relevance Score: {metrics['average_relevance']:.2f}")
print(f"Diversity: {metrics['average_diversity']:.2f}")

train_df = pd.read_csv('train.csv')
groups_df = pd.read_csv('RecSysProyectoFinal/mostpopular_group_recommendations.csv')
parsed_data = parse_group_data(groups_df)
metrics = calculate_group_metrics(parsed_data, train_df, k=5)
print(f"Precision: {metrics['average_precision']:.6f}")
print(f"Recall: {metrics['average_recall']:.6f}")
print(f"nDCG: {metrics['average_ndcg']:.6f}")
print(f"Relevance Score: {metrics['average_relevance']:.2f}")
print(f"Diversity: {metrics['average_diversity']:.2f}")

