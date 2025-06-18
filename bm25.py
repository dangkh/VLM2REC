import os
import gzip
import json
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")

import numpy as np
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="BM25 Retrieval and NDCG Evaluation")
    
    parser.add_argument("--gen_des_path", type=str, default="./data/amazon_descriptions_full.csv",
                        help="Path to generated description CSV file")
    
    parser.add_argument("--meta_path", type=str, default="./data/meta_Industrial_and_Scientific.json.gz",
                        help="Path to compressed metadata JSON file")
    
    parser.add_argument("--topK", type=int, default=5,
                        help="Top-K items to retrieve for evaluation (e.g., NDCG@K)")
    
    parser.add_argument("--typeG", type=int, default=1,
                        help="Type use original or generated description")
    
    return parser.parse_args()
def dcg(relevance_list):
	return sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(relevance_list))

def ndcg_at_k(pred_asins, relevant_asins, k):
	relevance_list = [1 if asin in relevant_asins else 0 for asin in pred_asins[:k]]
	ideal_relevance = sorted(relevance_list, reverse=True)
	
	dcg_val = dcg(relevance_list)
	idcg_val = dcg(ideal_relevance)
	return dcg_val / idcg_val if idcg_val > 0 else 0.0


if __name__ == "__main__":
	args = parse_args()
	pathGenDes = args.gen_des_path
	pathMeta = args.meta_path
	topK = args.topK
	typeG = args.typeG

	dfGenDes = pd.read_csv(pathGenDes)
	dfGenDes = dfGenDes.dropna(subset=["description"]).reset_index(drop=True)
	valid_asins = set(dfGenDes["asin"])
	print(f" item contain images and description: {len(valid_asins)}")

	data = []
	with gzip.open(pathMeta, "r") as f:
		for line in f:
			data.append(json.loads(line))

	df = pd.DataFrame(data)
	df_meta = df[["asin", "title", "feature", "brand", "category", "description"]].copy()
	df_meta_filtered = (df_meta[df_meta["asin"].isin(valid_asins)].drop_duplicates(subset="asin").reset_index(drop=True))
	print(f" item contain images and description and metadata: {len(df_meta_filtered)}")

	corpus = []
	item_id_mapping = []
	print(f" Building corpus......")

	for _, row in df_meta_filtered.iterrows():
		tit = "".join(row["title"]).lower()
		des = "".join(row["brand"]).lower()
		cat = " ".join( row["category"]).lower()
		full_text = tit + des + cat
		tokens = word_tokenize(full_text)
		tokens = [token for token in tokens if re.fullmatch(r"[a-zA-Z0-9\-]+", token)]
		corpus.append(tokens)
		item_id_mapping.append(row["asin"])

	print(f" size of corpus: {len(corpus)}")

	bm25 = BM25Okapi(corpus)

	listScore = []
	dfTest = df_meta_filtered
	if typeG == 1:
		dfTest = dfGenDes
	for _, row in tqdm(dfTest.iterrows(), total=len(dfTest), desc="Evaluating NDCG"):
		query = row['description']
		if typeG == 0:
			if len(query) > 0:
				query = query[0]
			else:
				query = ""
		query_tokens = word_tokenize(query)
		scores = bm25.get_scores(query_tokens)
		top_indices = np.argsort(scores)[::-1][:topK]
		top_asins = [item_id_mapping[i] for i in top_indices]
		relevant_asins = {row['asin']}
		ndcg_score = ndcg_at_k(top_asins, relevant_asins, topK)
		listScore.append(ndcg_score)


	print(f"NDCG@{topK}: {np.mean(listScore):.4f}")
