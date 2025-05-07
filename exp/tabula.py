import os
import time
import pickle
import pandas as pd
import torch
from pathlib import Path
import sys
sys.path.insert(0, '/home/yjung/SynPriv')
from src.tabula import Tabula
import warnings
warnings.filterwarnings("ignore")

exp_name = "tabula_retail"
sample_sizes = [10000] # [10, 100, 500, 1000, 5000, 10000]
data_path = Path("../data/sampled")
base_save_dir = Path(f"../results/{exp_name}")
categorical_columns = ['fkey', 'date', 'CustomerID', 'Country', 'Description']
n_samples = 1000

def run_tabula(sample_size):
    sub_exp_name = f"train{sample_size}"
    save_dir = base_save_dir / sub_exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_path / f"retail_sample_{sample_size}.csv"
    data = pd.read_csv(csv_path)

    print(f"{sample_size} 학습 시작")
    model = Tabula(
        llm="distilgpt2",
        experiment_dir=str(save_dir),
        batch_size=32,
        epochs=10,
        categorical_columns=categorical_columns
    )

    start_time = time.time()
    model.fit(data)
    train_time = (time.time() - start_time) / 60
    print(f"{sample_size} 학습 소요 시간: {train_time:.2f}분")
    model.save(str(save_dir))
    
    print(f"{sample_size} 생성 시작")
    start_time2 = time.time()
    synthetic_data = model.sample(n_samples=n_samples)
    test_time = (time.time() - start_time2) / 60
    print(f"{sample_size} 생성 소요 시간: {test_time:.2f}분")

    syn_csv = save_dir / f"syn_tabula_{n_samples}.csv"
    synthetic_data.to_csv(syn_csv, index=False)
    print(f"{sample_size} 완료: {syn_csv}")

for size in sample_sizes:
    run_tabula(size)