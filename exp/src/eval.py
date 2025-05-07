import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors as NN
from tqdm import tqdm
from pathlib import Path
from sdmetrics.single_table import CategoricalCAP, DisclosureProtectionEstimate

def preprocess(dfs, num_cols, cat_cols):
    test_df = dfs['Test'].copy()
    test_df[num_cols] = test_df[num_cols].apply(pd.to_numeric, errors='coerce')
    scaler = StandardScaler()
    scaler.fit(test_df[num_cols])
    cat_df = dfs['Test'][cat_cols].astype(str)
    ohe_cols = pd.get_dummies(cat_df, columns=cat_cols)
    ohe_all_cols = ohe_cols.columns.tolist()

    dfs_transformed = {}
    for name, df in dfs.items():
        df = df.copy()
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        scaled_num = scaler.transform(df[num_cols])
        cat_part = df[cat_cols].astype(str)
        ohe_cat = pd.get_dummies(cat_part, columns=cat_cols)
        ohe_cat = ohe_cat.reindex(columns=ohe_all_cols, fill_value=0)
        ohe_cat = ohe_cat.astype(float).values
        all_parts = np.concatenate([scaled_num, ohe_cat], axis=1)
        dfs_transformed[name] = all_parts

    return dfs_transformed, scaler

def dcr(real, syn):
    nn = NN(n_neighbors=1, algorithm='auto', metric='euclidean')
    nn.fit(real)
    all_dists = []
    batch_size = 2000

    for i in tqdm(range(0, len(syn), batch_size), desc="Computing DCR"):
        batch = syn[i:i + batch_size]
        dists, _ = nn.kneighbors(batch)
        all_dists.append(dists)

    all_dists = np.vstack(all_dists).flatten()
    print(f"DCR Median: {np.median(all_dists):.4f}")
    print(f"최소 거리: {np.min(all_dists):.4f}, 최대 거리: {np.max(all_dists):.4f}")
    return all_dists

def nndr(real, syn):
    nn = NN(n_neighbors=2)
    nn.fit(real)
    ratios = []
    batch_size = 2000

    for i in tqdm(range(0, len(syn), batch_size), desc="Computing NNDR"):
        batch = syn[i:i + batch_size]
        dists, _ = nn.kneighbors(batch)
        first = dists[:, 0]
        second = dists[:, 1]
        ratio = first / np.maximum(second, 1e-6)
        ratios.append(ratio)

    all_ratios = np.concatenate(ratios)
    nndr_value = np.median(all_ratios)
    print(f"NNDR (median dist_1 / dist_2): {nndr_value:.4f}")
    return nndr_value

def plot_dcr(dcr_dict):
    plt.figure()
    for label, distances in dcr_dict.items():
        plt.hist(distances, bins=50, alpha=0.4, edgecolor='black', label=f"{label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_categorical_cap(real_table, dfs, key_fields=['Country'], sensitive_fields=['Description', 'Quantity']):
    results = []
    for sensitive in sensitive_fields:
        for name, synthetic_table in dfs.items():
            if name == 'Train':
                continue
            try:
                score = CategoricalCAP.compute(
                    real_data=real_table,
                    synthetic_data=synthetic_table,
                    key_fields=key_fields,
                    sensitive_fields=[sensitive]
                )
                results.append({
                    'dataset': name,
                    'sensitive_field': sensitive,
                    'CategoricalCAP': round(score, 4)
                })
            except Exception:
                results.append({
                    'dataset': name,
                    'sensitive_field': sensitive,
                    'CategoricalCAP': 'error'
                })
    return pd.DataFrame(results)


# def get_common_rare_combos(df1, df2, top_n=10):
#     df1 = df1.copy()
#     df2 = df2.copy()
#     df1['GENDER'] = df1['GENDER'].astype(str)
#     df2['GENDER'] = df2['GENDER'].astype(str)
#     df1['BTH_YR'] = df1['BTH_YR'].astype(int)
#     df2['BTH_YR'] = df2['BTH_YR'].astype(int)

#     count1 = df1.groupby(['BTH_YR', 'GENDER']).size().reset_index(name='count1')
#     count2 = df2.groupby(['BTH_YR', 'GENDER']).size().reset_index(name='count2')

#     merged = pd.merge(count1, count2, on=['BTH_YR', 'GENDER'])
#     merged['min_count'] = merged[['count1', 'count2']].min(axis=1)

#     rare_combos = merged.nsmallest(top_n, 'min_count')[['BTH_YR', 'GENDER']]
#     return rare_combos

# def get_fkeys_by_combos(df, rare_combos):
#     df = df.copy()
#     df['GENDER'] = df['GENDER'].astype(str)
#     df['BTH_YR'] = df['BTH_YR'].astype(int)
#     merged = df.merge(rare_combos, on=['BTH_YR', 'GENDER'])
#     return merged['key'].unique()

# def filter_df_by_fkeys(df, fkeys):
#     return df[df['key'].isin(fkeys)]

# def flatten_by_fkey(vectors, df, fkey_col='key'):
#     groups = df.groupby(fkey_col).indices
#     flattened = [vectors[idxs].flatten() for idxs in groups.values()]
#     return np.vstack(flattened)

# def dcr_by_fkey_group(name1, name2, alls, dfs_transformed, num_cols):
#     rare_combos = get_common_rare_combos(alls[name1], alls[name2])
#     fkeys1 = get_fkeys_by_combos(alls[name1], rare_combos)
#     fkeys2 = get_fkeys_by_combos(alls[name2], rare_combos)
#     df1 = filter_df_by_fkeys(alls[name1], fkeys1)
#     df2 = filter_df_by_fkeys(alls[name2], fkeys2)

#     vec1 = flatten_by_fkey(df1, dfs_transformed[name1][df1.index])
#     vec2 = flatten_by_fkey(df2, dfs_transformed[name2][df2.index])
#     return dcr(vec1, vec2)

# def seq_repr(df, fkey_col, sort_col, cat_cols, num_cols, round_digits=4):
#     group_sorted = df.sort_values(sort_col)
#     cat_part = group_sorted[cat_cols].astype(str)
#     cat_seq = cat_part.apply(lambda row: '|'.join(row), axis=1).tolist()

#     num_part = group_sorted[num_cols].round(round_digits)
#     num_seq = num_part.apply(lambda row: '|'.join(map(str, row)), axis=1).tolist()

#     full_seq = [c + '||' + n for c, n in zip(cat_seq, num_seq)]
#     seq_str = '----'.join(full_seq)
#     return seq_str


# def find_copied_fkeys(train_df, syn_df, fkey_col='fkey', sort_col='YM', cat_cols=None, num_cols=None, num_tol=0.01):
#     train_repr = seq_repr(train_df, fkey_col, sort_col, cat_cols, num_cols, num_tol)
#     syn_repr = seq_repr(syn_df, fkey_col, sort_col, cat_cols, num_cols, num_tol)

#     syn_set = set(syn_repr)
#     copied_fkeys = [k for k in train_repr if k in syn_set]

#     return {
#         'copied_fkeys': copied_fkeys,
#         'num_train_fkeys': len(train_repr),
#         'num_syn_fkeys': len(syn_repr),
#         'num_copied': len(copied_fkeys),
#         'copied_ratio': round(len(copied_fkeys) / len(train_repr), 4)
#     }