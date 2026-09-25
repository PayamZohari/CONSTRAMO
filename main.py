import argparse
from sklearn.model_selection import train_test_split
from train import *
import pandas as pd
import numpy as np
import torch
import snf
import os
import time

from oversmoothing_analysis import *

# ── seeds ────────────────────────────────────────────────────────────────────
SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]

# ── contrastive hyper-parameters ────────────────────────────────────────────
CONTRASTIVE_WEIGHT = 0.3   # lambda
TEMPERATURE        = 0.5   # tau
MAX_PAIRS          = 256
PATIENCE           = 30
N_EPOCHS_GCN       = 200

# ── argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--cancer_type', '-t', type=str,
    choices=['lgg','ucec','stad','sarc','coadread','cesc','hnsc','brca'],
    default='lgg')
parser.add_argument('--no_contrastive', action='store_true',
    help='Run original MO-GCAN baseline (no contrastive loss)')
parser.add_argument('--contrastive_weight', type=float, default=CONTRASTIVE_WEIGHT)
parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
args = parser.parse_args()

cancer_type        = args.cancer_type
use_contrastive    = not args.no_contrastive
contrastive_weight = args.contrastive_weight
seeds              = args.seeds

# ── data loading ─────────────────────────────────────────────────────────────
files = os.listdir('data/' + cancer_type)
for file in files:
    if file[0].isdigit():
        os.rename('data/' + cancer_type + '/' + file,
                  'data/' + cancer_type + '/' + file[file.find('_') + 1:])

print("loading cna, met, mrna, and rppa data...")
cna_df    = pd.read_csv('data/' + cancer_type + '/cna_data.csv',     header=0, index_col=None)
met_df    = pd.read_csv('data/' + cancer_type + '/met_data.csv',     header=0, index_col=None)
mrna_df   = pd.read_csv('data/' + cancer_type + '/mrna_data.csv',    header=0, index_col=None)
rppa_df   = pd.read_csv('data/' + cancer_type + '/rppa_data.csv',    header=0, index_col=None)
labels_df = pd.read_csv('data/' + cancer_type + '/subtype_data.csv', header=0, index_col=None)
print("data loading is finished!\n")

# ── output setup ─────────────────────────────────────────────────────────────
new_path = os.getcwd() + '/result/' + cancer_type
if not os.path.exists(new_path):
    os.makedirs(new_path)

mode_tag = 'contrastive' if use_contrastive else 'baseline'
log_f = open('result/' + cancer_type + '/' + cancer_type + '_' + mode_tag + '.log',
             'w', encoding='utf-8')

print(f"Mode: {'MO-GCAN + Cross-Modal Relational Contrastive' if use_contrastive else 'MO-GCAN Baseline'}")
print(f"Seeds: {seeds}")
print(f"Mode: {'MO-GCAN + Cross-Modal Relational Contrastive' if use_contrastive else 'MO-GCAN Baseline'}", file=log_f)
print(f"Seeds: {seeds}", file=log_f)

# ── helpers ───────────────────────────────────────────────────────────────────
def report_to_dict(labels_t, pred_t):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    y_true   = labels_t.numpy()
    y_pred   = pred_t.numpy()
    classes  = sorted(np.unique(y_true))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    acc      = accuracy_score(y_true, y_pred)
    result   = {'accuracy': acc, 'classes': {}}
    for idx, c in enumerate(classes):
        result['classes'][c] = {'precision': p[idx], 'recall': r[idx],
                                'f1': f[idx], 'support': s[idx]}
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro',    zero_division=0)
    p_wt,  r_wt,  f_wt,  _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    result['macro']    = {'precision': p_mac, 'recall': r_mac, 'f1': f_mac}
    result['weighted'] = {'precision': p_wt,  'recall': r_wt,  'f1': f_wt}
    return result


def print_averaged_report(reports, label, log_f):
    if not reports:
        return
    accs        = [r['accuracy'] for r in reports]
    all_classes = sorted(set(c for r in reports for c in r['classes']))

    header = f"\n{'='*60}\nAVERAGED results ({len(reports)} seeds): {label}\n{'='*60}"
    print(header);  print(header, file=log_f)

    col = f"{'':>12}  {'precision':>10}  {'recall':>10}  {'f1-score':>10}  {'support':>10}"
    print(col);  print(col, file=log_f)

    for c in all_classes:
        ps = [r['classes'][c]['precision'] for r in reports if c in r['classes']]
        rs = [r['classes'][c]['recall']    for r in reports if c in r['classes']]
        fs = [r['classes'][c]['f1']        for r in reports if c in r['classes']]
        ss = [r['classes'][c]['support']   for r in reports if c in r['classes']]
        row = f"{str(c):>12}  {np.mean(ps):10.4f}  {np.mean(rs):10.4f}  {np.mean(fs):10.4f}  {np.mean(ss):10.1f}"
        print(row);  print(row, file=log_f)

    print('');  print('', file=log_f)
    for avg_key in ['macro', 'weighted']:
        ps  = [r[avg_key]['precision'] for r in reports]
        rs  = [r[avg_key]['recall']    for r in reports]
        fs  = [r[avg_key]['f1']        for r in reports]
        row = f"{avg_key+' avg':>12}  {np.mean(ps):10.4f}  {np.mean(rs):10.4f}  {np.mean(fs):10.4f}  {'':>10}"
        print(row);  print(row, file=log_f)

    acc_row = f"\n  accuracy (mean +/- std): {np.mean(accs):.4f} +/- {np.std(accs):.4f}"
    print(acc_row);  print(acc_row, file=log_f)


# ── multi-seed loop ───────────────────────────────────────────────────────────
start_time = time.time()
all_reports_all      = []
all_reports_selected = []

for seed_idx, seed in enumerate(seeds):
    print(f"\n{'='*60}\n  SEED {seed}  ({seed_idx+1}/{len(seeds)})\n{'='*60}")
    print(f"\n--- SEED {seed} ---", file=log_f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, x_test, y_train, y_test = train_test_split(
        cna_df, labels_df['class'],
        random_state=seed, test_size=0.25, stratify=labels_df['class'])

    # affinity networks
    print("making the affinity networks for training data...")
    train_affinity = snf.compute.make_affinity(
        cna_df.iloc[x_train.index].iloc[:, 1:].values.astype(np.float64),
        met_df.iloc[x_train.index].iloc[:, 1:].values.astype(np.float64),
        mrna_df.iloc[x_train.index].iloc[:, 1:].values.astype(np.float64),
        rppa_df.iloc[x_train.index].iloc[:, 1:].values.astype(np.float64),
        metric='sqeuclidean', K=20, mu=0.5)
    print("affinity network for training data is finished!\n.")

    cna_data, met_data, mrna_data, rppa_data, labels = load_to_tensor(
        cna_df, met_df, mrna_df, rppa_df, labels_df)
    train_lap = load_adj_to_tensor(train_affinity)

    # ── train GCN models ──────────────────────────────────────────────────────
    if use_contrastive:
        print("Start to train omics-specific models WITH contrastive loss...")
        gcn_models, gcn_train_accs = train_omics_models_with_contrastive(
            features_list=[cna_data[x_train.index], met_data[x_train.index],
                           mrna_data[x_train.index], rppa_data[x_train.index]],
            adj_list=train_lap,
            labels=labels[x_train.index],
            n_hid=100,
            n_epochs=N_EPOCHS_GCN,
            lr=0.001,
            weight_decay=0.01,
            dropout=0.5,
            contrastive_weight=contrastive_weight,
            temperature=TEMPERATURE,
            patience=PATIENCE,
            max_pairs=MAX_PAIRS,
        )
        cna_model, met_model, mrna_model, rppa_model = gcn_models
        cna_train_acc, met_train_acc, mrna_train_acc, rppa_train_acc = gcn_train_accs
    else:
        print("Start to train omics-specific models...")
        cna_model,  cna_train_acc  = train_model('GCN', 100, cna_data[x_train.index],  train_lap[0], labels[x_train.index])
        met_model,  met_train_acc  = train_model('GCN', 100, met_data[x_train.index],  train_lap[1], labels[x_train.index])
        mrna_model, mrna_train_acc = train_model('GCN', 100, mrna_data[x_train.index], train_lap[2], labels[x_train.index])
        rppa_model, rppa_train_acc = train_model('GCN', 100, rppa_data[x_train.index], train_lap[3], labels[x_train.index])
    print("The training of omics-specific models is finished!\n")

    # SNF fusion
    print("Making fused networks by SNF...")
    train_fused_net = torch.tensor(
        make_laplacian(snf.snf(train_affinity, K=20)),
        dtype=torch.float, device=torch.device('cpu'))
    print("Fused networks are finished!\n")

    # embeddings
    print("Retrived embeddings from omics-specifc models...")

    cna_embedding = normalize(cna_model.forward_1(cna_data[x_train.index], train_lap[0]))
    met_embedding = normalize(met_model.forward_1(met_data[x_train.index], train_lap[1]))
    mrna_embedding = normalize(mrna_model.forward_1(mrna_data[x_train.index], train_lap[2]))
    rppa_embedding = normalize(rppa_model.forward_1(rppa_data[x_train.index], train_lap[3]))

    # --------------------------------------------------
    # Oversmoothing diagnostics
    # --------------------------------------------------

    for name, emb in zip(
            ["cna", "met", "mrna", "rppa"],
            [cna_embedding, met_embedding, mrna_embedding, rppa_embedding]
    ):
        print(
            f"Average cosine similarity for {name}:",
            average_cosine_similarity(emb)
        )

    for name, emb in zip(
            ["cna", "met", "mrna", "rppa"],
            [cna_embedding, met_embedding, mrna_embedding, rppa_embedding]
    ):
        print(
            f"Average pairwise distance for {name}:",
            average_pairwise_distance(emb)
        )

    for name, emb in zip(
            ["cna", "met", "mrna", "rppa"],
            [cna_embedding, met_embedding, mrna_embedding, rppa_embedding]
    ):
        print(
            f"Embedding rank for {name}:",
            embedding_rank(emb)
        )

    for name, emb in zip(
            ["cna", "met", "mrna", "rppa"],
            [cna_embedding, met_embedding, mrna_embedding, rppa_embedding]
    ):
        within, between, gap = class_separation_score(
            emb,
            labels[x_train.index]
        )

        print(
            f"{name.upper()} "
            f"Within={within:.4f} "
            f"Between={between:.4f} "
            f"Gap={gap:.4f}"
        )

    hidden_embeddings = torch.cat(
        (
            cna_embedding,
            met_embedding,
            mrna_embedding,
            rppa_embedding
        ),
        dim=1
    )

    within, between, gap = class_separation_score(
        hidden_embeddings,
        labels[x_train.index]
    )

    print(
        f"FUSED "
        f"Within={within:.4f} "
        f"Between={between:.4f} "
        f"Gap={gap:.4f}"
    )

    hidden_embeddings = torch.cat(
        (cna_embedding, met_embedding, mrna_embedding, rppa_embedding), dim=1)
    print("Embeddings from omics-specifc models are concatenated!\n")

    # final GAT model
    print("Start to train the final model for all the omics...")
    final_model, _ = train_model('GAT', 100, hidden_embeddings,
                                 train_fused_net, labels[x_train.index])
    print('The training of final model for all the omics is finished!')

    # test inference
    test_affinity = snf.compute.make_affinity(
        cna_df.iloc[x_test.index].iloc[:, 1:].values.astype(np.float64),
        met_df.iloc[x_test.index].iloc[:, 1:].values.astype(np.float64),
        mrna_df.iloc[x_test.index].iloc[:, 1:].values.astype(np.float64),
        rppa_df.iloc[x_test.index].iloc[:, 1:].values.astype(np.float64),
        metric='sqeuclidean', K=20, mu=0.5)
    test_lap = load_adj_to_tensor(test_affinity)

    test_cna_embedding  = normalize(cna_model.forward_1(cna_data[x_test.index],  test_lap[0]))
    test_met_embedding  = normalize(met_model.forward_1(met_data[x_test.index],  test_lap[1]))
    test_mrna_embedding = normalize(mrna_model.forward_1(mrna_data[x_test.index],test_lap[2]))
    test_rppa_embedding = normalize(rppa_model.forward_1(rppa_data[x_test.index],test_lap[3]))
    test_hidden_embeddings = torch.cat(
        (test_cna_embedding, test_met_embedding,
         test_mrna_embedding, test_rppa_embedding), dim=1)
    test_fused_net = torch.tensor(
        make_laplacian(snf.snf(test_affinity, K=20)),
        dtype=torch.float, device=torch.device('cpu'))

    # evaluate
    evaluation(cna_model,  cna_data[x_test.index],  test_lap[0], cancer_type, labels[x_test.index], 'cna',  log_f)
    evaluation(met_model,  met_data[x_test.index],  test_lap[1], cancer_type, labels[x_test.index], 'met',  log_f)
    evaluation(mrna_model, mrna_data[x_test.index], test_lap[2], cancer_type, labels[x_test.index], 'mrna', log_f)
    evaluation(rppa_model, rppa_data[x_test.index], test_lap[3], cancer_type, labels[x_test.index], 'rppa', log_f)
    evaluation(final_model, test_hidden_embeddings, test_fused_net,
               cancer_type, labels[x_test.index], 'all the omics', log_f)

    final_model.eval()
    with torch.no_grad():
        out_all  = final_model(test_hidden_embeddings, test_fused_net)
        pred_all = out_all.max(1)[1].type_as(labels[x_test.index])
    all_reports_all.append(report_to_dict(labels[x_test.index], pred_all))

    # ── selected omics ────────────────────────────────────────────────────────
    acc_list  = [cna_train_acc, met_train_acc, mrna_train_acc, rppa_train_acc]
    threshold = 0.8
    indices   = [i for i, acc in enumerate(acc_list) if acc > threshold]
    while len(indices) < 2:
        threshold -= 0.05
        indices = [i for i, acc in enumerate(acc_list) if acc > threshold]

    omic_names = ['cna', 'met', 'mrna', 'rppa']
    acc_summary = ', '.join(f"{omic_names[i]}={acc_list[i]:.3f}" for i in range(4))
    print(f"  Train accs: {acc_summary}  ->  selected: {[omic_names[i] for i in indices]}")
    print(f"  Train accs: {acc_summary}  ->  selected: {[omic_names[i] for i in indices]}", file=log_f)

    selected_omics           = []
    selected_train_affinity  = []
    selected_test_affinity   = []
    selected_embeddings      = torch.empty(0)
    selected_test_embeddings = torch.empty(0)
    train_embeddings = [cna_embedding, met_embedding, mrna_embedding, rppa_embedding]
    test_embeddings  = [test_cna_embedding, test_met_embedding,
                        test_mrna_embedding, test_rppa_embedding]

    for i in indices:
        selected_omics.append(omic_names[i])
        selected_train_affinity.append(train_affinity[i])
        selected_test_affinity.append(test_affinity[i])
        selected_embeddings      = torch.cat((selected_embeddings,      train_embeddings[i]), dim=1)
        selected_test_embeddings = torch.cat((selected_test_embeddings, test_embeddings[i]),  dim=1)

    selected_train_fused_net = torch.tensor(
        make_laplacian(snf.snf(selected_train_affinity, K=20)),
        dtype=torch.float, device=torch.device('cpu'))
    selected_test_fused_net = torch.tensor(
        make_laplacian(snf.snf(selected_test_affinity, K=20)),
        dtype=torch.float, device=torch.device('cpu'))

    print(f"Start to train the final model for the selected omics: {selected_omics}")
    print(f"selected omics are : {selected_omics}", file=log_f)
    final_selected_model, _ = train_model(
        'GAT', 100, selected_embeddings, selected_train_fused_net, labels[x_train.index])
    print('The training of final model for the selected omics is finished!')
    evaluation(final_selected_model, selected_test_embeddings, selected_test_fused_net,
               cancer_type, labels[x_test.index], 'selected omics', log_f)

    final_selected_model.eval()
    with torch.no_grad():
        out_sel  = final_selected_model(selected_test_embeddings, selected_test_fused_net)
        pred_sel = out_sel.max(1)[1].type_as(labels[x_test.index])
    all_reports_selected.append(report_to_dict(labels[x_test.index], pred_sel))

    torch.save(final_selected_model.state_dict(),
               f'result/{cancer_type}/selected_omics_seed{seed}.pkl')


# ── summary ───────────────────────────────────────────────────────────────────
elapsed = time.time() - start_time
print(f"\nTraining time : {elapsed} seconds")
print(f"Training time : {elapsed} seconds", file=log_f)

print_averaged_report(all_reports_all,      'ALL OMICS',      log_f)
print_averaged_report(all_reports_selected, 'SELECTED OMICS', log_f)

log_f.close()

