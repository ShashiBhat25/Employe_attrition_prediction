"""
pipeline.py  -  Employee Attrition: data prep, training, fairness, risk tiers.
Usage: python src/pipeline.py data/WA_Fn-UseC_-HR-Employee-Attrition.csv
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from fairlearn.metrics import MetricFrame, demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, selection_rate

OUT   = "artifacts"
PLOTS = "artifacts/plots"
SEED  = 42
DROP  = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
TARGET = "Attrition"
ACTIONS = {"Critical":"Immediate manager conversation","High":"Enrol in engagement programme",
           "Medium":"Monitor closely","Low":"No action required"}

# ── Data ──────────────────────────────────────────────────────────────────────
def prepare(path):
    df = pd.read_csv(path)
    df.drop(columns=[c for c in DROP if c in df.columns], inplace=True)
    df[TARGET] = df[TARGET].map({"Yes":1,"No":0})
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes":1,"No":0})
    df = pd.get_dummies(df, columns=df.select_dtypes("object").columns.tolist())
    df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)
    sat = [c for c in ["JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","WorkLifeBalance"] if c in df.columns]
    if sat: df["SatisfactionAvg"] = df[sat].mean(axis=1)
    if {"MonthlyIncome","TotalWorkingYears"}.issubset(df.columns):
        df["IncomePerYear"] = df["MonthlyIncome"] / df["TotalWorkingYears"].replace(0,np.nan).fillna(1)
    if {"YearsInCurrentRole","TotalWorkingYears"}.issubset(df.columns):
        df["Stagnation"] = df["YearsInCurrentRole"] / df["TotalWorkingYears"].replace(0,np.nan).fillna(1)
    X = df.drop(columns=[TARGET]).fillna(df.drop(columns=[TARGET]).median(numeric_only=True))
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X.columns)
    X_te = pd.DataFrame(scaler.transform(X_te), columns=X.columns)
    os.makedirs(OUT, exist_ok=True)
    joblib.dump(scaler, f"{OUT}/scaler.pkl")
    json.dump(X.columns.tolist(), open(f"{OUT}/feature_names.json","w"))
    return X_tr, X_te, y_tr.reset_index(drop=True), y_te.reset_index(drop=True)

# ── Train ─────────────────────────────────────────────────────────────────────
def train(X_tr, X_te, y_tr, y_te):
    X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        "RandomForest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost":            XGBClassifier(n_estimators=200, scale_pos_weight=5, learning_rate=0.05, max_depth=4, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    }
    results, best_name, best_recall = {}, None, -1
    for name, m in models.items():
        m.fit(X_sm, y_sm)
        yp = m.predict(X_te); yprob = m.predict_proba(X_te)[:,1]
        met = dict(ROC_AUC=roc_auc_score(y_te,yprob), Recall=recall_score(y_te,yp),
                   F1=f1_score(y_te,yp), Precision=precision_score(y_te,yp))
        results[name] = {"model":m, "metrics":met}
        joblib.dump(m, f"{OUT}/{name}.pkl")
        if met["Recall"] > best_recall:
            best_recall, best_name = met["Recall"], name
    joblib.dump(results[best_name]["model"], f"{OUT}/best_model.pkl")
    json.dump({"best_model":best_name,"metrics":{n:{k:float(v) for k,v in r["metrics"].items()} for n,r in results.items()}},
              open(f"{OUT}/metrics.json","w"), indent=2)

    # Feature importance from best model
    model = results[best_name]["model"]
    feat_names = X_tr.columns.tolist()
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"Feature":feat_names,"Importance":model.feature_importances_})
    else:
        imp = pd.DataFrame({"Feature":feat_names,"Importance":np.abs(model.coef_[0])})
    imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    imp.to_csv(f"{OUT}/feature_importance.csv", index=False)

    os.makedirs(PLOTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9,5))
    top = imp.head(15)
    ax.barh(top["Feature"][::-1], top["Importance"][::-1], color="#c0392b")
    ax.set_xlabel("Importance"); ax.set_title("Top 15 Attrition Drivers")
    plt.tight_layout(); fig.savefig(f"{PLOTS}/feature_importance.png", dpi=130); plt.close()

    print(f"Best model: {best_name}  Recall={best_recall:.3f}")
    return results, best_name

# ── Fairness ──────────────────────────────────────────────────────────────────
def fairness(y_te, y_pred, raw_test):
    os.makedirs(PLOTS, exist_ok=True)
    report, sf_map = {}, {}
    if "Gender" in raw_test.columns:
        sf_map["Gender"] = raw_test["Gender"].astype(str)
    if "Age" in raw_test.columns:
        sf_map["AgeGroup"] = pd.cut(raw_test["Age"], bins=[0,25,35,45,55,100],
                                    labels=["<25","25-34","35-44","45-54","55+"]).astype(str)
    for attr, sf in sf_map.items():
        dpd = demographic_parity_difference(y_te, y_pred, sensitive_features=sf)
        dpr = demographic_parity_ratio(y_te, y_pred, sensitive_features=sf)
        eod = equalized_odds_difference(y_te, y_pred, sensitive_features=sf)
        mf  = MetricFrame(metrics={"selection_rate":selection_rate,"recall":recall_score,"precision":precision_score},
                          y_true=y_te, y_pred=y_pred, sensitive_features=sf)
        grp = mf.by_group.reset_index()
        grp.columns = [attr] + list(grp.columns[1:])
        report[attr] = {"demographic_parity_difference":float(dpd),"demographic_parity_ratio":float(dpr),
                        "equalized_odds_difference":float(eod),"by_group":grp.to_dict("records")}
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(grp[attr].astype(str), grp["selection_rate"].astype(float), color="#2980b9")
        ax.set_title(f"Predicted Attrition Rate by {attr}"); ax.set_ylim(0,1)
        plt.tight_layout(); fig.savefig(f"{PLOTS}/fairness_{attr}.png", dpi=130); plt.close()
    json.dump(report, open(f"{OUT}/fairness_report.json","w"), indent=2, default=str)
    return report

# ── Risk tiers ────────────────────────────────────────────────────────────────
def risk_table(raw_test, y_prob):
    df = raw_test.copy().reset_index(drop=True)
    df["AttritionProbability"] = np.round(y_prob, 4)
    df["RiskTier"] = df["AttritionProbability"].apply(
        lambda p: "Critical" if p>=0.70 else "High" if p>=0.50 else "Medium" if p>=0.30 else "Low")
    df["RecommendedAction"] = df["RiskTier"].map(ACTIONS)
    df.sort_values("AttritionProbability", ascending=False).reset_index(drop=True).to_csv(f"{OUT}/risk_table.csv", index=False)
    return df

# ── Main ──────────────────────────────────────────────────────────────────────
def run(path):
    print("Loading & preparing data...")
    raw = pd.read_csv(path)
    X_tr, X_te, y_tr, y_te = prepare(path)
    print("Training models...")
    results, best = train(X_tr, X_te, y_tr, y_te)
    model = results[best]["model"]
    y_pred = model.predict(X_te); y_prob = model.predict_proba(X_te)[:,1]
    _, raw_te = train_test_split(raw, test_size=0.2, random_state=SEED, stratify=raw[TARGET])
    raw_te = raw_te.reset_index(drop=True)
    print("Fairness analysis...")
    fairness(y_te, y_pred, raw_te)
    print("Building risk table...")
    risk_table(raw_te, y_prob)
    pred_df = raw_te.copy()
    pred_df["AttritionProbability"] = np.round(y_prob,4)
    pred_df["Predicted"] = y_pred; pred_df["Actual"] = y_te.values
    pred_df.to_parquet(f"{OUT}/test_predictions.parquet", index=False)
    print(f"Done. Artifacts saved to: {os.path.abspath(OUT)}")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv)>1 else "data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
