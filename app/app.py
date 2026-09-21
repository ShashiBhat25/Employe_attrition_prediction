"""
app.py  -  Employee Attrition Analytics  (Flask)
Run: python app/app.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app  = Flask(__name__, template_folder="templates", static_folder="static")
A    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

TIER_CLR = {"Critical":"#c0392b","High":"#e67e22","Medium":"#f1c40f","Low":"#27ae60"}
ACTIONS  = {"Critical":"Immediate manager conversation",
            "High":"Enrol in engagement programme",
            "Medium":"Monitor closely","Low":"No action required"}

# ── helpers ───────────────────────────────────────────────────────────────────
def af(name):
    return os.path.join(A, name)

def load_all():
    pred  = pd.read_parquet(af("test_predictions.parquet"))
    risk  = pd.read_csv(af("risk_table.csv"))
    met   = json.load(open(af("metrics.json")))
    fair  = json.load(open(af("fairness_report.json")))
    # Cast Arrow/object string columns to plain str
    for df in [pred, risk]:
        for col in df.select_dtypes(include=["object","string"]).columns:
            df[col] = df[col].astype(str)
    return pred, risk, met, fair

def fig_json(fig):
    return fig.to_json()

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def overview():
    pred, risk, met, fair = load_all()
    best = met["best_model"]
    m    = met["metrics"][best]

    # KPIs
    kpis = {
        "Employees": len(pred),
        "Actual Attrition": f"{int(pred['Actual'].sum())} ({pred['Actual'].mean():.1%})",
        "Predicted Attrition": int(pred["Predicted"].sum()),
        "Best Model AUC": f"{m['ROC_AUC']:.3f}  (Recall {m['Recall']:.3f})",
    }

    # Charts
    d = pred.groupby("Department")["Actual"].agg(["sum","count"]).reset_index()
    d["Rate"] = d["sum"] / d["count"]
    fig_dept = px.bar(d, x="Department", y="Rate", color="Rate",
        color_continuous_scale="Reds", title="Attrition Rate by Department",
        text=d["Rate"].apply(lambda x: f"{x:.0%}")).update_layout(
        coloraxis_showscale=False, height=320, margin=dict(t=40,b=20))

    r = pred.groupby("JobRole")["Actual"].agg(["sum","count"]).reset_index()
    r["Rate"] = r["sum"] / r["count"]; r = r.sort_values("Rate")
    fig_role = px.bar(r, x="Rate", y="JobRole", orientation="h", color="Rate",
        color_continuous_scale="Reds", title="Attrition Rate by Job Role").update_layout(
        coloraxis_showscale=False, height=320, margin=dict(t=40,b=20))

    # Model table
    model_rows = [{"Model":n,"AUC":f"{v['ROC_AUC']:.4f}","Recall":f"{v['Recall']:.4f}",
                   "F1":f"{v['F1']:.4f}","Precision":f"{v['Precision']:.4f}"}
                  for n,v in met["metrics"].items()]

    return render_template("overview.html", kpis=kpis,
        fig_dept=fig_json(fig_dept), fig_role=fig_json(fig_role),
        model_rows=model_rows, best=best)


@app.route("/heatmap")
def heatmap():
    pred, *_ = load_all()

    # Heatmap: fill missing combos with 0
    pivot = pred.groupby(["Department","JobRole"])["AttritionProbability"].mean()\
                .reset_index().pivot(index="Department", columns="JobRole", values="AttritionProbability")\
                .fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values.tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="Dept: %{y}<br>Role: %{x}<br>Prob: %{z:.2%}<extra></extra>",
        colorbar=dict(title="Prob")
    ))
    fig.update_layout(
        title="Mean Attrition Probability — Department x Job Role",
        height=420,
        xaxis=dict(tickangle=-30),
        margin=dict(t=50, b=120, l=160, r=20)
    )

    ot_fig = None
    if "OverTime" in pred.columns:
        ot = pred.copy()
        ot["OT"] = ot["OverTime"].map({1: "Yes", 0: "No"})
        ot_fig = fig_json(px.box(
            ot, x="OT", y="AttritionProbability", color="OT",
            color_discrete_map={"Yes": "#c0392b", "No": "#27ae60"},
            title="Attrition Probability by OverTime",
            labels={"OT": "OverTime", "AttritionProbability": "Attrition Probability"}
        ).update_layout(showlegend=False, height=340))

    # Bar chart: avg attrition prob by department
    dept_avg = pred.groupby("Department")["AttritionProbability"].mean().reset_index()
    dept_fig = fig_json(px.bar(
        dept_avg, x="Department", y="AttritionProbability",
        color="AttritionProbability", color_continuous_scale="Reds",
        title="Avg Attrition Probability by Department",
        text=dept_avg["AttritionProbability"].apply(lambda x: f"{x:.1%}")
    ).update_traces(textposition="outside").update_layout(coloraxis_showscale=False, height=320))

    return render_template("heatmap.html", fig=fig_json(fig), ot_fig=ot_fig, dept_fig=dept_fig)


@app.route("/atrisk")
def atrisk():
    _, risk, *_ = load_all()

    # Ensure RiskTier is plain Python str for comparisons
    risk = risk.copy()
    risk["RiskTier"] = risk["RiskTier"].astype(str)

    # Handle multiple checkbox values
    tier_args = request.args.getlist("tier")
    tier_filter = tier_args if tier_args else ["Critical", "High"]

    filt = risk[risk["RiskTier"].isin(tier_filter)]
    counts = {t: int((risk["RiskTier"] == t).sum()) for t in ["Critical","High","Medium","Low"]}
    total  = len(risk)

    # Build pie data manually to avoid pandas version issues
    tier_order = ["Critical","High","Medium","Low"]
    tier_counts_vals = [counts[t] for t in tier_order]
    pie = fig_json(go.Figure(go.Pie(
        labels=tier_order,
        values=tier_counts_vals,
        hole=0.45,
        marker=dict(colors=[TIER_CLR[t] for t in tier_order]),
        textinfo="label+percent",
    )).update_layout(title="Risk Tier Distribution", height=320, showlegend=True))

    cols = [c for c in ["EmployeeNumber","Age","Department","JobRole","MonthlyIncome",
            "OverTime","JobSatisfaction","YearsSinceLastPromotion",
            "AttritionProbability","RiskTier","RecommendedAction"] if c in filt.columns]
    rows = filt[cols].head(50).to_dict("records")
    # Convert all values to plain Python types for Jinja2
    rows = [{k: (str(v) if hasattr(v, '__class__') and 'Arrow' in type(v).__name__ else v)
             for k, v in r.items()} for r in rows]

    return render_template("atrisk.html", counts=counts, pie=pie, rows=rows,
        cols=cols, tier_filter=tier_filter, tier_clr=TIER_CLR, total=total)


@app.route("/drivers")
def drivers():
    imp_path = af("feature_importance.csv")
    if not os.path.exists(imp_path):
        return "Run pipeline first", 500
    imp = pd.read_csv(imp_path)
    n = int(request.args.get("n", 15))
    top = imp.head(n)
    fig = px.bar(top[::-1], x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Reds",
        title=f"Top {n} Attrition Drivers — Feature Importance").update_layout(
        coloraxis_showscale=False, height=max(380, n*26))
    img = os.path.join(A, "plots", "feature_importance.png")
    has_img = os.path.exists(img)
    return render_template("drivers.html", fig=fig_json(fig), n=n, has_img=has_img)


@app.route("/download_atrisk")
def download_atrisk():
    from flask import Response
    _, risk, *_ = load_all()
    return Response(risk.to_csv(index=False), mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=at_risk_employees.csv"})


@app.route("/artifacts/plots/<path:filename>")
def plot_img(filename):
    return send_from_directory(os.path.join(A, "plots"), filename)


@app.route("/fairness")
def fairness():
    *_, fair = load_all()
    charts = {}
    for attr in fair:
        img = os.path.join(A, "plots", f"fairness_{attr}.png")
        charts[attr] = os.path.exists(img)
    return render_template("fairness.html", fair=fair, charts=charts)


@app.route("/roi")
def roi():
    _, risk, *_ = load_all()
    sal   = int(request.args.get("sal",   600000))
    rpct  = int(request.args.get("rpct",  125)) / 100
    icost = int(request.args.get("icost", 50000))
    srate = int(request.args.get("srate", 40)) / 100
    at    = risk[risk["RiskTier"].isin(["Critical","High"])]
    exp   = at["AttritionProbability"].sum()
    c_wo  = exp * sal * rpct
    c_int = len(at) * icost
    saved = exp * srate * sal * rpct
    net   = saved - c_int
    roi_v = net / c_int * 100 if c_int else 0
    fig = fig_json(go.Figure(go.Waterfall(
        measure=["absolute","relative","relative","total"],
        x=["Cost w/o Action","Intervention","Savings","Net Benefit"],
        y=[c_wo, -c_int, saved, net],
        decreasing={"marker":{"color":"#c0392b"}},
        increasing={"marker":{"color":"#27ae60"}},
        totals={"marker":{"color":"#2980b9"}}
    )).update_layout(title="Financial Impact (INR)", height=380))
    kpis = {"At-Risk":len(at),"Expected Attritions":f"{exp:.0f}",
            "Cost w/o Action":f"Rs.{c_wo:,.0f}","Intervention Cost":f"Rs.{c_int:,.0f}",
            "Prevented":f"{exp*srate:.0f}","Savings":f"Rs.{saved:,.0f}",
            "Net Benefit":f"Rs.{net:,.0f}","ROI":f"{roi_v:.0f}%"}
    return render_template("roi.html", kpis=kpis, fig=fig,
        sal=sal, rpct=int(rpct*100), icost=icost, srate=int(srate*100))


@app.route("/predict", methods=["GET","POST"])
def predict():
    result = None
    if request.method == "POST":
        f      = request.form
        model  = joblib.load(af("best_model.pkl"))
        scaler = joblib.load(af("scaler.pkl"))
        feats  = json.load(open(af("feature_names.json")))

        # Start with median values from training data so unseen features are reasonable
        pred_df = pd.read_parquet(af("test_predictions.parquet"))
        num_cols = [c for c in feats if c in pred_df.select_dtypes("number").columns]
        medians  = pred_df[num_cols].median().to_dict()
        row = {k: medians.get(k, 0) for k in feats}

        # Override with form values
        age  = int(f["age"]);   income = int(f["income"]); yac = int(f["yac"])
        ysp  = int(f["ysp"]);   yir    = int(f["yir"]);    twy = int(f["twy"])
        js   = int(f["js"]);    wlb    = int(f["wlb"]);    es  = int(f["es"])
        rs   = int(f["rs"]);    nc     = int(f["nc"]);      dist = int(f["dist"])
        so   = int(f["so"]);    tt     = int(f["tt"]);      ot  = int(f["ot"]=="Yes")

        overrides = {
            "Age": age, "MonthlyIncome": income, "YearsAtCompany": yac,
            "YearsSinceLastPromotion": ysp, "YearsInCurrentRole": yir,
            "TotalWorkingYears": twy, "OverTime": ot,
            "JobSatisfaction": js, "WorkLifeBalance": wlb,
            "EnvironmentSatisfaction": es, "RelationshipSatisfaction": rs,
            "NumCompaniesWorked": nc, "DistanceFromHome": dist,
            "StockOptionLevel": so, "TrainingTimesLastYear": tt,
            # Engineered features
            "SatisfactionAvg": np.mean([js, es, rs, wlb]),
            "IncomePerYear":   income / max(twy, 1),
            "Stagnation":      yir / max(twy, 1),
            "PromotionLag":    ysp,
        }
        for k, v in overrides.items():
            if k in row:
                row[k] = v

        X    = pd.DataFrame([row])[feats]
        prob = float(model.predict_proba(pd.DataFrame(scaler.transform(X), columns=feats))[0, 1])
        tier = "Critical" if prob>=0.70 else "High" if prob>=0.50 else "Medium" if prob>=0.30 else "Low"
        gauge = fig_json(go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100, number={"suffix":"%"},
            title={"text": "Attrition Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": TIER_CLR[tier]},
                "steps": [
                    {"range": [0,  30], "color": "#eafaf1"},
                    {"range": [30, 50], "color": "#fef9e7"},
                    {"range": [50, 70], "color": "#fdebd0"},
                    {"range": [70,100], "color": "#fadbd8"},
                ],
            }
        )).update_layout(height=300))
        result = {"prob": f"{prob:.1%}", "tier": tier,
                  "color": TIER_CLR[tier], "action": ACTIONS[tier], "gauge": gauge}
    return render_template("predict.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
