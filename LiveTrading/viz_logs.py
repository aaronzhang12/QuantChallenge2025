# viz_logs.py
import argparse, json, os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

KEY_EVENTS = {"SCORE","STEAL","TURNOVER","BLOCK","FOUL","TIMEOUT","SUBSTITUTION","END_PERIOD","END_GAME"}

def load_logs(csv_path: str, run_id: str | None):
    df = pd.read_csv(csv_path)
    if run_id:
        df = df[df["run_id"] == run_id]
    # Ensure numeric & sort by time_seconds DESC (game time remaining)
    for c in ["time_seconds","fair_value","best_bid","best_ask","last_trade","position","capital","home_score","away_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # For nicer tooltips
    df["score_str"] = df["home_score"].fillna(0).astype(int).astype(str) + "-" + df["away_score"].fillna(0).astype(int).astype(str)
    # Avoid dup rows (multiple traders) if you want—here we keep all and let you filter by trader_id
    return df

def load_events(json_path: str):
    with open(json_path, "r") as f:
        events = json.load(f)
    ev = pd.DataFrame(events)
    # standardize missing cols for safety
    for c in ["shot_type","player_name","home_away","event_type","time_seconds","home_score","away_score"]:
        if c not in ev.columns:
            ev[c] = None
    # Only keep key events for markers
    ev_key = ev[ev["event_type"].isin(KEY_EVENTS)].copy()
    return ev_key

def plot_interactive(df: pd.DataFrame, ev_key: pd.DataFrame, trader: str | None, out_html: str):
    # Filter to a single trader if specified
    if trader:
        df = df[df["trader_id"] == trader].copy()

    df = df.sort_values("time_seconds", ascending=False)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        specs=[[{}],[{}]],
        subplot_titles=("Price / Fair Value", "Position")
    )

    # --- Top panel: Fair value + bid/ask + last trade ---
    hover = ("<b>%{customdata[0]}</b><br>Time rem: %{x:.1f}s<br>"
             "Score: %{customdata[1]}<br>Event: %{customdata[2]} %{customdata[3]} (%{customdata[4]})<br>"
             "Fair: %{customdata[5]:.2f}<br>Bid: %{customdata[6]:.2f}<br>Ask: %{customdata[7]:.2f}<br>"
             "Last trade: %{customdata[8]:.2f}<br>Pos: %{customdata[9]:.0f}<br>Cap: %{customdata[10]:.0f}")

    custom = df[[
        "trader_id","score_str","event_type","shot_type","home_away",
        "fair_value","best_bid","best_ask","last_trade","position","capital"
    ]].fillna("")

    fig.add_trace(go.Scatter(
        x=df["time_seconds"], y=df["fair_value"],
        name="Fair Value",
        mode="lines",
        customdata=custom.values,
        hovertemplate=hover
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["time_seconds"], y=df["best_bid"],
        name="Best Bid",
        mode="lines",
        line=dict(dash="dot"),
        customdata=custom.values,
        hovertemplate=hover
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["time_seconds"], y=df["best_ask"],
        name="Best Ask",
        mode="lines",
        line=dict(dash="dot"),
        customdata=custom.values,
        hovertemplate=hover
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["time_seconds"], y=df["last_trade"],
        name="Last Trade",
        mode="lines",
        line=dict(dash="dash"),
        customdata=custom.values,
        hovertemplate=hover
    ), row=1, col=1)

    # --- Event markers (vertical lines) ---
    # Use modest number of markers to keep UI responsive
    ev_plot = ev_key.copy()
    ev_plot = ev_plot.sort_values("time_seconds", ascending=False)
    ev_plot = ev_plot[ev_plot["time_seconds"].between(df["time_seconds"].min(), df["time_seconds"].max())]

    for _, r in ev_plot.iterrows():
        x = float(r["time_seconds"])
        name = f"{r['event_type']}"
        if pd.notna(r.get("shot_type")) and r["shot_type"]:
            name += f" ({r['shot_type']})"
        if pd.notna(r.get("home_away")) and r["home_away"]:
            name += f" [{r['home_away']}]"
        fig.add_vline(
            x=x, line_width=1, line_dash="dot", line_color="#999",
            annotation_text=name, annotation_position="top right", opacity=0.3, row=1, col=1
        )

    # --- Bottom panel: Position (bar/line) ---
    fig.add_trace(go.Scatter(
        x=df["time_seconds"], y=df["position"],
        name="Position",
        mode="lines",
        hovertemplate="Time rem: %{x:.1f}s<br>Position: %{y:.0f}"
    ), row=2, col=1)

    # Layout tweaks
    fig.update_layout(
        title=f"Basketball Binary Market — Run: {df['run_id'].iloc[0] if len(df)>0 else 'N/A'} — Trader: {trader or 'ALL'}",
        xaxis=dict(title="Time Remaining (s)", autorange="reversed"),
        yaxis=dict(title="Price"),
        xaxis2=dict(title="Time Remaining (s)", autorange="reversed"),
        yaxis2=dict(title="Position"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=800,
        hovermode="x unified",
        template="plotly_white"
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved interactive viz to: {out_html}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/backtest_log.csv", help="Path to the CSV log")
    ap.add_argument("--json", default="example-game.json", help="Path to the game JSON")
    ap.add_argument("--run", default=None, help="Filter by run_id (optional)")
    ap.add_argument("--trader", default="MyStrategy", help="Filter by trader_id (e.g., MyStrategy)")
    ap.add_argument("--out", default="viz.html", help="Output HTML file")
    args = ap.parse_args()

    df = load_logs(args.csv, args.run)
    if df.empty:
        print("No rows found. Check CSV path or run_id filter.")
        return
    ev_key = load_events(args.json)
    plot_interactive(df, ev_key, args.trader, args.out)

if __name__ == "__main__":
    main()