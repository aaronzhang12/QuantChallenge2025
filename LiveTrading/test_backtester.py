#!/usr/bin/env python3
import csv, os, json
from datetime import datetime
from backtester import BacktesterEngine, TradingAlgorithm, Side, GameEvent, Trade
from liquidity_bots import SimpleLiquidityBot, AdvancedLiquidityBot, NoiseTrader, JumpChaserBot, ContrarianSnapBot, EndgameWidenMM, PatientValueInvestorBot, ConfirmationMomentumBot, RunAwareContrarianBot, RiskParityMarketMaker
from algorithm import Strategy, QuantitativeWinningStrategy
from adversarial_bots import MomentumMamba, EventHorizon, ArbitrageAce, VolatilityVulture
from advanced_adversarial_bots import StatisticalArbitrageBot, HighFrequencyScalper, GameFlowSpecialist, PsychologicalManipulator, RiskParityBot
from bayesian_updater import BayesianRegressionStrategy

class TestAlgorithm(TradingAlgorithm):
    """Simple test algorithm that buys when home team scores."""

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.last_home_score = 0

    def on_game_event_update(self, event: GameEvent):
        # Buy when home team scores
        if event.event_type == "SCORE" and event.home_away == "home":
            if event.home_score > self.last_home_score:
                # Buy some shares if we have capital
                if self.capital_remaining > 5000:
                    self.place_market_order("BASKETBALL", Side.BUY, 50)

        self.last_home_score = event.home_score


class MomentumAlgorithm(TradingAlgorithm):
    """Algorithm that trades based on momentum indicators."""

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.momentum_score = 0
        self.recent_events = []

    def on_game_event_update(self, event: GameEvent):
        # Track recent events for momentum
        self.recent_events.append(event)
        if len(self.recent_events) > 20:
            self.recent_events.pop(0)

        # Update momentum score
        old_momentum = self.momentum_score
        self._calculate_momentum()

        # Trade on momentum changes
        momentum_change = self.momentum_score - old_momentum

        if abs(momentum_change) > 2:  # Significant momentum shift
            if momentum_change > 0:  # Home team momentum
                # Buy if we don't have too large a position
                if self.positions["BASKETBALL"] < 200:
                    self.place_market_order("BASKETBALL", Side.BUY, 30)
            else:  # Away team momentum
                # Sell if we don't have too large a position
                if self.positions["BASKETBALL"] > -200:
                    self.place_market_order("BASKETBALL", Side.SELL, 30)

    def _calculate_momentum(self):
        """Calculate momentum based on recent events."""
        self.momentum_score = 0

        for event in self.recent_events[-10:]:  # Last 10 events
            if event.event_type == "SCORE":
                if event.home_away == "home":
                    self.momentum_score += 2
                elif event.home_away == "away":
                    self.momentum_score -= 2

            elif event.event_type in ["STEAL", "BLOCK"]:
                if event.home_away == "home":
                    self.momentum_score += 1
                elif event.home_away == "away":
                    self.momentum_score -= 1

            elif event.event_type == "TURNOVER":
                if event.home_away == "home":
                    self.momentum_score -= 1
                elif event.home_away == "away":
                    self.momentum_score += 1

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "backtest_log.csv")

# Prepare CSV writer with header
CSV_FIELDS = [
    "run_id","time_seconds","home_score","away_score","event_type","shot_type",
    "home_away","player_name","fair_value","best_bid","best_ask","last_trade",
    "position","capital","trader_id"
]
_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create the file with header once at start
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()

def make_logger():
    # Returns a function that appends a row to CSV
    def _log(row: dict):
        row_out = {k: None for k in CSV_FIELDS}
        row_out.update(row)
        row_out["run_id"] = _run_id
        # Coerce numeric fields nicely
        for fld in ["time_seconds","home_score","away_score","fair_value","best_bid","best_ask","last_trade","position","capital"]:
            if fld in row_out and row_out[fld] is not None:
                try:
                    row_out[fld] = float(row_out[fld])
                except Exception:
                    pass
        with open(LOG_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writerow(row_out)
    return _log

# ---- Sandbox Strategy Adapter (paste above main()) ----
# Import your single-file submission strategy module:
import importlib

# If your single-file submission is named e.g. `submission_strategy.py`
# put that filename here (without .py):
SUBMISSION_MODULE_NAME = "gam_algo"  # <-- change if you used a different name

class SandboxAdapter(TradingAlgorithm):
    """
    Wraps the sandbox Strategy so it runs inside our BacktesterEngine.

    - Patches the submission module's global place_* functions to call the
      backtester's engine-bound methods (attached by engine.add_algorithm).
    - Translates enums and callbacks between the two APIs.
    """

    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)

        # Load/refresh the submission module so edits are picked up.
        self.sub = importlib.import_module(SUBMISSION_MODULE_NAME)
        importlib.reload(self.sub)

        # Build the submission Strategy instance
        self.s = self.sub.Strategy()

        # Monkey-patch the module-level order funcs to call our backtest methods
        # (engine.add_algorithm will bind self.place_* to the engine)
        self.sub.place_market_order = self._submit_market
        self.sub.place_limit_order  = self._submit_limit
        self.sub.cancel_order       = self._submit_cancel

        # Keep a small snapshot of best bid/ask for the strategy if it wants them
        self._last_best_bid = None
        self._last_best_ask = None

    # ---------- order routing ----------
    def _submit_market(self, side, ticker, quantity: float):
        # map submission Side -> backtester Side
        bt_side = Side.BUY if side == self.sub.Side.BUY else Side.SELL
        # our backtester uses ticker string "BASKETBALL"
        return self.place_market_order("BASKETBALL", bt_side, int(round(quantity)))

    def _submit_limit(self, side, ticker, quantity: float, price: float, ioc: bool = False):
        bt_side = Side.BUY if side == self.sub.Side.BUY else Side.SELL
        return self.place_limit_order("BASKETBALL", bt_side, int(round(quantity)), float(price), ioc=bool(ioc))

    def _submit_cancel(self, ticker, order_id: int):
        return self.cancel_order("BASKETBALL", order_id)

    # ---------- callback bridges ----------
    def on_orderbook_update(self, ticker: str, side: Side, price: float, quantity: int):
        if ticker != "BASKETBALL":
            return
        # Track bests for convenience (optional)
        if side == Side.BUY:
            self._last_best_bid = price if (self._last_best_bid is None or price > self._last_best_bid) else self._last_best_bid
        else:
            self._last_best_ask = price if (self._last_best_ask is None or price < self._last_best_ask) else self._last_best_ask

        # Map to submission enums and forward
        sub_side = self.sub.Side.BUY if side == Side.BUY else self.sub.Side.SELL
        try:
            self.s.on_orderbook_update(self.sub.Ticker.TEAM_A, sub_side, float(quantity), float(price))
        except TypeError:
            # Some templates use (ticker, side, quantity, price) or (ticker, side, price, quantity).
            # The user’s template says (ticker, side, quantity, price); we handled that.
            pass

    def on_trade_update(self, trade: Trade):
        # The template's on_trade_update signature: (ticker, side, quantity, price).
        # Use the aggressive side from the trade as "side".
        sub_side = self.sub.Side.BUY if trade.aggressive_side == Side.BUY else self.sub.Side.SELL
        self.s.on_trade_update(self.sub.Ticker.TEAM_A, sub_side, float(trade.quantity), float(trade.price))

    def on_account_update(self, order_id: str, quantity: int, price: float, capital_remaining: float):
        # Backtester gives quantity signed by us in separate calls; template expects (ticker, side, price, quantity, capital_remaining)
        side = self.sub.Side.BUY if quantity > 0 else self.sub.Side.SELL
        self.s.on_account_update(self.sub.Ticker.TEAM_A, side, float(price), float(abs(quantity)), float(capital_remaining))

    def on_game_event_update(self, event: GameEvent):
        # Forward exactly per template fields
        self.s.on_game_event_update(
            event_type=event.event_type,
            home_away=event.home_away,
            home_score=event.home_score,
            away_score=event.away_score,
            player_name=event.player_name,
            substituted_player_name=event.substituted_player_name,
            shot_type=event.shot_type,
            assist_player=event.assist_player,
            rebound_type=event.rebound_type,
            coordinate_x=event.coordinate_x,
            coordinate_y=event.coordinate_y,
            time_seconds=event.time_seconds
        )


def main():
    print("Testing Basketball Trading Backtester")
    print("=" * 50)

    # Create backtester
    engine = BacktesterEngine()
    logger_fn = make_logger()

    my_strategy = Strategy("MyStrategy")

    quant_strategy = QuantitativeWinningStrategy("QuantStrategy")
    quant_strategy.set_logger(logger_fn)

    bayes = BayesianRegressionStrategy(
    "BayesBaseline",
    model_path="models/bayes_market_model.npz",  # or the path you saved
    min_elapsed_sec=600,        # ignore the first ~10 minutes of game time
    base_threshold=1.8,         # conservative early-game thresholding
    max_position=250
    )

    # Add various trading algorithms
    algorithms = [
        SimpleLiquidityBot("SimpleBot1"),
        SimpleLiquidityBot("SimpleBot2"),
        AdvancedLiquidityBot("AdvancedBot1"),
        NoiseTrader("NoiseTrader1"),
        NoiseTrader("NoiseTrader2"), 
        # NEW adversaries:
        JumpChaserBot("JumpChaser"),
        ContrarianSnapBot("ContrarianSnap"),
        EndgameWidenMM("EndgameMM"),
        MomentumMamba("MomentumMamba"),
        EventHorizon("EventHorizon"),
        ArbitrageAce("ArbitrageAce"),
        VolatilityVulture("VolatilityVulture"),
        PatientValueInvestorBot("PatientValue"),
        ConfirmationMomentumBot("ConfirmMo"),
        RunAwareContrarianBot("RunFade"),
        RiskParityMarketMaker("RiskParityMM"),
        StatisticalArbitrageBot("StatisticalArbitrageBot"),
        HighFrequencyScalper("HighFrequencyScalper"), 
        GameFlowSpecialist("GameFlowSpecialist"),
        RiskParityBot("RiskParityBot"),
        SandboxAdapter("GAMKelly"),    # trader_id label for the submission bot

        # Your strategies:
        my_strategy,
        quant_strategy,
        bayes
    ]

    for algo in algorithms:
        engine.add_algorithm(algo)

    # Load game events
    print("Loading game events from example-game.json...")
    engine.load_game_events("example-game.json")

    # Run backtest
    print("\nRunning backtest...")
    results = engine.run_backtest()

    # Display results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Algorithms: {results['algorithms']}")
    print()

    print("FINAL RANKINGS:")
    print("-" * 50)
    for i, result in enumerate(results['rankings']):
        print(f"{result['rank']}. {result['trader_id']:<15} | "
              f"PnL: ${result['pnl']:>8.2f} | "
              f"Return: {result['return_pct']:>6.2f}% | "
              f"Capital: ${result['final_capital']:>10.2f}")

    print()
    print("Competition Scoring (R^1.25 where R is rank):")
    print("-" * 50)
    total_score = 0
    for result in results['rankings']:
        rank_score = result['rank'] ** 1.25
        total_score += rank_score
        print(f"{result['trader_id']:<15} | Rank: {result['rank']:>2} | Score: {rank_score:>6.2f}")

    print(f"\nTotal Score Sum: {total_score:.2f}")

    # Additional statistics
    print("\nTRADE ANALYSIS:")
    print("-" * 50)
    if engine.trades_history:
        total_volume = sum(trade.quantity for trade in engine.trades_history)
        avg_price = sum(trade.price * trade.quantity for trade in engine.trades_history) / total_volume
        min_price = min(trade.price for trade in engine.trades_history)
        max_price = max(trade.price for trade in engine.trades_history)

        print(f"Total Volume: {total_volume:,} shares")
        print(f"Average Price: ${avg_price:.2f}")
        print(f"Price Range: ${min_price:.2f} - ${max_price:.2f}")

        # Show sample trades
        print(f"\nSample Trades (first 5):")
        for i, trade in enumerate(engine.trades_history[:5]):
            print(f"  {trade.buy_trader_id} -> {trade.sell_trader_id}: "
                  f"{trade.quantity} @ ${trade.price:.2f} "
                  f"(time: {trade.timestamp:.1f}s)")


if __name__ == "__main__":
    main()