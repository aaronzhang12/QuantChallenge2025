# Basketball Trading - LiveTrading Directory

A comprehensive backtesting framework and strategy collection for the QuantChallenge 2025 basketball trading competition.

## Overview

This directory contains a backtesting engine for basketball game probability trading, along with multiple trading strategies, supporting bots, game data, and visualization tools. Algorithms trade contracts that pay $100 if the home team wins and $0 if they lose.

## Core Files

### Backtesting Engine
- **`backtester.py`** - Full order matching engine with price-time priority, market/limit orders, position tracking, and PnL calculation
- **`test_backtester.py`** - Test runner with comprehensive tournament setup including 20+ competing algorithms
- **`liquidity_bots.py`** - Market-making bots that provide liquidity to the order book during backtesting
- **`adversarial_bots.py`** - Basic adversarial trading bots for testing strategy robustness
- **`advanced_adversarial_bots.py`** - More sophisticated adversarial bots with advanced behaviors

### Game Data
- **`example-game.json`** - Sample basketball game event data
- **`game1.json` through `game5.json`** - Additional game event datasets for comprehensive backtesting

### Trading Strategies

#### Algorithm Templates
- **`algorithm.py`** - Base strategy implementations:
  - `Strategy`: Fair value calculation with threshold-based trading and position limits
  - `QuantitativeWinningStrategy`: Conservative strategy with edge requirements and trade limits
- **`algorithm_template.py`** - Template for creating new strategies

#### Bayesian Trading Strategies with Online Learning

These strategies implement **Bayesian linear regression** with **online parameter updates** and **Kelly Criterion** for optimal position sizing:

- **`bayesian_algo_updated.py`** - Full-featured Bayesian strategy with:
  - Early-game guard for price learning (600 seconds)
  - Bayesian linear regression on game & microstructure features
  - Scaled Kelly criterion sizing adapted for prediction markets
  - Per-trade and rolling-window notional throttles
  - Comprehensive feature engineering (events, shots, momentum, pace)

- **`bayesian_algo_conservative.py`** - Balanced Bayesian + Kelly strategy:
  - Dynamic 4-minute learning guard (works for any game format)
  - Conservative sizing with smaller edges for active engagement
  - Hard position caps, per-order caps, and cooldowns
  - Take-profit and soft stop-loss risk management
  - Spread-aware execution (crosses only when reasonable)

- **`bayesian_new.py`** - Bayesian strategy with trained priors:
  - Informed priors based on offline analysis of historical game data
  - Emphasizes time-scaled score differential (most critical feature)
  - Robust risk management with position limits and throttles
  - Adaptive player features learned online
  - Dynamic learning guard and profit-taking mechanisms

- **`bayesian_new_log_prune.py`** - Dynamic risk-adjusted Bayesian strategy:
  - Reduces Kelly scaling as realized profits accumulate
  - Takes profits earlier to lock in gains
  - Adjusts risk exposure based on performance
  - Same Bayesian framework with profit-based dampening

- **`bayesian_updater.py`** - Bayesian regression baseline with model persistence:
  - Loads pre-trained models from disk (`models/bayes_market_model.npz`)
  - Predicts market prices from game state features
  - Conservative early-game behavior (600+ second guard)
  - Online posterior updates from observed trade prints
  - Standardized features with proper normalization

**Key Bayesian Features:**
- **Online Parameter Updates**: All Bayesian strategies continuously update their model parameters as new market data arrives
- **Kelly Criterion Sizing**: Position sizes are calculated using scaled Kelly criterion (`f = (edge) / (variance)`), scaled conservatively (typically 0.04-0.25x full Kelly)
- **Risk Management**: Multiple layers including:
  - Position limits (typically 80 shares max)
  - Per-trade caps (typically 10 shares)
  - Cooldown periods between trades (3-5 seconds)
  - Take-profit levels ($1.75-$2.25 per share)
  - Soft stop-losses ($3.25 per share)
  - Rolling notional throttles (limit $ traded in time windows)

#### Other Live Trading Strategies
- **`low_risk_taker.py`** - Risk-managed strategy with stop-loss and take-profit levels, profit target cutoff
- **`low_risk_taker_v2.py`** - Enhanced version with order pending state to prevent duplicate orders and multi-game detection
- **`player_substitution.py`** - Tracks player performance ratings and trades on lineup changes
- **`second_order.py`** - Exploits second-order effects: bonus foul situations leading to free throws
- **`second_order_rebound.py`** - Combines foul bonus and offensive rebound trading signals
- **`gam_algo.py`** - Generalized Additive Model (GAM) based trading strategy

### Visualization Tools
- **`viz_logs.py`** - Interactive visualization tool for analyzing strategy performance:
  - Plots fair value, bid/ask spreads, last trade prices over time
  - Displays position and capital evolution
  - Marks key game events (scores, steals, turnovers, fouls, etc.)
  - Supports filtering by trader ID and run ID
  - Outputs interactive HTML reports with Plotly
- **`viz_quant.html`** - Sample visualization output
- **`viz.html`** - Additional visualization output

## Trading Engine Features

### Order Types
- **Market orders**: Immediate execution with market impact (slippage increases with order size)
- **Limit orders**: Rest in book at specified price with optional IOC (immediate-or-cancel)
- **Order cancellation**: Remove resting orders from the book

### Market Structure
- **Contract payout**: $100 if home team wins, $0 if loses
- **Starting capital**: $100,000 per algorithm
- **Transaction costs**: $0.005 per share traded
- **Market impact**: Dynamic slippage based on order size (up to 15% for 50+ shares)
- **Settlement**: Positions automatically settled at game end

### Data Callbacks

Algorithms receive updates via:
```python
def on_game_event_update(event: GameEvent)        # Game events (scores, fouls, etc.)
def on_orderbook_update(ticker, price, quantity)  # Book changes
def on_trade_update(trade: Trade)                 # Market trades
def on_account_update(order_id, quantity, price, capital)  # Order fills
```

## Game Events

The backtester processes real basketball events:
- **SCORE** - Successful shots (TWO_POINT, THREE_POINT, FREE_THROW)
- **MISSED** - Missed shots
- **REBOUND** - Offensive/defensive rebounds
- **FOUL** - Personal fouls (triggers bonus after 4 team fouls)
- **STEAL**, **BLOCK** - Defensive plays
- **TURNOVER** - Possession changes
- **SUBSTITUTION** - Player lineup changes
- **START_PERIOD**, **END_PERIOD**, **END_GAME** - Game flow

Each event includes:
- Home/away scores
- Time remaining (seconds, counts down from 2880)
- Player information
- Event-specific data (shot coordinates, shot type, rebound type, etc.)

## Strategy Approaches

### Bayesian + Kelly Criterion (`bayesian_*.py`)
The Bayesian strategies are the most sophisticated in the suite:

**Core Methodology:**
1. **Bayesian Linear Regression**: Predict fair contract price from features
   - Features include: score differential, time remaining, momentum, pace, events, shots, player impact
   - Online posterior updates: `μ ← μ + K·(y - ŷ)` and `Σ ← Σ - K·(Σx)ᵀ`
   - Some strategies use informed priors from offline analysis

2. **Kelly Criterion Position Sizing**: Optimal capital allocation
   - Long fraction: `f_L = (π - p)/(1 - p)`
   - Short fraction: `f_S = (p - π)/p`
   - Scaled conservatively (0.04-0.25x) for real-world risk tolerance
   - Converts to shares: `shares = (f × capital) / price`

3. **Risk Management Layers**:
   - **Early-game guard**: Learn for 4-10 minutes before taking directional risk
   - **Position limits**: Hard cap at 80-250 shares (strategy-dependent)
   - **Per-trade caps**: Limit order size to 10-2000 shares
   - **Cooldowns**: 3-5 second minimum between trades
   - **Profit taking**: Exit at $1.75-$2.25 profit per share
   - **Stop loss**: Cut losses at $3.25 per share
   - **Notional throttles**: Limit $ traded in rolling time windows

4. **Execution Logic**:
   - Cross the spread only when edge exceeds threshold (time-decaying)
   - Post passive limit orders when edge is small or spread is wide
   - IOC (immediate-or-cancel) quotes to avoid cluttering the book

**Strategy Variants:**
- **`bayesian_algo_updated.py`**: Full-featured with comprehensive features
- **`bayesian_algo_conservative.py`**: More conservative sizing, good for engagement
- **`bayesian_new.py`**: Uses trained priors from historical analysis
- **`bayesian_new_log_prune.py`**: Reduces risk as profits accumulate
- **`bayesian_updater.py`**: Loads pre-trained models, emphasizes model persistence

### Fair Value Trading (`algorithm.py`)
- Calculate win probability from score differential and time remaining
- Enter positions when market price deviates from fair value
- Manage position limits and capital

### Low Risk (`low_risk_taker.py`, `low_risk_taker_v2.py`)
- Strict risk management with stop-loss and take-profit levels
- Profit target cutoff ($100 realized PnL, then stop trading)
- Game change detection to prevent trading across different games

### Event-Based Trading
- **Player substitution** (`player_substitution.py`): Track player impact ratings and trade on lineup strength changes
- **Second order effects** (`second_order.py`, `second_order_rebound.py`): Exploit predictable outcomes like free throws from bonus fouls and offensive rebound putbacks

## Usage

### Run Backtest

Basic backtest with single game:
```bash
python3 test_backtester.py
```

This runs a tournament with 20+ algorithms including your strategies, liquidity bots, and adversarial bots.

### Visualize Results

After running backtests that generate logs (CSV) and game data (JSON), visualize the results:

```bash
# Basic usage - visualize all runs
python3 viz_logs.py logs.csv game1.json

# Filter by specific run ID
python3 viz_logs.py logs.csv game1.json --run_id abc123

# Filter by specific trader
python3 viz_logs.py logs.csv game1.json --trader MyStrategy

# Specify output HTML file
python3 viz_logs.py logs.csv game1.json --output my_results.html
```

The visualizer creates an interactive HTML report with:
- **Top panel**: Fair value, bid/ask spread, and last trade price over time
- **Bottom panel**: Position size evolution
- **Event markers**: Vertical lines marking key game events (scores, steals, etc.)
- **Interactive tooltips**: Hover over any point to see detailed information

### Test with Multiple Games

To backtest across multiple games for more robust evaluation:

```python
from backtester import BacktesterEngine
from bayesian_algo_updated import Strategy as BayesianStrategy

engine = BacktesterEngine()
engine.add_algorithm(BayesianStrategy("BayesBot"))

# Run across multiple games
for game_file in ["game1.json", "game2.json", "game3.json", "game4.json", "game5.json"]:
    engine.load_game_events(game_file)
    results = engine.run_backtest()
    print(f"\n{game_file} Results:")
    for result in results['rankings']:
        print(f"{result['rank']}. {result['trader_id']}: ${result['pnl']:.2f}")
```

### Create New Strategy
```python
from backtester import TradingAlgorithm, Side, GameEvent

class MyStrategy(TradingAlgorithm):
    def __init__(self, trader_id: str, capital: float = 100000.0):
        super().__init__(trader_id, capital)
        self.fair_value = 50.0

    def on_game_event_update(self, event: GameEvent):
        if event.event_type == "SCORE" and event.home_away == "home":
            # Home team scored, consider buying
            if self.capital_remaining > 5000:
                self.place_market_order("BASKETBALL", Side.BUY, 10)

    def on_account_update(self, order_id, quantity, price, capital):
        print(f"Filled: {quantity} @ ${price}")
```

### Add to Backtester
```python
from backtester import BacktesterEngine
from my_strategy import MyStrategy

engine = BacktesterEngine()
engine.add_algorithm(MyStrategy("MyBot"))
engine.load_game_events("example-game.json")
results = engine.run_backtest()

for result in results['rankings']:
    print(f"{result['rank']}. {result['trader_id']}: ${result['pnl']:.2f}")
```

## Competition Scoring

Uses `R^1.25` ranking score where R is final rank:
- 1st place: 1.00 points
- 2nd place: 2.38 points
- 3rd place: 3.95 points

Lower cumulative score across all games wins the competition.

## Strategy Tips

1. **Time decay**: Win probability becomes more certain as time decreases - the Bayesian strategies account for this with time-scaled features
2. **Event magnitude**: Three-pointers have more impact than two-pointers or free throws
3. **Market impact**: Large orders incur significant slippage
4. **Transaction costs**: $0.005/share adds up with frequent trading
5. **Position limits**: Avoid excessive exposure to single outcomes
6. **Second-order effects**: Fouls lead to free throws, rebounds lead to putbacks
7. **Game detection**: Score drops of 20+ points indicate a new game started
8. **Kelly sizing**: Use scaled Kelly (0.1-0.25x) rather than full Kelly to avoid over-betting
9. **Learning period**: Allow your model to observe price action before taking large directional bets
10. **Spread awareness**: Don't cross wide spreads unless edge is substantial
11. **Momentum**: Recent scoring runs contain predictive information about near-term outcomes
12. **Take profits**: Lock in gains at reasonable levels rather than hoping for perfection

## Bayesian Strategy Configuration

All Bayesian strategies can be tuned via their class constants. Key parameters:

### Sizing & Risk
```python
KELLY_SCALE = 0.10              # Fraction of full Kelly (0.04-0.25 typical)
MAX_POSITION_SHARES = 80.0      # Hard inventory cap
MAX_ORDER_SHARES = 10.0         # Per-trade size limit
```

### Edge Thresholds (probability points)
```python
EDGE_ENTRY_EARLY = 0.060        # Required edge in early game
EDGE_ENTRY_MID = 0.050          # Required edge in mid game
EDGE_ENTRY_LATE = 0.040         # Required edge in late game
```

### Execution & Timing
```python
TRADE_COOLDOWN_MS = 4000        # Min milliseconds between trades
MAX_CROSS_SPREAD = 1.50         # Only cross if spread ≤ $1.50
```

### Profit Taking & Loss Protection
```python
TAKE_PROFIT = 2.25              # Exit at this profit per share
SOFT_STOP_LOSS = 3.25           # Cut losses at this level
```

### Model Parameters
```python
NOISE_VAR = 9.0                 # Observation noise (price variance)
PRIOR_VAR = 400.0               # Prior uncertainty on parameters
```

## Liquidity Bots

The `liquidity_bots.py` file provides market makers that populate the order book during backtesting:

- **SimpleLiquidityBot**: Posts two-sided quotes around fair value
- **NoisyLiquidityBot**: Adds random noise to quote prices
- **AdaptiveLiquidityBot**: Adjusts spreads based on market conditions

These bots are essential for realistic backtesting, as they:
1. Provide counterparty liquidity for strategy orders
2. Create realistic bid-ask spreads
3. Generate price discovery through continuous quoting
4. Test how strategies perform in different market conditions

Configure in `test_backtester.py` by adding liquidity bots to the engine before running.

## Adversarial Bots

Two levels of adversarial bots test strategy robustness:

### `adversarial_bots.py` - Basic Adversaries
- **FrontRunnerBot**: Attempts to detect and front-run large orders
- **StopHunterBot**: Tries to trigger stop losses
- **NoiseTraderBot**: Random uninformed trading

### `advanced_adversarial_bots.py` - Sophisticated Adversaries
- **PredictiveAdversaryBot**: Uses simple models to predict strategy behavior
- **AdaptiveAdversaryBot**: Learns from strategy patterns over time
- **CoordinatedAdversaryBot**: Multiple bots working together

These help ensure strategies are robust to:
- Information leakage through order placement patterns
- Manipulation attempts
- Adverse selection
- Market microstructure effects

## Technical Details

- **Order matching**: Price-time priority (best price first, then earliest timestamp)
- **Market impact**: Slippage = price × (min(quantity/50, 0.15)), compounds for large orders
- **Game timing**: 2880 seconds total (48 minutes), time counts down to 0
- **Settlement**: Positions valued at $100 (home win) or $0 (home loss)
- **Multi-game support**: Backtester automatically detects and handles multiple games via END_GAME events

## Game Event Data Format

Game JSON files contain arrays of event objects with the following structure:

```json
{
  "event_type": "SCORE",
  "home_away": "home",
  "home_score": 45,
  "away_score": 42,
  "player_name": "John Smith",
  "shot_type": "THREE_POINT",
  "time_seconds": 1440.0,
  "coordinate_x": 23.5,
  "coordinate_y": 8.2
}
```

**Supported event types:**
- `SCORE`, `MISSED` - Shot attempts (with `shot_type`: THREE_POINT, TWO_POINT, DUNK, LAYUP, FREE_THROW)
- `REBOUND` - Rebounds (with `rebound_type`: offensive/defensive)
- `FOUL` - Personal fouls (triggers bonus after 4 team fouls)
- `STEAL`, `BLOCK` - Defensive plays
- `TURNOVER` - Possession changes
- `SUBSTITUTION` - Player lineup changes (with `substituted_player_name`)
- `TIMEOUT` - Team timeouts
- `START_PERIOD`, `END_PERIOD`, `END_GAME` - Game flow markers

Each event includes scores and time remaining (seconds). The Bayesian strategies extract rich features from these events.

## Project Structure

```
LiveTrading/
├── Core Engine
│   ├── backtester.py              # Order matching and execution
│   ├── test_backtester.py         # Tournament runner
│   └── algorithm_template.py      # Strategy template
│
├── Base Strategies
│   ├── algorithm.py               # Fair value strategies
│   ├── low_risk_taker.py          # Risk-managed trading
│   ├── low_risk_taker_v2.py       # Enhanced risk management
│   ├── player_substitution.py     # Player-based trading
│   ├── second_order.py            # Foul/free throw exploitation
│   ├── second_order_rebound.py    # Rebound exploitation
│   └── gam_algo.py                # GAM-based trading
│
├── Bayesian Strategies (Online Learning + Kelly)
│   ├── bayesian_algo_updated.py       # Full-featured Bayesian
│   ├── bayesian_algo_conservative.py  # Conservative Bayesian
│   ├── bayesian_new.py                # Trained priors
│   ├── bayesian_new_log_prune.py      # Profit-dampened risk
│   └── bayesian_updater.py            # Model persistence
│
├── Market Infrastructure
│   ├── liquidity_bots.py              # Market makers
│   ├── adversarial_bots.py            # Basic adversaries
│   └── advanced_adversarial_bots.py   # Advanced adversaries
│
├── Game Data
│   ├── example-game.json
│   ├── game1.json
│   ├── game2.json
│   ├── game3.json
│   ├── game4.json
│   └── game5.json
│
├── Visualization
│   ├── viz_logs.py                    # Interactive plotting
│   ├── viz_quant.html                 # Sample output
│   └── viz.html                       # Sample output
│
└── README.md                          # This file
```

## Competition Scoring

Uses `R^1.25` ranking score where R is final rank:
- 1st place: 1.00 points
- 2nd place: 2.38 points
- 3rd place: 3.95 points

Lower cumulative score across all games wins the competition.

## Next Steps

1. **Run backtests** with `python3 test_backtester.py` to see all strategies compete
2. **Visualize results** with `viz_logs.py` to analyze performance
3. **Tune Bayesian strategies** by adjusting their configuration parameters
4. **Test across multiple games** using game1-5.json for robust evaluation
5. **Create new strategies** using the algorithm_template.py as a starting point
6. **Analyze edge cases** using adversarial bots to stress-test strategies
7. **Optimize Kelly scaling** based on your risk tolerance and capital constraints

The Bayesian strategies with Kelly criterion sizing represent state-of-the-art approaches combining:
- Statistical learning (Bayesian updates)
- Optimal capital allocation (Kelly criterion)  
- Practical risk management (scaling, limits, throttles)
- Market microstructure awareness (spread-based execution)

Good luck with the competition! 🏀📈

