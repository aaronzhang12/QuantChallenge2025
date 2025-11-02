from enum import Enum
from typing import Optional, Set

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    return True

class Strategy:

    def reset_state(self) -> None:
        print("Resetting strategy state for a new game.")
        self.home_team_fouls = 0
        self.away_team_fouls = 0
        self.home_team_in_bonus = False
        self.away_team_in_bonus = False
        self.home_players_on_court: Set[str] = set()
        self.away_players_on_court: Set[str] = set()
        self.mid_price: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.current_position: float = 0.0
        self.entry_price: float = 0.0
        self.entry_reason: Optional[str] = None

    def __init__(self) -> None:
        self.TRADE_QUANTITY = 5.0
        self.STOP_LOSS = 2.5
        self.FOUL_BONUS_THRESHOLD = 4
        self.reset_state()

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        pass

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        pass

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        print(f"Account Update: Filled {side.name} order for {quantity} at {price}")

        if side == Side.BUY:
            self.current_position += quantity
        else:
            self.current_position -= quantity

        if abs(self.current_position) < 0.01:
            self.entry_price = 0.0
            self.entry_reason = None
            print("Position closed.")
        else:
            self.entry_price = price

    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        if player_name:
            if home_away == "home" and len(self.home_players_on_court) < 5:
                self.home_players_on_court.add(player_name)
            elif home_away == "away" and len(self.away_players_on_court) < 5:
                self.away_players_on_court.add(player_name)

        if event_type == "SUBSTITUTION" and player_name and substituted_player_name:
            target_set = self.home_players_on_court if home_away == "home" else self.away_players_on_court
            if substituted_player_name in target_set:
                target_set.remove(substituted_player_name)
            target_set.add(player_name)

        if event_type == "SCORE" and self.entry_reason == "FOUL_BONUS_TRADE":
            if shot_type in ["TWO_POINT", "THREE_POINT"]:
                print(f"Exiting position on live-play score ({shot_type}).")
                self._close_position_at_market()
            return

        if event_type == "START_PERIOD":
            print("New period starting. Resetting foul counts.")
            self.home_team_fouls = 0
            self.away_team_fouls = 0
            self.home_team_in_bonus = False
            self.away_team_in_bonus = False
            return

        if event_type == "FOUL":
            is_home_foul = (home_away == "home")

            if is_home_foul:
                self.home_team_fouls += 1
                if self.home_team_fouls > self.FOUL_BONUS_THRESHOLD:
                    self.home_team_in_bonus = True
            else:
                self.away_team_fouls += 1
                if self.away_team_fouls > self.FOUL_BONUS_THRESHOLD:
                    self.away_team_in_bonus = True

            if self.current_position == 0:
                if not is_home_foul and self.away_team_in_bonus:
                    print(f"TRADE SIGNAL: Buying due to Away Team bonus foul ({self.away_team_fouls}).")
                    place_market_order(Side.BUY, Ticker.TEAM_A, self.TRADE_QUANTITY)
                    self.entry_reason = "FOUL_BONUS_TRADE"

                elif is_home_foul and self.home_team_in_bonus:
                    print(f"TRADE SIGNAL: Selling due to Home Team bonus foul ({self.home_team_fouls}).")
                    place_market_order(Side.SELL, Ticker.TEAM_A, self.TRADE_QUANTITY)
                    self.entry_reason = "FOUL_BONUS_TRADE"

        if event_type == "END_GAME":
            if self.current_position != 0:
                self._close_position_at_market()
            self.reset_state()
            return

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        if not bids or not asks:
            return

        self.best_bid = bids[0][0]
        self.best_ask = asks[0][0]
        self.mid_price = (self.best_bid + self.best_ask) / 2.0

        if self.current_position > 0 and self.best_bid <= self.entry_price - self.STOP_LOSS:
            print(f"STOP-LOSS TRIGGERED for LONG position. Best bid at {self.best_bid}.")
            self._close_position_at_market()
        elif self.current_position < 0 and self.best_ask >= self.entry_price + self.STOP_LOSS:
            print(f"STOP-LOSS TRIGGERED for SHORT position. Best ask at {self.best_ask}.")
            self._close_position_at_market()

    def _close_position_at_market(self) -> None:
        if self.current_position > 0:
            place_market_order(Side.SELL, Ticker.TEAM_A, abs(self.current_position))
        elif self.current_position < 0:
            place_market_order(Side.BUY, Ticker.TEAM_A, abs(self.current_position))