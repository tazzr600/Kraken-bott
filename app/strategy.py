from dataclasses import dataclass
from decimal import Decimal

@dataclass
class PairState:
    up: Decimal = Decimal("0")
    down: Decimal = Decimal("0")

def complete_set_cost(up_vwap, down_vwap):
    if up_vwap is None or down_vwap is None:
        return None
    return Decimal(str(up_vwap)) + Decimal(str(down_vwap))

def next_action(fair_up, up_ask, down_ask, state, min_entry=0.025, min_add=0.018, max_unpaired=10):
    edge_up = fair_up - up_ask
    edge_down = (1.0 - fair_up) - down_ask
    if state.up == 0 and state.down == 0:
        if edge_up >= min_entry:
            return "ENTRY", "UP", edge_up
        if edge_down >= min_entry:
            return "ENTRY", "DOWN", edge_down
        return "HOLD", "UP", max(edge_up, edge_down)
    if fair_up >= 0.5:
        if state.down < state.up and state.up - state.down > max_unpaired:
            return "HEDGE", "DOWN", edge_down
        if edge_up >= min_add:
            return "ADD", "UP", edge_up
    else:
        if state.up < state.down and state.down - state.up > max_unpaired:
            return "HEDGE", "UP", edge_up
        if edge_down >= min_add:
            return "ADD", "DOWN", edge_down
    return "HOLD", "UP", max(edge_up, edge_down)
