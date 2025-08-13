import json
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime, timedelta

import azure.functions as func


# ---------- Employee -> Line: all maximum-cardinality assignments ----------
def employee_assignment(lines_available: List[str], employees_available: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
    # Build graph: line -> list of qualified employee IDs (preserve given order)
    line_to_emps = {
        line: [emp["id"] for emp in employees_available if line in emp.get("qualifiedLines", [])]
        for line in lines_available
    }

    # 1) Compute one maximum matching size using Kuhn's algorithm
    match_emp_to_line: Dict[str, str] = {}

    def try_augment(line: str, seen: Set[str]) -> bool:
        for emp in line_to_emps.get(line, []):
            if emp in seen:
                continue
            seen.add(emp)
            if emp not in match_emp_to_line or try_augment(match_emp_to_line[emp], seen):
                match_emp_to_line[emp] = line
                return True
        return False

    for line in lines_available:
        try_augment(line, set())
    max_size = len(match_emp_to_line)
    if max_size == 0:
        return []

    # 2) Enumerate all matchings of size == max_size (backtracking + pruning)
    ordered_lines = sorted(lines_available, key=lambda ln: len(line_to_emps.get(ln, [])))
    results_set: Set[Tuple[Tuple[str, str], ...]] = set()
    used_emps: Set[str] = set()
    cur_pairs: List[Tuple[str, str]] = []

    total_emps = len({emp["id"] for emp in employees_available})

    def backtrack(i: int):
        remaining_lines = len(ordered_lines) - i
        remaining_emps = total_emps - len(used_emps)
        if len(cur_pairs) + min(remaining_lines, remaining_emps) < max_size:
            return

        if i == len(ordered_lines):
            if len(cur_pairs) == max_size:
                results_set.add(tuple(sorted(cur_pairs)))
            return

        line = ordered_lines[i]
        candidates = line_to_emps.get(line, [])

        # Option: skip line (only if still possible to reach max_size)
        backtrack(i + 1)

        # Option: assign an eligible employee
        for emp in candidates:
            if emp in used_emps:
                continue
            used_emps.add(emp)
            cur_pairs.append((line, emp))
            backtrack(i + 1)
            cur_pairs.pop()
            used_emps.remove(emp)

    backtrack(0)

    # Convert to required format
    results = [[{line: emp} for (line, emp) in combo] for combo in results_set]
    # Deterministic order
    results.sort(key=lambda comb: [(list(d.keys())[0], list(d.values())[0]) for d in comb])
    return results


# ---------- Quantity scoring for an assignment (by line availability only) ----------
def score_assignment_quantity(lines_for_combo: List[str],
                              production_orders: List[Dict[str, Any]],
                              materials: List[Dict[str, Any]]) -> int:
    mat_to_lines: Dict[str, Set[str]] = {m["material_number"]: set(m.get("line", [])) for m in materials}
    lines_set = set(lines_for_combo)
    total_qty = 0
    for po in production_orders:
        allowed = mat_to_lines.get(po["material_number"], set())
        if lines_set & allowed:
            total_qty += int(po["quantity"])
    return total_qty


# ---------- Non-splittable greedy schedule (density + best-fit) w/ start times ----------
def schedule_orders_greedy_nosplit(
    available_lines: List[str],
    production_orders: List[Dict[str, Any]],
    materials: List[Dict[str, Any]],
    shift_start: datetime,
    shift_length_seconds: int,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Returns:
      {
        line: [
          {"order_number": "PO-001", "start": "2025-08-13T06:00:00+00:00"},
          ...
        ],
        ...
      }
    """
    shift_len = int(shift_length_seconds)
    if shift_len <= 0 or not available_lines:
        return {ln: [] for ln in available_lines}

    mat_to_lines = {m["material_number"]: set(m.get("line", [])) for m in materials}

    def eligible_lines_for_order(po) -> List[str]:
        return sorted(set(available_lines) & mat_to_lines.get(po["material_number"], set()))

    # Enrich + filter by eligibility
    enriched = []
    for po in production_orders:
        q = int(po["quantity"])
        tpu = float(po["time_to_complete"])
        dur = int(q * tpu)
        elig = eligible_lines_for_order(po)
        if not elig:
            continue
        enriched.append({
            **po,
            "duration_s": dur,
            "eligible_lines": elig,
            "density": (1.0 / tpu) if tpu > 0 else float("inf"),
        })

    # Sort: higher density first, then shorter duration, then order_number
    enriched.sort(key=lambda x: (-x["density"], x["duration_s"], x["order_number"]))

    # Line state
    line_time_used = {ln: 0 for ln in available_lines}
    per_line_sequence: Dict[str, List[Dict[str, str]]] = {ln: [] for ln in available_lines}

    for job in enriched:
        dur = job["duration_s"]
        # lines that can fit the whole job
        candidates = [ln for ln in job["eligible_lines"] if line_time_used[ln] + dur <= shift_len]
        if not candidates:
            continue
        # Choose line with MOST remaining capacity
        best_line = max(candidates, key=lambda ln: (shift_len - line_time_used[ln], -line_time_used[ln], ln))
        # Start time = shift_start + used_time on that line
        start_dt = shift_start + timedelta(seconds=line_time_used[best_line])
        per_line_sequence[best_line].append({
            "order_number": job["order_number"],
            "start": start_dt.isoformat()
        })
        line_time_used[best_line] += dur

    return per_line_sequence


# ---------- HTTP-trigger entry ----------
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Expected JSON body:
    {
      "lines_available": [...],
      "employees_available": [{"id": "...","qualifiedLines":[...]}],
      "production_orders": [{"order_number":"...", "material_number":"...", "quantity":int, "time_to_complete":seconds}, ...],
      "materials": [{"material_number":"...", "line":[...]}],
      "shift_start": "2025-08-13T06:00:00Z",   # ISO 8601 (optional; default now, UTC)
      "shift_length_seconds": 28800            # e.g., 8*3600
    }
    """
    try:
        body = req.get_json()
    except Exception:
        return func.HttpResponse("Invalid or missing JSON body.", status_code=400)

    lines_available = body.get("lines_available", [])
    employees_available = body.get("employees_available", [])
    production_orders = body.get("production_orders", [])
    materials = body.get("materials", [])
    shift_start_raw = body.get("shift_start")
    shift_length_seconds = int(body.get("shift_length_seconds", 0))

    # Parse shift_start
    try:
        if shift_start_raw:
            # Allow 'Z'
            shift_start = datetime.fromisoformat(shift_start_raw.replace("Z", "+00:00"))
        else:
            shift_start = datetime.utcnow()
    except Exception:
        return func.HttpResponse("Invalid 'shift_start' format. Use ISO 8601.", status_code=400)

    # 1) All maximum-cardinality assignments
    assignments = employee_assignment(lines_available, employees_available)

    if not assignments:
        resp = {"shift_start": shift_start.isoformat(), "lines": [], "assignment": {}, "sequence": {}}
        return func.HttpResponse(json.dumps(resp), status_code=200, mimetype="application/json")

    # Helper to extract lines from combo
    def lines_from_combo(combo: List[Dict[str, str]]) -> List[str]:
        return [next(iter(d.keys())) for d in combo]

    # 2) Pick assignment maximizing theoretical producible quantity
    best_idx = 0
    best_qty = -1
    for i, combo in enumerate(assignments):
        lines_combo = lines_from_combo(combo)
        qty = score_assignment_quantity(lines_combo, production_orders, materials)
        if qty > best_qty:
            best_qty = qty
            best_idx = i

    chosen = assignments[best_idx]
    chosen_lines = lines_from_combo(chosen)
    assignment_map = {list(d.keys())[0]: list(d.values())[0] for d in chosen}

    # 3) Build non-splittable greedy schedule sequence per line (with order start times)
    sequence = schedule_orders_greedy_nosplit(
        available_lines=chosen_lines,
        production_orders=production_orders,
        materials=materials,
        shift_start=shift_start,
        shift_length_seconds=shift_length_seconds,
    )

    # 4) Simple response (including shift_start and order start times)
    resp = {
        "shift_start": shift_start.isoformat(),
        "lines": sorted(chosen_lines),
        "assignment": assignment_map,
        "sequence": sequence
    }
    return func.HttpResponse(json.dumps(resp), status_code=200, mimetype="application/json")
