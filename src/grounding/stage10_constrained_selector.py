from collections import defaultdict


def constrained_topk(
    example,
    scores,
    top_k=30,
    max_tables=8,
    min_tables=None,
    connectivity_weight=0.10,
    baseline_retention_weight=0.05,
    required_local_ids=None,
):
    """Select a typed, owner-closed schema subset from reranker utilities.

    This is a deterministic greedy decoder for the partition/ownership-constrained
    set objective used in Stage 10-A.  A column and its absent owner table are
    evaluated as one package, so value evidence cannot silently evict the table
    required to interpret that column.
    """
    nodes = example.get("candidate_nodes", [])
    if len(nodes) != len(scores):
        raise ValueError(f"Node/score mismatch: nodes={len(nodes)} scores={len(scores)}")
    top_k = min(int(top_k), len(nodes))
    if top_k <= 0:
        return [], {"selected_table_count": 0, "owner_closure_additions": 0}

    adjacency = defaultdict(set)
    for edge in example.get("schema_edges", []):
        src = int(edge["src"])
        dst = int(edge["dst"])
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    baseline_ids = set(int(item_id) for item_id in example.get("baseline_selected_ids", []))
    local_by_schema_id = {
        int(node["schema_item_id"]): int(node["local_id"]) for node in nodes
    }
    table_locals = [
        int(node["local_id"]) for node in nodes if node.get("type") == "table"
    ]
    baseline_table_count = sum(
        1
        for node in nodes
        if node.get("type") == "table" and int(node["schema_item_id"]) in baseline_ids
    )
    if min_tables is None:
        min_tables = min(baseline_table_count, max_tables, top_k)
    min_tables = min(int(min_tables), len(table_locals), int(max_tables), top_k)

    def unary(local_id):
        node = nodes[local_id]
        retention = (
            baseline_retention_weight
            if int(node["schema_item_id"]) in baseline_ids
            else 0.0
        )
        return float(scores[local_id]) + retention

    required = {int(local_id) for local_id in (required_local_ids or [])}
    if any(local_id < 0 or local_id >= len(nodes) for local_id in required):
        raise ValueError("required_local_ids contains an out-of-range local id")

    # A forced feasible completion is used only during structured training. Gold
    # columns bring their owner tables with them, exactly as ordinary decoding does.
    forced_owner_additions = 0
    for local_id in list(required):
        node = nodes[local_id]
        if node.get("type") != "column":
            continue
        owner_schema_id = node.get("owner_table_id")
        owner_local = (
            local_by_schema_id.get(int(owner_schema_id))
            if owner_schema_id is not None
            else None
        )
        if owner_local is not None and owner_local not in required:
            required.add(owner_local)
            forced_owner_additions += 1

    required_table_count = sum(
        1 for local_id in required if nodes[local_id].get("type") == "table"
    )
    if len(required) > top_k or required_table_count > max_tables:
        return [], {
            "selected_table_count": 0,
            "selected_column_count": 0,
            "owner_closure_additions": 0,
            "forced_owner_additions": forced_owner_additions,
            "required_count": len(required),
            "required_feasible": False,
            "minimum_table_count": min_tables,
            "max_tables": max_tables,
            "selected_count": 0,
        }

    selected = set(required)
    if sum(1 for index in selected if nodes[index].get("type") == "table") < min_tables:
        for local_id in sorted(table_locals, key=unary, reverse=True):
            if local_id in selected:
                continue
            if len(selected) >= top_k:
                break
            selected.add(local_id)
            if sum(
                1 for index in selected if nodes[index].get("type") == "table"
            ) >= min_tables:
                break
    owner_closure_additions = 0

    def table_count_after(package):
        return sum(1 for index in selected | package if nodes[index].get("type") == "table")

    while len(selected) < top_k:
        best = None
        for local_id, node in enumerate(nodes):
            if local_id in selected:
                continue
            package = {local_id}
            owner_added = False
            if node.get("type") == "column":
                owner_schema_id = node.get("owner_table_id")
                owner_local = local_by_schema_id.get(int(owner_schema_id)) if owner_schema_id is not None else None
                if owner_local is not None and owner_local not in selected:
                    package.add(owner_local)
                    owner_added = True
            if len(selected) + len(package) > top_k:
                continue
            if table_count_after(package) > max_tables:
                continue
            utility = sum(unary(index) for index in package)
            connected_edges = sum(
                1
                for index in package
                for neighbor in adjacency.get(index, set())
                if neighbor in selected or neighbor in package
            )
            utility += connectivity_weight * connected_edges
            density = utility / len(package)
            key = (density, utility, -len(package), -local_id)
            if best is None or key > best[0]:
                best = (key, package, owner_added)
        if best is None:
            # Ownership/max-table constraints can leave one residual slot. Fill it
            # only with an individually feasible node.
            feasible = [
                index
                for index, node in enumerate(nodes)
                if index not in selected
                and not (
                    node.get("type") == "table"
                    and table_count_after({index}) > max_tables
                )
                and not (
                    node.get("type") == "column"
                    and node.get("owner_local_id") is not None
                    and int(node["owner_local_id"]) not in selected
                )
            ]
            if not feasible:
                break
            selected.add(max(feasible, key=unary))
            continue
        selected.update(best[1])
        owner_closure_additions += int(best[2])

    ranked_selected = sorted(selected, key=unary, reverse=True)[:top_k]
    return ranked_selected, {
        "selected_table_count": sum(
            1 for index in ranked_selected if nodes[index].get("type") == "table"
        ),
        "selected_column_count": sum(
            1 for index in ranked_selected if nodes[index].get("type") == "column"
        ),
        "owner_closure_additions": owner_closure_additions,
        "forced_owner_additions": forced_owner_additions,
        "required_count": len(required),
        "required_feasible": required.issubset(set(ranked_selected)),
        "minimum_table_count": min_tables,
        "max_tables": max_tables,
        "selected_count": len(ranked_selected),
    }
