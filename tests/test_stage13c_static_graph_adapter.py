import unittest


class Stage13CStaticGraphAdapterTest(unittest.TestCase):
    def test_memory_projector_normalizes_input_dtype(self):
        try:
            import torch
            from src.modeling.static_graph_adapter import GraphMemoryProjector
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        projector = GraphMemoryProjector(
            graph_dim=8, semantic_dim=12, query_dim=12, llm_dim=16
        )
        memory = torch.randn(4, 8, dtype=torch.bfloat16)
        semantics = torch.randn(4, 12, dtype=torch.bfloat16)
        query = torch.randn(1, 12, dtype=torch.bfloat16)
        projected = projector(memory, semantics, query)
        self.assertEqual(projected.dtype, projector.semantic_norm.weight.dtype)
        self.assertEqual(tuple(projected.shape), (4, 16))

    def test_semantic_path_survives_graph_corruption(self):
        try:
            import torch
            from src.modeling.static_graph_adapter import GraphMemoryProjector
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        torch.manual_seed(17)
        projector = GraphMemoryProjector(
            graph_dim=8, semantic_dim=12, query_dim=12, llm_dim=16
        )
        semantics = torch.randn(4, 12)
        query = torch.randn(1, 12)
        first = projector(torch.randn(4, 8), semantics, query, return_components=True)
        second = projector(torch.randn(4, 8), semantics, query, return_components=True)
        self.assertTrue(torch.allclose(first["semantic_memory"], second["semantic_memory"]))
        self.assertFalse(torch.allclose(first["memory"], second["memory"]))

    def test_cross_adapter_uses_fp32_parameters_and_bounded_context(self):
        try:
            import torch
            from src.modeling.static_graph_adapter import StaticGraphCrossAdapter
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        adapter = StaticGraphCrossAdapter(llm_dim=16, graph_dim=8, num_heads=4)
        hidden = torch.randn(1, 3, 16, dtype=torch.bfloat16)
        memory = torch.randn(5, 8)
        output = adapter(hidden, memory)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(adapter.norm.weight.dtype, torch.float32)
        self.assertLess(adapter.last_diagnostics["mean_update_ratio"], 0.1)

    def test_runtime_accepts_legacy_index_cache(self):
        try:
            import numpy as np
            import torch
            from src.modeling.stage13c_static_runtime import graph_tensors
        except ImportError:
            self.skipTest("NumPy or PyTorch is unavailable")
        example = {
            "metadata": {"record_index": 7},
            "inference_inputs": {
                "schema_nodes": [
                    {"id": 0, "type": "table", "name": "schools"},
                    {"id": 1, "type": "column", "name": "schools.Phone"},
                ],
                "schema_edges": [
                    {"src": 0, "dst": 1, "type": "table_to_column"}
                ],
            },
        }
        cache = {
            "query": np.ones((1, 4), dtype=np.float32),
            "node": np.ones((2, 4), dtype=np.float32),
            "index": {
                7: {
                    "query_embedding_index": 0,
                    "node_embedding_start": 0,
                    "node_count": 2,
                }
            },
        }
        dense, query, node_types, edges, edge_types, nodes = graph_tensors(
            example, cache, {"table_to_column": 0}, torch.device("cpu")
        )
        self.assertEqual(tuple(dense.shape), (2, 4))
        self.assertEqual(tuple(query.shape), (1, 4))
        self.assertEqual(node_types.tolist(), [0, 1])
        self.assertEqual(tuple(edges.shape), (2, 1))
        self.assertEqual(edge_types.tolist(), [0])
        self.assertEqual(len(nodes), 2)

    def test_graph_encoder_changes_when_topology_changes(self):
        try:
            import torch
            from src.modeling.static_graph_adapter import StaticSchemaGraphEncoder
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        torch.manual_seed(7)
        encoder = StaticSchemaGraphEncoder(
            dense_dim=8, hidden_dim=16, relation_count=2, num_layers=2, dropout=0.0
        )
        dense = torch.randn(4, 8)
        query = torch.randn(1, 8)
        node_types = torch.tensor([0, 1, 0, 1])
        edge_type = torch.tensor([0, 1, 0, 1])
        first_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        second_edges = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]])
        first, _ = encoder(dense, query, node_types, first_edges, edge_type)
        second, _ = encoder(dense, query, node_types, second_edges, edge_type)
        self.assertFalse(torch.allclose(first, second))
        loss = encoder.structure_loss(first, first_edges, edge_type)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(encoder.structure_src.weight.grad)

    def test_static_adapter_changes_hidden_and_receives_gradients(self):
        try:
            import torch
            from src.modeling.static_graph_adapter import StaticGraphCrossAdapter
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        torch.manual_seed(11)
        adapter = StaticGraphCrossAdapter(
            llm_dim=16, graph_dim=8, num_heads=4, residual_scale_init=0.1
        )
        hidden = torch.randn(1, 5, 16)
        memory = torch.randn(6, 8, requires_grad=True)
        output = adapter(hidden, memory)
        self.assertFalse(torch.equal(hidden, output))
        output.sum().backward()
        self.assertIsNotNone(memory.grad)
        self.assertIsNotNone(adapter.residual_scale.grad)
        self.assertGreater(adapter.last_diagnostics["mean_update_norm"], 0.0)

    def test_wrapper_has_one_fixed_memory_for_all_tokens(self):
        try:
            import torch
            import torch.nn as nn
            from src.modeling.static_graph_adapter import StaticGraphConditionedCausalLM
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        class Block(nn.Module):
            def __init__(self):
                super().__init__(); self.linear = nn.Linear(16, 16)
            def forward(self, hidden):
                return self.linear(hidden)

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__(); self.layers = nn.ModuleList([Block(), Block()])

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"hidden_size": 16})()
                self.model = Backbone()
            def forward(self, hidden):
                for layer in self.model.layers:
                    hidden = layer(hidden)
                return hidden

        torch.manual_seed(13)
        base = FakeLM()
        hidden = torch.randn(1, 4, 16)
        baseline = base(hidden).detach()
        wrapper = StaticGraphConditionedCausalLM(
            base, graph_dim=8, layer_indices=[1], num_heads=4
        )
        wrapper.freeze_base_model()
        wrapper.set_graph_memory(torch.randn(5, 8))
        conditioned = wrapper(hidden)
        self.assertFalse(torch.equal(baseline, conditioned))
        self.assertFalse(base.model.layers[0].linear.weight.requires_grad)
        self.assertTrue(wrapper.adapters["1"].query.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
