import unittest


class Stage12AdapterTest(unittest.TestCase):
    def test_layer_fraction_resolution(self):
        from src.modeling.dynamic_grounded_llm import layer_indices_from_fractions

        self.assertEqual(layer_indices_from_fractions(8, [0.25, 0.5, 1.0]), [1, 3, 7])

    def test_adapter_wrapper_is_identity_then_receives_gradients(self):
        try:
            import torch
            import torch.nn as nn
            from src.modeling.dynamic_grounded_llm import DynamicGroundedCausalLM
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

            def forward(self, hidden):
                return self.linear(hidden)

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block(), Block()])

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"hidden_size": 16})()
                self.model = Backbone()

            def forward(self, hidden):
                for layer in self.model.layers:
                    hidden = layer(hidden)
                return hidden

        torch.manual_seed(3)
        base = FakeLM()
        hidden = torch.randn(1, 5, 16)
        expected = base(hidden).detach()
        wrapped = DynamicGroundedCausalLM(
            base, grounding_dim=8, layer_indices=[1], num_heads=4
        )
        wrapped.freeze_base_model()
        wrapped.set_grounding_context(
            [torch.randn(4, 8)], [torch.randn(8)], torch.tensor([-1, 0, 0, 0, -1])
        )
        actual = wrapped(hidden)
        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        adapter = wrapped.adapters["1"]
        self.assertIsNotNone(adapter.cross_scale.grad)
        self.assertFalse(base.model.layers[0].linear.weight.requires_grad)

    def test_teacher_forcing_alignment_changes_at_clause_boundaries(self):
        from src.data.stage12_llm_grounding_data import teacher_forcing_token_steps

        class Tokenizer:
            @staticmethod
            def decode(ids, skip_special_tokens=True):
                del skip_special_tokens
                return " ".join(ids)

        sql = ["SELECT", "name", "FROM", "users", "WHERE", "id", "=", "1"]
        steps = [
            {"operation": "PROJECT"},
            {"operation": "JOIN"},
            {"operation": "FILTER"},
        ]
        aligned = teacher_forcing_token_steps(Tokenizer(), ["p0", "p1"], sql, steps)
        # Hidden at the final prompt position predicts SELECT using PROJECT.
        self.assertEqual(aligned[1], 0)
        self.assertIn(1, aligned)
        self.assertIn(2, aligned)


if __name__ == "__main__":
    unittest.main()
