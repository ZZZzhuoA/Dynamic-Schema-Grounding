import unittest


class Stage12AdapterTest(unittest.TestCase):
    def test_teacher_forcing_role_tensor_matches_input_shape(self):
        try:
            import torch
            from src.training.stage12_train_dynamic_grounded_llm import encode_teacher_forcing
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        class CharacterTokenizer:
            eos_token_id = 999

            @staticmethod
            def apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
                del tokenize, add_generation_prompt, kwargs
                return "\n".join(item["content"] for item in messages)

            @staticmethod
            def __call__(text, add_special_tokens=False, return_offsets_mapping=False):
                del add_special_tokens
                result = {"input_ids": [ord(char) for char in text]}
                if return_offsets_mapping:
                    result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
                return result

            @staticmethod
            def decode(ids, skip_special_tokens=True):
                del skip_special_tokens
                return "".join(chr(value) for value in ids if value != 999)

        encoded = encode_teacher_forcing(
            CharacterTokenizer(), "question", "SELECT schools.Phone FROM schools",
            [{"operation": "PROJECT"}, {"operation": "JOIN"}],
            {"inference_inputs": {"schema_nodes": [
                {"type": "table", "name": "schools"},
                {"type": "column", "table": "schools", "column": "Phone"},
            ]}},
            1024, torch.device("cpu"), torch,
        )
        input_ids, _labels, token_steps, token_roles = encoded
        self.assertEqual(tuple(token_roles.shape), tuple(input_ids.shape))
        self.assertEqual(token_steps.numel(), input_ids.shape[1])

    def test_sql_token_roles_mark_schema_operator_and_value(self):
        from src.data.stage12_llm_grounding_data import (
            TOKEN_ROLE_BASE,
            TOKEN_ROLE_OPERATOR,
            TOKEN_ROLE_SCHEMA,
            TOKEN_ROLE_VALUE,
            sql_token_roles,
        )

        class CharacterTokenizer:
            eos_token_id = 999

            @staticmethod
            def __call__(text, add_special_tokens=False, return_offsets_mapping=False):
                del add_special_tokens
                result = {"input_ids": [ord(char) for char in text]}
                if return_offsets_mapping:
                    result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
                return result

        sql = "SELECT schools.Phone FROM schools WHERE schools.Kind = 'Direct' ORDER BY schools.Phone"
        graph = {
            "inference_inputs": {
                "schema_nodes": [
                    {"type": "table", "name": "schools"},
                    {"type": "column", "table": "schools", "column": "Phone"},
                    {"type": "column", "table": "schools", "column": "Kind"},
                ]
            }
        }
        _, roles = sql_token_roles(CharacterTokenizer(), sql, graph)
        self.assertEqual(roles[sql.index("schools.Phone")], TOKEN_ROLE_SCHEMA)
        self.assertEqual(roles[sql.index("ORDER BY")], TOKEN_ROLE_OPERATOR)
        self.assertEqual(roles[sql.index("'Direct'")], TOKEN_ROLE_VALUE)
        self.assertEqual(roles[sql.index("SELECT")], TOKEN_ROLE_BASE)

    def test_semantic_utility_rewards_key_logprob_gain(self):
        try:
            import torch
            from src.training.stage12_train_dynamic_grounded_llm import (
                semantic_utility_objective,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        normal = torch.tensor([[0.5, 0.7, 0.4, 0.2]], requires_grad=True)
        zero = torch.tensor([[0.8, 0.6, 0.9, 0.2]])
        valid = torch.ones_like(normal, dtype=torch.bool)
        # The first entry is shifted away; target roles are schema/operator/value/base.
        roles = torch.tensor([[0, 1, 2, 3, 0]])
        objective, metrics = semantic_utility_objective(
            normal, zero, valid, roles,
            {0: 0.5, 1: 4.0, 2: 2.5, 3: 3.0},
            counterfactual_weight=0.5,
            counterfactual_margin=0.05,
            preservation_weight=0.1,
            torch=torch,
        )
        self.assertGreater(float(metrics["mean_key_logprob_gain"].detach()), 0.0)
        self.assertEqual(int(metrics["key_token_count"]), 3)
        objective.backward()
        self.assertIsNotNone(normal.grad)

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
        scales = wrapped.adapter_scale_summary()
        self.assertEqual(scales["1"]["cross_scale_raw"], 0.0)
        self.assertEqual(scales["1"]["steering_scale_raw"], 0.0)
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
