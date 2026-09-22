import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from kvpress import KnormPress
from kvpress.indexmem import IndexMemConfig, IndexMemTextGenerationPipeline


def test_general_kvpress_compression_still_works():
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    ).eval()
    model.config._attn_implementation = "sdpa"
    cache = DynamicCache()
    with torch.inference_mode(), KnormPress(compression_ratio=0.5)(model):
        model(torch.arange(16).view(1, 16), past_key_values=cache)
    assert cache.layers[0].keys.shape[2] == 8
    assert not model.model.layers[0].self_attn._forward_hooks


def test_indexmem_pipeline_has_its_own_registration():
    from transformers.pipelines import PIPELINE_REGISTRY

    assert IndexMemConfig().inference_mode == "evict"
    assert PIPELINE_REGISTRY.supported_tasks["indexmem-text-generation"]["impl"] is IndexMemTextGenerationPipeline
    assert PIPELINE_REGISTRY.supported_tasks["kv-press-text-generation"]["impl"] is not IndexMemTextGenerationPipeline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pipeline")
@pytest.mark.parametrize("mode", ["mask", "evict"])
def test_public_pipeline_reuses_context_for_multiple_questions(tmp_path, mode):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from kvpress.indexmem.checkpoint import attach_scorers, save_scorer_checkpoint

    torch.manual_seed(97)
    vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocabulary.update({f"t{i}": i + 4 for i in range(60)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_max_length=256,
    )
    model_config = Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        eos_token_id=2,
        pad_token_id=0,
    )
    model_config._attn_implementation = "flash_attention_2"
    model = Qwen3ForCausalLM(model_config).to(device="cuda", dtype=torch.bfloat16).eval()
    scorer_config = dict(kind="mlp", hidden_size=64, n_heads=2, mid_dim=16, decay=True, gate_scale=True)
    attach_scorers(model, scorer_config)
    checkpoint = tmp_path / "scorers.pt"
    save_scorer_checkpoint(checkpoint, model, scorer_config)
    pipeline = IndexMemTextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        scorer_checkpoint=checkpoint,
        config=IndexMemConfig(
            cache_budget=16,
            sink_size=2,
            window_size=4,
            cmp_slots=3,
            min_head_budget=0,
            head_budget="uniform",
            inference_mode=mode,
            decode_batch=2,
        ),
    )
    context = " ".join(f"t{i % 60}" for i in range(64))
    answers = pipeline(context, questions=["t1 t2", "t3 t4 t5 t6"], max_new_tokens=3)
    assert len(answers["answers"]) == 2
    assert all(isinstance(answer, str) for answer in answers["answers"])
    assert isinstance(pipeline(context, question="t1", max_new_tokens=2)["answer"], str)
    assert model.config._attn_implementation == "flash_attention_2"
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)
