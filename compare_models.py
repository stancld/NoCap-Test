"""Compare model parameters between baseline and QwenGPT."""

from model import GPT, QwenGPT
from training_configurations import MODEL_CONFIGS


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def format_params(num_params):
    """Format parameter count in human readable form."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    if num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def main():
    print("Model Parameter Comparison")
    print("=" * 50)

    # Compare baseline and qwen models
    models_to_compare = ["baseline", "qwen"]

    for model_name in models_to_compare:
        config = MODEL_CONFIGS[model_name]

        # Create model based on config type
        model = QwenGPT(config) if model_name == "qwen" else GPT(config)

        param_count = count_parameters(model)
        formatted_count = format_params(param_count)

        print(f"{model_name:15}: {param_count:>12,} ({formatted_count})")

        # For QwenGPT, show breakdown by layer type
        if model_name == "qwen":
            print("  Layer breakdown:")
            attention_params = 0
            linear_params = 0

            for i, block in enumerate(model.transformer.h):
                block_params = count_parameters(block.mixer)
                if block.use_attention:
                    attention_params += block_params
                    print(f"    Layer {i:2d} (attention): {format_params(block_params)}")
                else:
                    linear_params += block_params
                    print(f"    Layer {i:2d} (linear):    {format_params(block_params)}")

            print(f"  Total attention layers: {format_params(attention_params)}")
            print(f"  Total linear layers:    {format_params(linear_params)}")
            print(
                f"  Other components:       {format_params(param_count - attention_params - linear_params)}"
            )

    print("=" * 50)

    # Calculate difference
    baseline_params = count_parameters(GPT(MODEL_CONFIGS["baseline"]))
    qwen_params = count_parameters(QwenGPT(MODEL_CONFIGS["qwen"]))
    diff = qwen_params - baseline_params
    diff_pct = (diff / baseline_params) * 100

    print(f"Difference: {diff:+,} parameters ({diff_pct:+.2f}%)")


if __name__ == "__main__":
    main()
