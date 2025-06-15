import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
import json
import re
import gc


class BERTHandler:
    """
    VRAM-safe BERT model handler for loading, tokenization, and saving
    Handles all token management and checkpoint operations with proper cleanup
    """

    def __init__(self, symbolic_tokens=None):
        # Default symbolic tokens
        self.symbolic_tokens = symbolic_tokens or [
            "<subject>", "<subject1>", "<subject2>", "<pose>", "<emotion>",
            "<surface>", "<lighting>", "<material>", "<accessory>", "<footwear>",
            "<upper_body_clothing>", "<hair_style>", "<hair_length>", "<headwear>",
            "<texture>", "<pattern>", "<grid>", "<zone>", "<offset>",
            "<object_left>", "<object_right>", "<relation>", "<intent>", "<style>",
            "<fabric>", "<jewelry>"
        ]

        # Generate shunt tokens
        self.shunt_tokens = [f"[SHUNT_{1000000 + i}]" for i in range(len(self.symbolic_tokens))]
        self.all_special_tokens = self.symbolic_tokens + self.shunt_tokens

        # Model components
        self.tokenizer = None
        self.model = None
        self.current_step = 0
        self.current_epoch = 1

        print(f"🎯 BERTHandler initialized with {len(self.all_special_tokens)} special tokens")

    def __del__(self):
        """Destructor to ensure cleanup when object is deleted"""
        try:
            self._cleanup_model()
        except Exception:
            # Ignore cleanup errors during shutdown
            pass

    def _cleanup_model(self):
        """
        CRITICAL: Comprehensive model cleanup to free VRAM
        This is the core method that prevents VRAM accumulation
        """
        if hasattr(self, 'model') and self.model is not None:
            print("🧹 Cleaning up existing model from VRAM...")

            # Check if torch is still available (can be None during shutdown)
            try:
                import torch as torch_module
                if torch_module is None:
                    return
            except (ImportError, AttributeError):
                return

            # Move model to CPU first to free GPU memory
            try:
                if torch_module.cuda.is_available() and next(self.model.parameters(), None) is not None:
                    if next(self.model.parameters()).is_cuda:
                        self.model = self.model.cpu()
            except Exception:
                # Continue cleanup even if moving to CPU fails
                pass

            # Delete the model
            try:
                del self.model
                self.model = None
            except Exception:
                pass

            # Force garbage collection
            try:
                gc.collect()
            except Exception:
                pass

            # Clear CUDA cache
            try:
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                    torch_module.cuda.synchronize()  # Ensure all CUDA operations complete
            except Exception:
                pass

            print("✅ Model cleanup complete")

    def _print_vram_usage(self, prefix=""):
        """Print current VRAM usage for monitoring"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"🎯 {prefix}VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            else:
                print(f"🎯 {prefix}CUDA not available")
        except Exception:
            print(f"🎯 {prefix}VRAM: Could not read CUDA memory")

    def load_fresh_model(self, model_name="nomic-ai/nomic-bert-2048"):
        """Load fresh model and add special tokens with proper VRAM management"""
        print(f"🆕 Loading fresh model: {model_name}")
        self._print_vram_usage("Before cleanup: ")

        # CRITICAL: Clean up existing model first
        self._cleanup_model()
        self._print_vram_usage("After cleanup: ")

        try:
            # Load base model and tokenizer
            print("📥 Loading base tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            print("📥 Loading base model...")
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Explicit dtype for consistency
            )

            # Add special tokens (ONLY for fresh models)
            original_size = len(self.tokenizer)
            special_tokens_dict = {"additional_special_tokens": self.all_special_tokens}
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)

            print(f"   - Original vocab size: {original_size}")
            print(f"   - Added {num_added} special tokens")
            print(f"   - New vocab size: {len(self.tokenizer)}")

            # Resize model embeddings (ONLY for fresh models)
            if num_added > 0:
                self._resize_embeddings()

            # Reset training state
            self.current_step = 0
            self.current_epoch = 1

            print("✅ Fresh model loaded successfully")
            self._print_vram_usage("After loading: ")
            return self.model, self.tokenizer

        except Exception as e:
            print(f"❌ Failed to load fresh model: {e}")
            # Clean up on failure
            self._cleanup_model()
            raise

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint - use saved tokenizer as-is, no modifications"""
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        self._print_vram_usage("Before cleanup: ")

        # CRITICAL: Clean up existing model first
        self._cleanup_model()
        self._print_vram_usage("After cleanup: ")

        try:
            # Load saved tokenizer AS-IS (already contains special tokens)
            print("📥 Loading saved tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            print(f"   - Tokenizer loaded: {len(self.tokenizer)} tokens (already includes special tokens)")

            # Load saved model AS-IS (already matches tokenizer)
            print("📥 Loading saved model...")
            self.model = AutoModelForMaskedLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

            print(f"✅ Model loaded successfully")
            print(f"   - Model vocab size: {self.model.config.vocab_size}")
            print(f"   - Embedding size: {self.model.bert.embeddings.word_embeddings.weight.shape[0]}")
            print(f"   - Tokenizer size: {len(self.tokenizer)}")

            # Check for vocab size mismatch and warn (but don't auto-fix for checkpoints)
            tokenizer_size = len(self.tokenizer)
            model_vocab_size = self.model.config.vocab_size
            embedding_size = self.model.bert.embeddings.word_embeddings.weight.shape[0]

            if not (tokenizer_size == model_vocab_size == embedding_size):
                print(f"⚠️  VOCAB SIZE MISMATCH DETECTED:")
                print(f"   - Tokenizer size: {tokenizer_size}")
                print(f"   - Model config vocab_size: {model_vocab_size}")
                print(f"   - Embedding size: {embedding_size}")
                print(f"   This might affect inference quality.")

            # Load training state
            self._load_training_state(checkpoint_path)

            print(f"✅ Checkpoint loaded - Step: {self.current_step}, Epoch: {self.current_epoch}")
            self._print_vram_usage("After loading: ")
            return self.model, self.tokenizer

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            # Clean up on failure
            self._cleanup_model()
            raise

    def save_checkpoint(self, save_path, step=None, epoch=None):
        """Save model checkpoint with consistency verification"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded to save")

        step = step or self.current_step
        epoch = epoch or self.current_epoch

        # CRITICAL: Verify consistency before saving
        tokenizer_size = len(self.tokenizer)
        model_vocab_size = self.model.config.vocab_size
        embedding_size = self.model.bert.embeddings.word_embeddings.weight.shape[0]

        if not (tokenizer_size == model_vocab_size == embedding_size):
            print(f"⚠️  CONSISTENCY CHECK FAILED before saving:")
            print(f"   - Tokenizer size: {tokenizer_size}")
            print(f"   - Model config vocab_size: {model_vocab_size}")
            print(f"   - Embedding size: {embedding_size}")

            # Force consistency before saving
            print(f"🔧 Forcing consistency to tokenizer size: {tokenizer_size}")
            self.model.config.vocab_size = tokenizer_size

            # Resize embeddings if needed
            if embedding_size != tokenizer_size:
                print(f"🔧 Resizing embeddings to match tokenizer: {embedding_size} → {tokenizer_size}")
                self._resize_embeddings()

        # Create checkpoint directory
        checkpoint_dir = Path(save_path) / f"symbolic_bert_step{step}_epoch{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving checkpoint: {checkpoint_dir}")

        try:
            # Save model and tokenizer
            print("💾 Saving model...")
            self.model.save_pretrained(checkpoint_dir)

            print("💾 Saving tokenizer...")
            self.tokenizer.save_pretrained(checkpoint_dir)

            # Save training state with consistency info
            training_state = {
                "step": step,
                "epoch": epoch,
                "vocab_size": len(self.tokenizer),
                "model_vocab_size": self.model.config.vocab_size,
                "embedding_size": self.model.bert.embeddings.word_embeddings.weight.shape[0],
                "consistency_verified": True,
                "special_tokens_count": len(self.all_special_tokens)
            }

            with open(checkpoint_dir / "training_config.json", "w") as f:
                json.dump(training_state, f, indent=2)

            # Save token mappings
            self._save_token_mappings(checkpoint_dir)

            # VERIFICATION: Load and check consistency
            print("🔍 Verifying saved checkpoint consistency...")
            test_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            test_config_path = checkpoint_dir / "config.json"

            with open(test_config_path) as f:
                test_config = json.load(f)

            saved_tokenizer_size = len(test_tokenizer)
            saved_model_vocab = test_config["vocab_size"]

            if saved_tokenizer_size != saved_model_vocab:
                raise RuntimeError(
                    f"CHECKPOINT SAVE FAILED! Inconsistency detected:\n"
                    f"  Saved tokenizer size: {saved_tokenizer_size}\n"
                    f"  Saved model vocab: {saved_model_vocab}"
                )

            # Update internal state
            self.current_step = step
            self.current_epoch = epoch

            print(f"✅ Checkpoint saved and verified successfully")
            print(f"   - Consistent vocab size: {saved_tokenizer_size}")
            return checkpoint_dir

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            raise

    def find_latest_checkpoint(self, base_path, pattern="symbolic_bert"):
        """Find latest checkpoint in directory"""
        path = Path(base_path)
        if not path.exists():
            print(f"⚠️  Checkpoint directory does not exist: {base_path}")
            return None

        # Find checkpoints
        checkpoints = list(path.glob(f"{pattern}_step*_epoch*"))
        if not checkpoints:
            print(f"⚠️  No checkpoints found in {base_path}")
            return None

        # Sort by step number (more reliable than modification time)
        def extract_step(checkpoint_path):
            match = re.search(r"step(\d+)", checkpoint_path.name)
            return int(match.group(1)) if match else 0

        checkpoints.sort(key=extract_step, reverse=True)
        latest = checkpoints[0]

        print(f"📂 Found latest checkpoint: {latest}")
        return latest

    def get_token_mappings(self):
        """Get token ID mappings"""
        if self.tokenizer is None:
            return {}, {}

        symbolic_ids = {}
        shunt_ids = {}

        for token in self.symbolic_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                symbolic_ids[token] = token_id

        for token in self.shunt_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                shunt_ids[token] = token_id

        return symbolic_ids, shunt_ids

    def to_device(self, device):
        """Move model to device with VRAM monitoring"""
        if self.model is not None:
            print(f"📱 Moving model to {device}...")
            self._print_vram_usage("Before device move: ")

            self.model = self.model.to(device)

            # Clear cache after moving to device
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"✅ Model moved to {device}")
            self._print_vram_usage("After device move: ")
        else:
            print(f"⚠️  No model loaded to move to {device}")
        return self

    def _resize_embeddings(self):
        """Resize model embeddings to match tokenizer (handles both expansion and shrinking)"""
        if self.model is None:
            raise RuntimeError("No model loaded")

        old_embeddings = self.model.bert.embeddings.word_embeddings
        old_size, embedding_dim = old_embeddings.weight.shape
        new_size = len(self.tokenizer)

        if old_size == new_size:
            print(f"✅ Embeddings already correct size: {new_size}")
            return

        print(f"🔄 Resizing embeddings: {old_size} → {new_size}")

        try:
            # Create new embeddings
            new_embeddings = nn.Embedding(new_size, embedding_dim)

            # Copy existing embeddings (handle both expansion and shrinking)
            with torch.no_grad():
                # Copy the minimum of old_size and new_size
                copy_size = min(old_size, new_size)
                new_embeddings.weight.data[:copy_size] = old_embeddings.weight.data[:copy_size].clone()

                # If expanding, initialize new token embeddings
                if new_size > old_size:
                    num_added = new_size - old_size
                    # Use small random initialization for new tokens
                    new_embeddings.weight.data[old_size:] = torch.randn(
                        num_added, embedding_dim, device=old_embeddings.weight.device
                    ) * 0.02
                    print(f"   - Added {num_added} new token embeddings")
                elif new_size < old_size:
                    num_removed = old_size - new_size
                    print(f"   - Removed {num_removed} token embeddings")

            # Replace embeddings
            self.model.bert.embeddings.word_embeddings = new_embeddings

            # Resize decoder if it exists
            if hasattr(self.model.cls.predictions, "decoder"):
                old_decoder = self.model.cls.predictions.decoder
                new_decoder = nn.Linear(embedding_dim, new_size, bias=True)

                with torch.no_grad():
                    # Copy existing weights (handle both expansion and shrinking)
                    copy_size = min(old_decoder.weight.shape[0], new_size)
                    new_decoder.weight.data[:copy_size] = old_decoder.weight.data[:copy_size].clone()

                    # Handle bias
                    if old_decoder.bias is not None:
                        new_decoder.bias.data[:copy_size] = old_decoder.bias.data[:copy_size].clone()

                    # If expanding, tie new decoder weights to new embeddings and init bias
                    if new_size > old_decoder.weight.shape[0]:
                        start_idx = old_decoder.weight.shape[0]
                        new_decoder.weight.data[start_idx:] = new_embeddings.weight.data[start_idx:].clone()
                        if old_decoder.bias is not None:
                            new_decoder.bias.data[start_idx:] = torch.zeros(
                                new_size - start_idx, device=old_decoder.bias.device
                            )

                self.model.cls.predictions.decoder = new_decoder

            # Update config
            self.model.config.vocab_size = new_size

            print(f"✅ Embeddings resized successfully")

        except Exception as e:
            print(f"❌ Failed to resize embeddings: {e}")
            raise

    def _load_training_state(self, checkpoint_path):
        """Load training state from checkpoint"""
        # Try training_config.json first
        config_path = Path(checkpoint_path) / "training_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                self.current_step = config.get("step", 0)
                self.current_epoch = config.get("epoch", 1)
                print(f"📊 Loaded training state: step {self.current_step}, epoch {self.current_epoch}")
                return
            except Exception as e:
                print(f"⚠️  Failed to load training_config.json: {e}")

        # Fallback: extract from path name
        match = re.search(r"step(\d+)_epoch(\d+)", str(checkpoint_path))
        if match:
            self.current_step = int(match.group(1))
            self.current_epoch = int(match.group(2))
            print(f"📊 Extracted training state from path: step {self.current_step}, epoch {self.current_epoch}")
        else:
            self.current_step = 0
            self.current_epoch = 1
            print(f"⚠️  Could not determine training state, using defaults: step 0, epoch 1")

    def _save_token_mappings(self, checkpoint_dir):
        """Save token ID mappings"""
        try:
            symbolic_ids, shunt_ids = self.get_token_mappings()

            token_mappings = {
                "symbolic_token_ids": symbolic_ids,
                "shunt_token_ids": shunt_ids,
                "symbolic_tokens": self.symbolic_tokens,
                "shunt_tokens": self.shunt_tokens,
                "total_special_tokens": len(self.all_special_tokens)
            }

            with open(checkpoint_dir / "special_token_ids.json", "w") as f:
                json.dump(token_mappings, f, indent=2)

            print(f"💾 Saved {len(symbolic_ids)} symbolic and {len(shunt_ids)} shunt token mappings")

        except Exception as e:
            print(f"⚠️  Failed to save token mappings: {e}")

    def summary(self):
        """Print comprehensive handler summary"""
        print(f"\n📋 BERT HANDLER SUMMARY:")

        if self.model is None:
            print("❌ No model loaded")
            return

        symbolic_ids, shunt_ids = self.get_token_mappings()

        print(f"   📚 Tokenizer:")
        print(f"     - Size: {len(self.tokenizer)}")
        print(f"     - Special tokens: {len(self.tokenizer.additional_special_tokens or [])}")

        print(f"   🤖 Model:")
        print(f"     - Config vocab size: {self.model.config.vocab_size}")
        print(f"     - Embedding vocab size: {self.model.bert.embeddings.word_embeddings.weight.shape[0]}")
        print(f"     - Embedding dim: {self.model.bert.embeddings.word_embeddings.weight.shape[1]}")

        if hasattr(self.model.cls.predictions, "decoder"):
            decoder = self.model.cls.predictions.decoder
            print(f"     - Decoder output size: {decoder.weight.shape[0]}")

        print(f"   🎯 Special Tokens:")
        print(f"     - Symbolic tokens mapped: {len(symbolic_ids)}")
        print(f"     - Shunt tokens mapped: {len(shunt_ids)}")
        print(f"     - Total defined: {len(self.all_special_tokens)}")

        print(f"   📊 Training State:")
        print(f"     - Current step: {self.current_step}")
        print(f"     - Current epoch: {self.current_epoch}")

        # VRAM usage
        self._print_vram_usage("   🎯 ")

        # Check for vocab consistency
        tokenizer_size = len(self.tokenizer)
        model_config_size = self.model.config.vocab_size
        embedding_size = self.model.bert.embeddings.word_embeddings.weight.shape[0]

        if tokenizer_size == model_config_size == embedding_size:
            print(f"   ✅ All vocab sizes consistent: {tokenizer_size}")
        else:
            print(f"   ⚠️  Vocab size mismatch detected:")
            print(f"     - Tokenizer: {tokenizer_size}")
            print(f"     - Model config: {model_config_size}")
            print(f"     - Embeddings: {embedding_size}")

    def clear_vram(self):
        """Explicit method to clear VRAM for debugging"""
        print("🧹 Explicit VRAM cleanup requested...")
        self._cleanup_model()
        self._print_vram_usage("After cleanup: ")


# Utility functions for safe usage patterns

def create_handler_with_fresh_model(model_name="nomic-ai/nomic-bert-2048", symbolic_tokens=None):
    """Factory function to create handler and load fresh model safely"""
    print("🔄 Creating new BERTHandler with fresh model...")
    handler = BERTHandler(symbolic_tokens=symbolic_tokens)
    model, tokenizer = handler.load_fresh_model(model_name)
    return handler, model, tokenizer


def create_handler_from_checkpoint(checkpoint_path, symbolic_tokens=None):
    """Factory function to create handler and load from checkpoint safely"""
    print("🔄 Creating new BERTHandler from checkpoint...")
    handler = BERTHandler(symbolic_tokens=symbolic_tokens)
    model, tokenizer = handler.load_checkpoint(checkpoint_path)
    return handler, model, tokenizer


# Usage examples and testing
if __name__ == "__main__":
    # Example usage with comprehensive error handling

    def test_vram_safety():
        """Test VRAM safety by loading multiple models"""
        print("🧪 Testing VRAM safety...")

        handler = BERTHandler()

        # Load model 1
        print("\n--- Loading Model 1 ---")
        handler.load_fresh_model("bert-base-uncased")
        handler.summary()

        # Load model 2 (should clean up model 1)
        print("\n--- Loading Model 2 (should cleanup Model 1) ---")
        handler.load_fresh_model("distilbert-base-uncased")
        handler.summary()

        # Explicit cleanup
        print("\n--- Explicit Cleanup ---")
        handler.clear_vram()

        print("✅ VRAM safety test complete")

    # Uncomment to run test
    # test_vram_safety()

"""
USAGE EXAMPLES:

# Safe way to work with fresh models:
handler, model, tokenizer = create_handler_with_fresh_model("nomic-ai/nomic-bert-2048")

# Safe way to work with checkpoints:
handler, model, tokenizer = create_handler_from_checkpoint("/path/to/checkpoint")

# Manual cleanup when needed:
handler.clear_vram()

# Always check summary for consistency:
handler.summary()
"""