# Modified for Agima: documentation-only path example update.
"""
Utility class for loading and randomly sampling paraphrased instructions.
Used during training to add diversity to task instructions.
"""

import json
import os
import random
from typing import Dict, List, Optional


class InstructionParaphraser:
    """
    Loads paraphrased instructions for VLM Gym tasks and provides random sampling.

    Usage:
        paraphraser = InstructionParaphraser("/path/to/instructions")
        # Get a random paraphrase for a task
        instruction = paraphraser.get_instruction("jigsaw-swap_v5", original_instruction)
    """

    def __init__(
        self,
        instructions_dir: str = "./instructions",
        use_paraphrases: bool = True,
        include_original: bool = True,
    ):
        """
        Args:
            instructions_dir: Directory containing {task_name}.json files
            use_paraphrases: If False, always return the original instruction
            include_original: If True, the original instruction is included in the sampling pool
        """
        self.instructions_dir = instructions_dir
        self.use_paraphrases = use_paraphrases
        self.include_original = include_original
        self._cache: Dict[str, List[str]] = {}
        self._task_mapping = self._build_task_mapping()

    def _build_task_mapping(self) -> Dict[str, str]:
        """
        Build a mapping from data directory names to instruction file names.
        Handles cases where the task name might differ slightly.
        """
        mapping = {}
        if not os.path.exists(self.instructions_dir):
            return mapping

        for filename in os.listdir(self.instructions_dir):
            if filename.endswith('.json'):
                task_name = filename[:-5]  # Remove .json extension
                mapping[task_name] = os.path.join(self.instructions_dir, filename)
                # Also map without _v5 suffix for flexibility
                if task_name.endswith('_v5'):
                    mapping[task_name[:-3]] = os.path.join(self.instructions_dir, filename)
        return mapping

    def _load_instructions(self, task_name: str) -> Optional[List[str]]:
        """Load instructions for a task from cache or disk."""
        if task_name in self._cache:
            return self._cache[task_name]

        # Try to find the instruction file
        filepath = self._task_mapping.get(task_name)
        if filepath is None:
            # Try with _v5 suffix
            filepath = self._task_mapping.get(f"{task_name}_v5")

        if filepath is None or not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            instructions = []
            if self.include_original and 'original' in data:
                instructions.append(data['original'])
            if 'paraphrases' in data:
                instructions.extend(data['paraphrases'])

            self._cache[task_name] = instructions
            return instructions
        except Exception as e:
            print(f"Warning: Failed to load instructions for {task_name}: {e}")
            return None

    def get_instruction(
        self,
        task_name: str,
        original_instruction: str,
        deterministic: bool = False,
    ) -> str:
        """
        Get an instruction for the given task.

        Args:
            task_name: Name of the task (e.g., "jigsaw-swap_v5")
            original_instruction: The original instruction from the JSONL file
            deterministic: If True, always return the first instruction (useful for eval)

        Returns:
            Either a randomly sampled paraphrase or the original instruction
        """
        if not self.use_paraphrases:
            return original_instruction

        instructions = self._load_instructions(task_name)
        if instructions is None or len(instructions) == 0:
            return original_instruction

        if deterministic:
            return instructions[0]

        return random.choice(instructions)

    def get_all_instructions(self, task_name: str) -> Optional[List[str]]:
        """Get all available instructions for a task (including original)."""
        return self._load_instructions(task_name)

    @staticmethod
    def extract_task_name_from_path(data_dir: str) -> str:
        """
        Extract task name from a data directory path.
        E.g., "datasets/VLM_Gym_Episodes_v5/jigsaw-swap_v5/train/" -> "jigsaw-swap_v5"
        """
        parts = data_dir.rstrip('/').split('/')
        for i, part in enumerate(parts):
            if 'VLM_Gym' in part and i + 1 < len(parts):
                return parts[i + 1]
        # Fallback: return second to last non-empty part
        non_empty = [p for p in parts if p]
        if len(non_empty) >= 2:
            return non_empty[-2]
        return ""


# Global singleton for convenience
_default_paraphraser: Optional[InstructionParaphraser] = None


def get_paraphraser(
    instructions_dir: str = "./instructions",
    use_paraphrases: bool = True,
) -> InstructionParaphraser:
    """Get or create a global InstructionParaphraser instance."""
    global _default_paraphraser
    if _default_paraphraser is None:
        _default_paraphraser = InstructionParaphraser(
            instructions_dir=instructions_dir,
            use_paraphrases=use_paraphrases,
        )
    return _default_paraphraser
