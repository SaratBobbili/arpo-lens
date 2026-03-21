We will be listing all our tasks to run the ECHO algorithm

**MAIN STEPS**
1. Modify the SFT training data to https://huggingface.co/datasets/dongguanting/ARPO-SFT-54K,
follow the tool trajectory template we want.


2. SFT-fine-tuning stage.

The SFT fine-tuning stage should enable trajectory stabilization for the model such that the model selects the tools to be used for generating rollouts after the prompt. Then after each reasoning turn, there will be a tool selection rationale that is generated.
