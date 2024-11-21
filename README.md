# tiny-blimp-eval

Notes: 

The checkpoints stored on HF do not include a checkpoint-19000. For now, I am hard-coding to skip that checkpoint when evaluating. Enyan, once you're back, maybe you could look into why 19000 isn't there...

For cleanest implementation so far, reference the .py file in the evaluate branch, not this main branch's contents.

The main branch does have the evaluation results in eval.ipynb for now, but I think the other branch will serve us better moving forward. Once I confirm that the "evaluate" branch is accurately storing the data, I will probably delete the jiant and .ipynb stuff from this branch and then merge the branches. Could be useful to keep the data, links or copies to Enyan's repo and hf (as well as any future ones), even though they'll be explictly used in the .py file

Tracking results: https://wandb.ai/ling380/tiny-blimp

