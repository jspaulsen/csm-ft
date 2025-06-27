# Training

Refer to the train.py script; basically, you'll just need to swap your datasets and change any filters you might want. By default, we filter out samples > 30 and < 1.2 seconds.

This does expect wandb, so you'll need to disable it if you're not using it.

>             report_to="wandb",
change to
>             report_to=None,
