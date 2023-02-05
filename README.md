# Tree Edit Distance - Python Reference
These little code snippets aim to be easy to follow implementations of the core ideas proposed by various authors on the Tree Edit Distance problem.

Sometime, I find it personally easier to understand an algorithm or its paper by reading a carefully coded and well documented version of it, and then referencing that back to the original paper. Or if I'm lucky I'll find a more recent, simplified blog post.

The code is written in modern python and in a style more conducive to a healthy, maintainable library. The only deviation from this is that it probably has way too many comments.

## Dabbling
To dabble with this code, I recommend using the `environment.yml` file to create a conda env:
```bash
conda env create --file environment.yml
```

If you make any changes, you can update the environment with:
```bash
conda env update --file environment.yml --prune
```

Active the enviroment with:
```bash
conda active ted-kitizz
```

And run the tests with:
```bash
pytest
```

Note also that it is all set up and ready to go in VSCode if that's your IDE of choice.