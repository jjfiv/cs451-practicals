# CS451-practicals
Practical Assignments (just starter code) for [CS451: Machine Learning](https://jjfiv.github.io/cs451-s2021/)

## To Fork or not to fork?

I am not bothered if you fork and push your solutions to github; none of these practicals are supposed to be overly complex.

I recommend it, actually for most setups.

## Opening in repl.it

The repl.it link to this repo is here:
https://repl.it/@jjfoley/cs451-practicals

... although you may want to fork it first on github and open your link instead, if you want your changes to exist on github outside of repl.it: repl.it has no way to adjust the git references later, AFAICT.

## Setting up a virtualenv: (MacOS / \*nix)

```bash
python3 -m venv venv
source venv/bin/activate
pip install black # the formatter repl.it and I really like
```

## Setting up in vscode

I recommend the following plugins:
  - Python
  - Pyright (static type checker for python)
  - Error Lens (the one with the most stars: ``usernamehw.errorlens``)
