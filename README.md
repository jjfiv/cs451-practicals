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
pip install -U pip # update pip
pip install black # the formatter repl.it and I really like
pip install -r requirements.txt
```

## Setting up a virtualenv: Windows 10

Install Python from https://www.python.org/downloads/windows/
Make sure you allow it to edit your path and install the 'py' launcher.

In POWERSHELL:

```pwsh
py -m venv venv
& venv/Scripts/Activate.ps1 # may need to change permissions
pip.exe install -U pip # update pip
pip.exe install black # the formatter repl.it and I really like
pip.exe install -r requirements.txt
```

Note that pip is a script on Windows which likely points to your global python and not the local virtualenv we just created.

I had to change the execution policy, a security setting; theoretically this leaves you less safe, as it will allow you to run powershell without admin privileges... I think. [You can read about alternatives here on StackOverflow](https://stackoverflow.com/questions/18713086/virtualenv-wont-activate-on-windows).
```
Set-ExecutionPolicy Unrestricted -Force -Scope CurrentUser
```

## Setting up in vscode

I recommend the following plugins:
  - Python
  - Pyright (static type checker for python)
  - Error Lens (the one with the most stars: ``usernamehw.errorlens``)
