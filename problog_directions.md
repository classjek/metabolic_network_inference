# Using Problog Environment

Inside `/storage1/fs1/bjuba/Active/seas-lab-juba/Pathway_Pred/query` there is a tarball `problog_env.tar.gz`. Move this somewhere into your home directory and unpack it. 

```bash
mkdir ~/envs
cp problog_env.tar.gz ~/envs
tar -xzf problog_env.tar.gz
```

Then do the following command:
```bash
if [ -x ./problog/bin/conda-unpack ]; then
  ./problog/bin/conda-unpack
fi
```

At this point, you should be all set to use it. Quick sanity check:

```bash
 ~/envs/problog/bin/problog --version
```
Expected output: `2.2.9`

All done! Now you should be able to run Problog like this: 
```bash
~/envs/problog/bin/problog yourfile.pl
```

Here's an example
```bash
cat <<'PL' | ~/envs/problog/bin/problog -
0.6::edge(a,b).
0.7::edge(b,c).

path(X,Y) :- edge(X,Y).
path(X,Z) :- edge(X,Y), path(Y,Z).

query(path(a,c)).
PL
```

Expected output: `path(a,c):	0.42   `