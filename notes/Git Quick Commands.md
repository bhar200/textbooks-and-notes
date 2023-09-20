#programming
[[Programming]]



#### Add Commit and Push
```
git commit -a -m "commit" && git push
```

#### DS_Store

Remove all instances in dir
```
find . -name .DS_Store -print0 | xargs -0 git rm -f --ignore-unmatch
```

Add it to gitignore
```
echo .DS_Store >> .gitignore
```

