import languagemodels as lm
p=input("prompt: ")
res=lm.classify(p, "hard question", "easy question")
print(res)